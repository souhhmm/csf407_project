from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional


NEURAL_MODELS = ("mlp", "cnn", "gru", "transformer", "dnabert2", "dnabert2_scratch")
DEFAULT_MODELS = ("mlp", "cnn", "gru", "transformer")
PRETRAINED_MODELS = ("dnabert2", "dnabert2_scratch")
DNABERT_PATH = "data/models/dnabert2"
# One setting per model is compared on validation data; the rest stay fixed.
GRIDS = {
    "mlp": {"hidden": [1024, 2048]},
    "cnn": {"filters": [256, 512]},
    "gru": {"hidden": [256, 512]},
    "transformer": {"layers": [6, 8]},
    "dnabert2": {"lr": [3e-5, 1e-5]},
    "dnabert2_scratch": {"lr": [1e-4, 5e-5]},
}
FIXED = {
    "mlp": {"lr": 1e-3, "dropout": 0.3},
    "cnn": {"lr": 1e-3, "dropout": 0.2, "kernel": 15},
    "gru": {"lr": 1e-3, "dropout": 0.2},
    "transformer": {"lr": 3e-4, "dropout": 0.1, "d_model": 512, "heads": 8},
    "dnabert2": {"lr": 3e-5, "dropout": 0.1, "pad_id": 3, "warmup": 500, "decay": True},
    "dnabert2_scratch": {"lr": 1e-4, "dropout": 0.1, "pad_id": 3, "warmup": 1000, "decay": True},
}
UNKNOWN_CODE = 4
MASKED_SCORE = -1e4


def reverse_complement(codes: torch.Tensor) -> torch.Tensor:
    """Reverse the strand of encoded reads: read backwards and swap A<->T, C<->G.

    Args:
        codes: Letter codes (A=0, C=1, G=2, T=3, unknown 4), shape (reads, length).

    Returns:
        Codes of the opposite strand; unknown letters stay unknown.
    """
    flipped = torch.flip(codes, dims=[1])
    return torch.where(flipped < UNKNOWN_CODE, 3 - flipped, flipped)


def kmer_tokens(codes: torch.Tensor, k: int) -> torch.Tensor:
    """Turn encoded reads into overlapping word numbers.

    Args:
        codes: Letter codes, shape (reads, length).
        k: Number of letters in each word.

    Returns:
        Word numbers in the column order of features.all_words, shape
        (reads, length - k + 1). Windows with an unknown letter get 4**k,
        which the models treat as padding.
    """
    codes = codes.long()
    windows = codes.shape[1] - k + 1
    index = torch.zeros(codes.shape[0], windows, dtype=torch.long, device=codes.device)
    unknown = torch.zeros_like(index, dtype=torch.bool)
    for offset in range(k):
        letters = codes[:, offset : offset + windows]
        index = index * 4 + letters.clamp(max=3)
        unknown |= letters == UNKNOWN_CODE
    return index.masked_fill(unknown, 4**k)


def _masked_pool(states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Concatenate the mean and maximum of states over valid positions."""
    weights = valid.unsqueeze(-1).to(states.dtype)
    mean = (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    biggest = states.masked_fill(~valid.unsqueeze(-1), MASKED_SCORE).max(dim=1).values
    return torch.cat([mean, biggest], dim=1)


class WordMLP(nn.Module):
    """Two-layer perceptron over a table of weighted word counts."""

    def __init__(self, features: int, hidden: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, table: torch.Tensor) -> torch.Tensor:
        return self.layers(table).squeeze(-1)


class ReadCNN(nn.Module):
    """One-hot convolutional network in the style of DeePaC, but deeper.

    Four convolution blocks (batch norm, ReLU) with pooling, then the mean and
    the strongest response of every filter over the read. Unknown letters are
    all zeros.
    """

    def __init__(self, filters: int, kernel: int, dropout: float):
        super().__init__()

        def block(inputs: int, outputs: int, width: int, pool: bool) -> list:
            layers = [
                nn.Conv1d(inputs, outputs, width, padding=width // 2),
                nn.BatchNorm1d(outputs),
                nn.ReLU(),
            ]
            return layers + ([nn.MaxPool1d(2)] if pool else [])

        self.convolutions = nn.Sequential(
            *block(4, filters, kernel, True),
            *block(filters, filters, 9, True),
            *block(filters, filters, 5, False),
            *block(filters, filters, 5, False),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * filters, filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(filters, 1),
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        onehot = functional.one_hot(codes.long().clamp(max=UNKNOWN_CODE), 5)[..., :4]
        states = self.convolutions(
            onehot.transpose(1, 2).to(self.head[1].weight.dtype)
        )
        pooled = torch.cat([states.mean(dim=2), states.max(dim=2).values], dim=1)
        return self.head(pooled).squeeze(-1)


class WordGRU(nn.Module):
    """Bidirectional GRU over overlapping words, starting from learned word vectors."""

    def __init__(
        self,
        k: int,
        word_vectors: np.ndarray,
        hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.k = k
        vocabulary, width = word_vectors.shape
        self.embedding = nn.Embedding(vocabulary + 1, width, padding_idx=vocabulary)
        with torch.no_grad():
            self.embedding.weight[:vocabulary] = torch.from_numpy(word_vectors)
        self.recurrent = nn.GRU(
            width,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, 1))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        tokens = kmer_tokens(codes, self.k)
        states, _ = self.recurrent(self.embedding(tokens))
        pooled = _masked_pool(states, tokens != self.embedding.padding_idx)
        return self.head(pooled).squeeze(-1)


class WordTransformer(nn.Module):
    """Small transformer encoder over overlapping words, trained from scratch."""

    def __init__(
        self,
        k: int,
        length: int,
        layers: int,
        d_model: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.k = k
        vocabulary = 4**k
        self.embedding = nn.Embedding(vocabulary + 1, d_model, padding_idx=vocabulary)
        self.positions = nn.Embedding(length - k + 1, d_model)
        block = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d_model, 1))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        tokens = kmer_tokens(codes, self.k)
        valid = tokens != self.embedding.padding_idx
        # A read with no usable word would leave attention nothing to look at.
        blocked = ~valid
        blocked[~valid.any(dim=1), 0] = False
        places = torch.arange(tokens.shape[1], device=tokens.device)
        states = self.embedding(tokens) + self.positions(places)
        states = self.encoder(states, src_key_padding_mask=blocked)
        return self.head(_masked_pool(states, valid)).squeeze(-1)


class BertReader(nn.Module):
    """DNABERT-2 (117M, byte-pair tokens) with a mean-pooled linear read-out.

    The encoder starts from the published pretrained weights, or from random
    weights of the same architecture, so the two differ only in pretraining.
    """

    def __init__(self, pretrained: bool, dropout: float, pad_id: int):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.pad_id = pad_id
        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                DNABERT_PATH, trust_remote_code=True
            )
        else:
            config = AutoConfig.from_pretrained(DNABERT_PATH, trust_remote_code=True)
            self.encoder = AutoModel.from_config(config, trust_remote_code=True)
        width = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        valid = ids != self.pad_id
        states = self.encoder(input_ids=ids, attention_mask=valid.long())[0]
        weights = valid.unsqueeze(-1).to(states.dtype)
        pooled = (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def count_parameters(network: nn.Module) -> int:
    """Count the trainable values in a network."""
    return sum(parameter.numel() for parameter in network.parameters())


def make_network(
    model: str,
    params: Dict[str, float],
    k: Optional[int],
    features: int,
    length: int,
    word_vectors: Optional[np.ndarray],
) -> nn.Module:
    """Create an untrained network.

    Args:
        model: Model name from NEURAL_MODELS.
        params: Chosen grid values; fixed settings come from FIXED.
        k: Word size for the word-based models, or None.
        features: Columns of the count table, used by the perceptron.
        length: Read length, used for transformer positions.
        word_vectors: Learned word vectors, used by the GRU.

    Returns:
        The network, whose forward pass returns one score (a logit) per read.

    Raises:
        ValueError: If the model name is unknown or a required input is missing.
    """
    fixed = {**FIXED.get(model, {}), **params}
    if model == "mlp":
        return WordMLP(features, int(fixed["hidden"]), fixed["dropout"])
    if model == "cnn":
        return ReadCNN(int(fixed["filters"]), int(fixed["kernel"]), fixed["dropout"])
    if model == "gru":
        if word_vectors is None or k is None:
            raise ValueError("The GRU needs a word size and learned word vectors.")
        return WordGRU(k, word_vectors, int(fixed["hidden"]), fixed["dropout"])
    if model == "transformer":
        if k is None:
            raise ValueError("The transformer needs a word size.")
        return WordTransformer(
            k,
            length,
            int(fixed["layers"]),
            int(fixed["d_model"]),
            int(fixed["heads"]),
            fixed["dropout"],
        )
    if model in PRETRAINED_MODELS:
        return BertReader(model == "dnabert2", fixed["dropout"], int(fixed["pad_id"]))
    raise ValueError("Unknown neural model: " + model)
