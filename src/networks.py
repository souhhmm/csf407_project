from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional


NEURAL_MODELS = ("mlp", "cnn", "gru", "transformer")
# One setting per model is compared on validation data; the rest stay fixed.
GRIDS = {
    "mlp": {"hidden": [128, 512]},
    "cnn": {"filters": [64, 128]},
    "gru": {"hidden": [64, 128]},
    "transformer": {"layers": [2, 4]},
}
FIXED = {
    "mlp": {"lr": 1e-3, "dropout": 0.3},
    "cnn": {"lr": 1e-3, "dropout": 0.3, "kernel": 15},
    "gru": {"lr": 1e-3, "dropout": 0.3},
    "transformer": {"lr": 5e-4, "dropout": 0.1, "d_model": 128, "heads": 4},
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
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, table: torch.Tensor) -> torch.Tensor:
        return self.layers(table).squeeze(-1)


class ReadCNN(nn.Module):
    """One-hot convolutional network in the style of DeePaC.

    Filters slide along the read and the strongest response of each filter is
    kept, so a motif counts wherever it sits. Unknown letters are all zeros.
    """

    def __init__(self, filters: int, kernel: int, dropout: float):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(4, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(filters, filters, 7, padding=3),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(filters, filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(filters, 1),
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        onehot = functional.one_hot(codes.long().clamp(max=UNKNOWN_CODE), 5)[..., :4]
        states = self.convolutions(onehot.transpose(1, 2).to(self.head[1].weight.dtype))
        return self.head(states.max(dim=2).values).squeeze(-1)


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
        self.recurrent = nn.GRU(width, hidden, batch_first=True, bidirectional=True)
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
    raise ValueError("Unknown neural model: " + model)
