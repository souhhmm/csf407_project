import contextlib
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import sparse
from torch import nn

from src.classifiers import fast_auc
from src.networks import reverse_complement


BATCH_EVALUATION = 1024
MIN_IMPROVEMENT = 1e-4


class TableInputs:
    """A sparse table of word counts, turned into dense batches on demand."""

    reverse_complement = False

    def __init__(self, table: sparse.csr_matrix, device: torch.device):
        self.table = sparse.csr_matrix(table, dtype=np.float32)
        self.device = device

    def __len__(self) -> int:
        return self.table.shape[0]

    def batch(self, indices: np.ndarray) -> torch.Tensor:
        """Return the rows at indices as a dense tensor on the device."""
        return torch.from_numpy(self.table[indices].toarray()).to(self.device)


class ReadInputs:
    """Encoded reads held on the device, which the networks read letter by letter."""

    reverse_complement = True

    def __init__(self, codes: np.ndarray, device: torch.device):
        self.device = device
        self.codes = torch.from_numpy(codes).to(device)

    def __len__(self) -> int:
        return self.codes.shape[0]

    def batch(self, indices: np.ndarray) -> torch.Tensor:
        """Return the reads at indices as letter codes on the device."""
        return self.codes[torch.from_numpy(indices).to(self.device)]


def select_device(name: str = "auto") -> torch.device:
    """Pick where to train.

    Args:
        name: auto (CUDA, then Apple MPS, then CPU), or cuda, mps, or cpu.

    Returns:
        The device.

    Raises:
        ValueError: If the requested device is unknown or unavailable.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name not in ("cuda", "mps", "cpu"):
        raise ValueError("Device must be auto, cuda, mps, or cpu.")
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if not available[name]:
        raise ValueError("The " + name + " device is not available here.")
    return torch.device(name)


def _autocast(device: torch.device):
    """Use bfloat16 arithmetic on CUDA only, where it is fast and well supported."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def predict_scores(
    network: nn.Module, inputs, device: torch.device, rc_average: bool = True
) -> np.ndarray:
    """Return each read's estimated probability of the positive class.

    Args:
        network: A trained network.
        inputs: TableInputs or ReadInputs.
        device: Where the network lives.
        rc_average: Whether to average the scores of a read and its reverse
            complement, so the answer does not depend on which strand was read.

    Returns:
        One probability per read.
    """
    network.eval()
    flip = rc_average and inputs.reverse_complement
    scores = []
    with torch.no_grad(), _autocast(device):
        for start in range(0, len(inputs), BATCH_EVALUATION):
            indices = np.arange(start, min(start + BATCH_EVALUATION, len(inputs)))
            batch = inputs.batch(indices)
            logits = network(batch).float()
            if flip:
                logits = (logits + network(reverse_complement(batch)).float()) / 2
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def train_network(
    network: nn.Module,
    train_inputs,
    train_labels: np.ndarray,
    val_inputs,
    val_labels: np.ndarray,
    lr: float,
    seed: int,
    device: torch.device,
    max_epochs: int = 15,
    patience: int = 3,
    batch_size: int = 256,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """Train a network, stopping when validation AUC stops improving.

    Only the validation reads decide when to stop and which weights to keep;
    the weights from the best validation epoch are restored before returning.

    Args:
        network: An untrained network on the device.
        train_inputs: TableInputs or ReadInputs for training.
        train_labels: Training labels, 1 for the positive class.
        val_inputs: Matching inputs for validation.
        val_labels: Validation labels.
        lr: Learning rate for AdamW.
        seed: Fixed random setting for batch order and strand flips.
        device: Where the network lives.
        max_epochs: Upper limit on passes over the training reads.
        patience: Epochs without improvement before stopping.
        batch_size: Reads per update.

    Returns:
        The network with its best weights, and one history row per epoch
        holding the epoch number, mean training loss, validation AUC, and
        whether that epoch was the best so far.
    """
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=0.01)
    targets = torch.from_numpy(train_labels.astype(np.float32)).to(device)
    best_auc, best_state, waiting = -np.inf, None, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        network.train()
        started = time.perf_counter()
        order = rng.permutation(len(train_inputs))
        total = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = train_inputs.batch(indices)
            if train_inputs.reverse_complement:
                flip = (torch.rand(len(indices), generator=generator) < 0.5).to(device)
                batch = torch.where(flip.unsqueeze(1), reverse_complement(batch), batch)
            with _autocast(device):
                logits = network(batch).float()
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets[torch.from_numpy(indices).to(device)]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        auc = fast_auc(val_labels, predict_scores(network, val_inputs, device))
        improved = auc > best_auc + MIN_IMPROVEMENT
        if improved:
            best_auc, waiting = auc, 0
            best_state = {
                name: value.detach().clone()
                for name, value in network.state_dict().items()
            }
        else:
            waiting += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(order),
                "val_auc": auc,
                "best_so_far": improved,
                "seconds": time.perf_counter() - started,
            }
        )
        if waiting >= patience:
            break
    if best_state is not None:
        network.load_state_dict(best_state)
    return network, history


def seed_everything(seed: int, threads: Optional[int] = None):
    """Fix torch's random numbers and, optionally, its CPU thread count."""
    torch.manual_seed(seed)
    if threads is not None:
        torch.set_num_threads(threads)
