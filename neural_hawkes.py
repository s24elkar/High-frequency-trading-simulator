#!/usr/bin/env python3
"""Training script for a neural Hawkes surrogate with PyTorch.

The script expects preprocessed event sequences of the form
[(t_0, type_0), (t_1, type_1), ...] where timestamps are expressed in
seconds. If no dataset path is provided, synthetic sequences are
generated so the script can be executed out-of-the-box.

Workflow
========
1. Load sequences (either from disk or synthetic generator).
2. Build a Dataset that emits sliding windows, pads variable-length
   windows, and provides masks for valid steps.
3. Define a neural Hawkes model backed by a GRU/LSTM or feedforward MLP backbone with two heads:
   (a) next-event type logits and (b) next inter-arrival time regression.
4. Train with a surrogate objective: cross-entropy for event types and
   mean-squared error for the inter-arrival targets.
5. Evaluate on held-out data, reporting proxy log-likelihood (average
   loss), event-type accuracy, and MAE of inter-arrival predictions.
6. Compare runtime on CPU vs GPU (when available) and print simple
   training curves.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------


@dataclass
class EventSequence:
    """Container for a single event sequence."""

    times: np.ndarray  # shape (N,)
    types: np.ndarray  # shape (N,)

    def __post_init__(self) -> None:
        if self.times.ndim != 1 or self.types.ndim != 1:
            raise ValueError("times and types must be 1-D arrays")
        if self.times.shape[0] != self.types.shape[0]:
            raise ValueError("times and types must have equal length")
        if not np.all(np.diff(self.times) >= 0):
            raise ValueError("timestamps must be non-decreasing")


class EventSequenceDataset(Dataset):
    """Sliding-window dataset that returns padded sequences with masks."""

    def __init__(
        self,
        sequences: List[EventSequence],
        window_size: int = 64,
        stride: int = 32,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.sequences = sequences
        self.window_size = window_size
        self.stride = stride
        self._index: List[Tuple[int, int, int]] = []  # (seq_id, start, end)
        self._build_index()

    def _build_index(self) -> None:
        for seq_id, seq in enumerate(self.sequences):
            length = seq.times.shape[0]
            if length < 2:
                continue
            start = 0
            while start < length - 1:
                end = min(start + self.window_size, length)
                if end - start >= 2:
                    self._index.append((seq_id, start, end))
                if end == length:
                    break
                start += self.stride
        if not self._index:
            raise ValueError("No valid windows produced; check window_size/stride")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        seq_id, start, end = self._index[idx]
        seq = self.sequences[seq_id]
        times = seq.times[start:end].astype(np.float32)
        types = seq.types[start:end].astype(np.int64)
        inter_arr = np.zeros_like(times, dtype=np.float32)
        inter_arr[1:] = times[1:] - times[:-1]
        item = {
            "times": torch.from_numpy(times),
            "types": torch.from_numpy(types),
            "inter_arr": torch.from_numpy(inter_arr),
        }
        return item


def collate_windows(batch: List[dict]) -> dict:
    batch_size = len(batch)
    max_len = max(item["types"].size(0) for item in batch)
    types = torch.zeros(batch_size, max_len, dtype=torch.long)
    deltas = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        length = item["types"].size(0)
        types[i, :length] = item["types"]
        deltas[i, :length] = item["inter_arr"]
        mask[i, :length] = True

    # Inputs exclude the final event; targets are the subsequent events.
    input_types = types[:, :-1]
    input_deltas = deltas[:, :-1]
    target_types = types[:, 1:]
    target_deltas = deltas[:, 1:]
    target_mask = mask[:, 1:]
    lengths = target_mask.sum(dim=1)

    return {
        "input_types": input_types,
        "input_deltas": input_deltas,
        "target_types": target_types,
        "target_deltas": target_deltas,
        "target_mask": target_mask,
        "lengths": lengths,
    }


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------


class NeuralHawkesModel(nn.Module):
    """Neural surrogate for Hawkes dynamics with configurable backbone."""

    def __init__(
        self,
        num_types: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        backbone: str = "gru",
        mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_types = num_types
        self.backbone = backbone.lower()
        self.type_embed = nn.Embedding(num_types, embed_dim)
        input_dim = embed_dim + 1

        if self.backbone == "mlp":
            layers: List[nn.Module] = []
            in_dim = input_dim
            depth = max(1, mlp_layers)
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            self.mlp = nn.Sequential(*layers)
        elif self.backbone == "transformer":
            self.delta_proj = nn.Linear(1, embed_dim)
            for candidate in (8, 6, 4, 2, 1):
                if embed_dim % candidate == 0:
                    nhead = candidate
                    break
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                dropout=0.1,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, mlp_layers))
            self.post_proj = nn.Linear(embed_dim, hidden_dim)
        else:
            rnn_cls = nn.LSTM if self.backbone == "lstm" else nn.GRU
            self.rnn = rnn_cls(input_dim, hidden_dim, batch_first=True)

        if self.backbone == "transformer":
            self.register_buffer(
                "pos_weights",
                torch.arange(0, 512, dtype=torch.float32).unsqueeze(0),
                persistent=False,
            )
        else:
            self.pos_weights = None

        self.type_head = nn.Linear(hidden_dim, num_types)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(
        self,
        input_types: torch.Tensor,
        input_deltas: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_types: (B, T), input_deltas: (B, T)
        emb = self.type_embed(input_types)
        delta_feat = input_deltas.unsqueeze(-1)
        x = torch.cat([emb, delta_feat], dim=-1)

        if self.backbone == "mlp":
            out = self.mlp(x)
        elif self.backbone == "transformer":
            delta_embed = self.delta_proj(delta_feat)
            transformer_in = emb + delta_embed
            out = self.transformer(transformer_in)
            out = self.post_proj(out)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=input_types.size(1)
            )
        type_logits = self.type_head(out)
        delta_pred = self.delta_head(out).squeeze(-1)
        return type_logits, delta_pred


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def compute_losses(
    logits: torch.Tensor,
    delta_pred: torch.Tensor,
    targets_type: torch.Tensor,
    targets_delta: torch.Tensor,
    mask: torch.Tensor,
    delta_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = mask
    if valid.sum() == 0:
        zero = logits.new_tensor(0.0)
        return zero, zero, zero
    flat_logits = logits[valid]
    flat_types = targets_type[valid]
    flat_deltas = targets_delta[valid]
    type_loss = F.cross_entropy(flat_logits, flat_types)
    delta_loss = F.mse_loss(delta_pred[valid], flat_deltas)
    total_loss = type_loss + delta_weight * delta_loss
    return total_loss, type_loss, delta_loss


@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    delta_pred: torch.Tensor,
    targets_type: torch.Tensor,
    targets_delta: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    valid = mask
    if valid.sum() == 0:
        return 0.0, 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == targets_type) & valid
    accuracy = correct.sum().float() / valid.sum().float()
    mae = torch.abs(delta_pred[valid] - targets_delta[valid]).mean()
    return accuracy.item(), mae.item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    delta_weight: float,
) -> dict:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_mse = 0.0
    total_steps = 0
    total_correct = 0
    total_valid = 0
    total_abs_err = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        logits, delta_pred = model(
            batch["input_types"], batch["input_deltas"], batch["lengths"]
        )
        loss, ce_loss, mse_loss = compute_losses(
            logits,
            delta_pred,
            batch["target_types"],
            batch["target_deltas"],
            batch["target_mask"],
            delta_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        valid = batch["target_mask"]
        valid_count = valid.sum().item()
        if valid_count > 0:
            preds = logits.argmax(dim=-1)
            total_correct += ((preds == batch["target_types"]) & valid).sum().item()
            total_abs_err += torch.abs(
                delta_pred[valid] - batch["target_deltas"][valid]
            ).sum().item()
            total_valid += valid_count

        total_loss += loss.item() * valid_count
        total_ce += ce_loss.item() * valid_count
        total_mse += mse_loss.item() * valid_count
        total_steps += valid_count

    avg_loss = total_loss / max(total_steps, 1)
    avg_ce = total_ce / max(total_steps, 1)
    avg_mse = total_mse / max(total_steps, 1)
    acc = total_correct / max(total_valid, 1)
    mae = total_abs_err / max(total_valid, 1)
    return {
        "loss": avg_loss,
        "ce": avg_ce,
        "mse": avg_mse,
        "acc": acc,
        "mae": mae,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, delta_weight: float) -> dict:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    total_correct = 0
    total_valid = 0
    total_abs_err = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        logits, delta_pred = model(
            batch["input_types"], batch["input_deltas"], batch["lengths"]
        )
        loss, _, _ = compute_losses(
            logits,
            delta_pred,
            batch["target_types"],
            batch["target_deltas"],
            batch["target_mask"],
            delta_weight,
        )
        valid = batch["target_mask"]
        valid_count = valid.sum().item()
        if valid_count > 0:
            preds = logits.argmax(dim=-1)
            total_correct += ((preds == batch["target_types"]) & valid).sum().item()
            total_abs_err += torch.abs(
                delta_pred[valid] - batch["target_deltas"][valid]
            ).sum().item()
            total_valid += valid_count
            total_loss += loss.item() * valid_count
            total_steps += valid_count

    avg_loss = total_loss / max(total_steps, 1)
    acc = total_correct / max(total_valid, 1)
    mae = total_abs_err / max(total_valid, 1)
    return {
        "loss": avg_loss,
        "acc": acc,
        "mae": mae,
    }


@torch.no_grad()
def measure_runtime(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup: int = 1,
) -> float:
    model.eval()
    model.to(device)
    # Warm-up iterations (important for CUDA context init)
    for _ in range(warmup):
        for batch in loader:
            batch = move_batch(batch, device)
            model(batch["input_types"], batch["input_deltas"], batch["lengths"])
        break
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            model(batch["input_types"], batch["input_deltas"], batch["lengths"])
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    return time.time() - start


# -----------------------------------------------------------------------------
# Diagnostics & experiment orchestration
# -----------------------------------------------------------------------------


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds: List[torch.Tensor] = []
    trues: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for batch in loader:
        batch = move_batch(batch, device)
        _, delta_pred = model(batch["input_types"], batch["input_deltas"], batch["lengths"])
        preds.append(delta_pred.detach().cpu())
        trues.append(batch["target_deltas"].detach().cpu())
        masks.append(batch["target_mask"].detach().cpu())
    pred_arr = torch.cat(preds, dim=0).numpy()
    true_arr = torch.cat(trues, dim=0).numpy()
    mask_arr = torch.cat(masks, dim=0).numpy()
    return pred_arr, true_arr, mask_arr


def time_rescaling_diagnostics(
    predicted_deltas: np.ndarray,
    true_deltas: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-6,
    max_points: int = 200,
) -> Dict[str, Any]:
    valid = mask.astype(bool)
    if not np.any(valid):
        return {
            "sample_size": 0,
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "mean_rescaled": 0.0,
            "var_rescaled": 0.0,
            "qq_theoretical": [],
            "qq_empirical": [],
        }
    preds = np.clip(predicted_deltas[valid], eps, None)
    trues = true_deltas[valid]
    rescaled = trues / preds
    rescaled = rescaled[rescaled >= 0]
    if rescaled.size == 0:
        return {
            "sample_size": 0,
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "mean_rescaled": 0.0,
            "var_rescaled": 0.0,
            "qq_theoretical": [],
            "qq_empirical": [],
        }

    rescaled.sort()
    n = rescaled.size
    empirical = np.arange(1, n + 1) / n
    theoretical = 1.0 - np.exp(-rescaled)
    ks_stat = float(np.max(np.abs(empirical - theoretical)))
    ks_pvalue = float(min(1.0, 2.0 * np.exp(-2.0 * n * ks_stat**2)))
    mean_rescaled = float(rescaled.mean())
    var_rescaled = float(rescaled.var())

    points = min(n, max_points)
    idx = np.linspace(0, n - 1, points, dtype=int)
    probs = (idx + 0.5) / n
    qq_theoretical = (-np.log(1.0 - probs)).tolist()
    qq_empirical = rescaled[idx].tolist()

    return {
        "sample_size": int(n),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "mean_rescaled": mean_rescaled,
        "var_rescaled": var_rescaled,
        "qq_theoretical": qq_theoretical,
        "qq_empirical": qq_empirical,
    }


def build_model_from_config(
    num_types: int,
    training_cfg: Dict[str, Any],
) -> NeuralHawkesModel:
    return NeuralHawkesModel(
        num_types=num_types,
        embed_dim=training_cfg.get("embed_dim", 32),
        hidden_dim=training_cfg.get("hidden_dim", 64),
        backbone=training_cfg.get("backbone", "gru"),
        mlp_layers=training_cfg.get("mlp_layers", 2),
    )


def run_experiment(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    config = json.loads(json.dumps(config))
    training_cfg = config.setdefault("training", {})
    dataset_cfg = config.get("dataset")
    synthetic_cfg = config.get("synthetic", {})
    seed = config.get("seed", training_cfg.get("seed", 2024))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        use_gpu = torch.cuda.is_available() and not config.get("force_cpu", False) and not training_cfg.get("force_cpu", False)
        device = torch.device("cuda" if use_gpu else "cpu")
    else:
        use_gpu = device.type == "cuda"

    sequences: List[EventSequence] = []
    num_types = int(config.get("num_types", synthetic_cfg.get("num_types", training_cfg.get("num_types", 4))))

    if dataset_cfg:
        sequences = load_sequences_from_path(Path(dataset_cfg))
        num_types = max(num_types, max(int(seq.types.max()) for seq in sequences) + 1)
    elif config.get("symbols"):
        for idx, spec in enumerate(config["symbols"]):
            if not isinstance(spec, dict):
                spec = {"symbol": str(spec)}
            symbol_seed = spec.get("seed", seed + idx)
            if "dataset" in spec:
                seqs = load_sequences_from_path(Path(spec["dataset"]))
                sequences.extend(seqs)
                num_types = max(num_types, max(int(seq.types.max()) for seq in seqs) + 1)
            else:
                synth = spec.get("synthetic", {})
                seqs = generate_synthetic_sequences(
                    num_sequences=synth.get("num_sequences", synthetic_cfg.get("num_sequences", 200)),
                    num_events=synth.get("num_events", synthetic_cfg.get("num_events", 400)),
                    num_types=synth.get("num_types", num_types),
                    seed=symbol_seed,
                )
                sequences.extend(seqs)
    else:
        sequences = generate_synthetic_sequences(
            num_sequences=synthetic_cfg.get("num_sequences", 200),
            num_events=synthetic_cfg.get("num_events", 400),
            num_types=synthetic_cfg.get("num_types", num_types),
            seed=synthetic_cfg.get("seed", seed),
        )

    dataset = EventSequenceDataset(
        sequences,
        window_size=training_cfg.get("window_size", 64),
        stride=training_cfg.get("stride", 32),
    )
    train_set, val_set, test_set = split_dataset(
        dataset, tuple(training_cfg.get("split", (0.7, 0.15, 0.15)))
    )

    collate = collate_windows
    train_loader = DataLoader(
        train_set,
        batch_size=training_cfg.get("batch_size", 64),
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 64)),
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 64)),
        shuffle=False,
        collate_fn=collate,
    )

    model = build_model_from_config(num_types, training_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.get("lr", 1e-3))
    epochs = int(training_cfg.get("epochs", 10))
    delta_weight = training_cfg.get("delta_weight", 1.0)

    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, delta_weight)
        val_metrics = evaluate(model, val_loader, device, delta_weight)
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        if training_cfg.get("verbose", True):
            print(
                f"Epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} "
                f"acc {train_metrics['acc']:.4f} mae {train_metrics['mae']:.4f} || "
                f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} mae {val_metrics['mae']:.4f}"
            )

    duration = time.time() - start_time
    test_metrics = evaluate(model, test_loader, device, delta_weight)

    rescaling: Dict[str, Any]
    try:
        pred_deltas, true_deltas, mask = collect_predictions(model, test_loader, device)
        rescaling = time_rescaling_diagnostics(pred_deltas, true_deltas, mask)
    except Exception as exc:  # pragma: no cover
        rescaling = {"error": str(exc)}

    runtime_stats: Dict[str, Any] = {}
    if training_cfg.get("measure_runtime", True):
        cpu_model = build_model_from_config(num_types, training_cfg)
        cpu_model.load_state_dict(model.state_dict())
        cpu_time = measure_runtime(cpu_model.to(torch.device("cpu")), test_loader, torch.device("cpu"))
        runtime_stats["cpu_seconds"] = cpu_time
        if use_gpu:
            gpu_model = build_model_from_config(num_types, training_cfg)
            gpu_model.load_state_dict(model.state_dict())
            gpu_time = measure_runtime(gpu_model.to(torch.device("cuda")), test_loader, torch.device("cuda"))
            runtime_stats["gpu_seconds"] = gpu_time

    result = {
        "name": config.get("name", "experiment"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "duration_sec": duration,
        "config": config,
        "train_history": train_history,
        "val_history": val_history,
        "test_metrics": test_metrics,
        "rescaling": rescaling,
        "runtime": runtime_stats,
        "device": device.type,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as fh:
            json.dump(result, fh, indent=2)

    return result


# -----------------------------------------------------------------------------
# Data loading / generation
# -----------------------------------------------------------------------------


def load_sequences_from_path(path: Path) -> List[EventSequence]:
    """Attempts to load sequences from JSON or NPZ; falls back otherwise."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset path {path} not found")
    if path.suffix == ".json":
        with path.open() as fh:
            payload = json.load(fh)
        sequences = []
        for seq in payload:
            arr = np.asarray(seq, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("JSON sequences must be list of [time, type]")
            sequences.append(EventSequence(times=arr[:, 0], types=arr[:, 1].astype(np.int64)))
        return sequences
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "times" in data and "types" in data:
            times_list = data["times"]
            types_list = data["types"]
            sequences = []
            for times, types in zip(times_list, types_list):
                sequences.append(EventSequence(times=np.asarray(times, dtype=np.float32), types=np.asarray(types, dtype=np.int64)))
            return sequences
        if "sequences" in data:
            sequences = []
            for seq in data["sequences"]:
                arr = np.asarray(seq, dtype=np.float32)
                sequences.append(EventSequence(times=arr[:, 0], types=arr[:, 1].astype(np.int64)))
            return sequences
        raise ValueError("NPZ file must contain 'times'/'types' arrays or 'sequences'")
    raise ValueError("Only JSON and NPZ formats are supported for dataset loading")


def generate_synthetic_sequences(
    num_sequences: int = 200,
    num_events: int = 400,
    num_types: int = 4,
    seed: int = 42,
) -> List[EventSequence]:
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(num_sequences):
        length = max(50, int(rng.normal(num_events, num_events * 0.1)))
        inter_arrivals = rng.exponential(scale=0.5, size=length).astype(np.float32)
        times = np.cumsum(inter_arrivals)
        types = rng.integers(low=0, high=num_types, size=length, dtype=np.int64)
        sequences.append(EventSequence(times=times, types=types))
    return sequences


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def split_dataset(dataset: EventSequenceDataset, ratios: Tuple[float, float, float]) -> Tuple[torch.utils.data.Dataset, ...]:
    assert math.isclose(sum(ratios), 1.0, rel_tol=1e-5)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_end = int(len(indices) * ratios[0])
    val_end = train_end + int(len(indices) * ratios[1])
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx),
    )


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neural Hawkes training script")
    parser.add_argument("--dataset", type=str, default="", help="Path to JSON/NPZ dataset")
    parser.add_argument("--num-types", type=int, default=4, help="Number of distinct event types")
    parser.add_argument("--window-size", type=int, default=64, help="Sliding window length")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride")
    parser.add_argument("--embed-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden size for backbone")
    parser.add_argument(
        "--backbone",
        type=str,
        default="gru",
        choices=["gru", "lstm", "mlp"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--mlp-layers", type=int, default=2, help="Number of layers when using the MLP backbone"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch count")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--delta-weight", type=float, default=1.0, help="Weight for inter-arrival loss")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")
    parser.add_argument("--skip-runtime", action="store_true", help="Disable runtime benchmarking")
    parser.add_argument("--output", type=str, default="", help="Optional JSON path for metrics")
    return parser


def main() -> None:
    parser = create_argparser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset:
        try:
            sequences = load_sequences_from_path(Path(args.dataset))
            num_types = args.num_types or max(int(seq.types.max()) for seq in sequences) + 1
        except Exception as exc:
            print(f"Failed to load dataset ({exc}); falling back to synthetic data")
            sequences = generate_synthetic_sequences(num_types=args.num_types)
            num_types = args.num_types
    else:
        sequences = generate_synthetic_sequences(num_types=args.num_types)
        num_types = args.num_types

    config = {
        "name": "cli_run",
        "dataset": args.dataset or None,
        "num_types": args.num_types,
        "seed": args.seed,
        "force_cpu": args.cpu,
        "synthetic": {
            "num_sequences": 200,
            "num_events": 400,
            "num_types": args.num_types,
            "seed": args.seed,
        } if not args.dataset else {},
        "training": {
            "window_size": args.window_size,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "delta_weight": args.delta_weight,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "backbone": args.backbone,
            "mlp_layers": args.mlp_layers,
            "measure_runtime": not args.skip_runtime,
        },
    }

    output_path = Path(args.output) if args.output else None
    result = run_experiment(config, device=None, output_path=output_path)

    print("\nTest set metrics:")
    print(json.dumps(result["test_metrics"], indent=2))
    if result.get("runtime"):
        print("\nRuntime benchmarks:")
        print(json.dumps(result["runtime"], indent=2))

    print("\nRescaling diagnostics:")
    print(json.dumps(result.get("rescaling", {}), indent=2))


if __name__ == "__main__":
    main()
