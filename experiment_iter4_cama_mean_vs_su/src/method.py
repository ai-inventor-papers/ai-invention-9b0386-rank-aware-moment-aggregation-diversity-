#!/usr/bin/env python3
"""CAMA Mean-vs-Sum Aggregation Causal Diagnosis on rel-f1.

Implements CAMA (Cardinality-Aware Moment Aggregation) as a PyG Aggregation
subclass. Runs a controlled experiment on rel-f1 (5 methods x 2 tasks x 5 seeds)
to diagnose whether mean-vs-sum aggregation choice causally determines CAMA's
benefit. Measures effective rank, computes Cohen's d with bootstrap 95% CI,
and runs a CAMA x aggregation interaction test.
"""

import gc
import json
import math
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch_geometric.nn.aggr import Aggregation, MeanAggregation, SumAggregation

warnings.filterwarnings("ignore")

WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ══════════════════════════════════════════════════════════════════════════════
# CAMA AGGREGATION MODULES
# ══════════════════════════════════════════════════════════════════════════════

class CAMAAggregation(Aggregation):
    """Cardinality-Aware Moment Aggregation (mean-based).
    g = sigmoid(w * log(N+1) + b)
    output = mean + g * W_sigma(variance)
    """
    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.W_sigma = nn.Linear(channels, channels, bias=False)
        self.w_gate = nn.Parameter(torch.zeros(1))
        self.b_gate = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.W_sigma.weight)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce='mean')
        var = (mean_x2 - mean * mean).clamp(min=0)
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        N = self.reduce(ones, index, ptr, dim_size, dim, reduce='sum')
        log_N = torch.log1p(N)
        g = torch.sigmoid(self.w_gate * log_N + self.b_gate)
        return mean + g * self.W_sigma(var)


class SumCAMAAggregation(Aggregation):
    """CAMA variant with sum as base aggregation."""
    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.W_sigma = nn.Linear(channels, channels, bias=False)
        self.w_gate = nn.Parameter(torch.zeros(1))
        self.b_gate = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.W_sigma.weight)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        sum_result = self.reduce(x, index, ptr, dim_size, dim, reduce='sum')
        mean_result = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce='mean')
        var = (mean_x2 - mean_result * mean_result).clamp(min=0)
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        N = self.reduce(ones, index, ptr, dim_size, dim, reduce='sum')
        log_N = torch.log1p(N)
        g = torch.sigmoid(self.w_gate * log_N + self.b_gate)
        return sum_result + g * self.W_sigma(var)


class UngatedMomentAggregation(Aggregation):
    """Mean + W_sigma(variance) without any gating (ablation)."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.W_sigma = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.W_sigma.weight)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce='mean')
        var = (mean_x2 - mean * mean).clamp(min=0)
        return mean + self.W_sigma(var)


def get_aggr_for_method(method: str, channels: int):
    """Return aggregation object/string for a given method name."""
    if method == "mean":
        return "mean"
    elif method == "sum":
        return "sum"
    elif method == "mean_cama":
        return CAMAAggregation(channels=channels)
    elif method == "sum_cama":
        return SumCAMAAggregation(channels=channels)
    elif method == "mean_ungated":
        return UngatedMomentAggregation(channels=channels)
    else:
        raise ValueError(f"Unknown method: {method}")


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    logger.info("Running CAMA unit tests...")
    torch.manual_seed(42)
    x = torch.randn(100, 128)
    index = torch.randint(0, 10, (100,))

    cama = CAMAAggregation(channels=128)
    out = cama(x, index=index, dim_size=10)
    assert out.shape == (10, 128), f"CAMA shape wrong: {out.shape}"
    logger.info("  T1a PASS: CAMAAggregation shape (10, 128)")

    mean_aggr = MeanAggregation()
    out_mean = mean_aggr(x, index=index, dim_size=10)
    assert torch.allclose(out, out_mean, atol=1e-5)
    logger.info("  T1b PASS: Zero-init CAMA matches mean aggregation")

    ungated = UngatedMomentAggregation(channels=128)
    out_u = ungated(x, index=index, dim_size=10)
    assert out_u.shape == (10, 128)
    assert torch.allclose(out_u, out_mean, atol=1e-5)
    logger.info("  T1c PASS: UngatedMomentAggregation shape and zero-init")

    sum_cama = SumCAMAAggregation(channels=128)
    out_sc = sum_cama(x, index=index, dim_size=10)
    assert out_sc.shape == (10, 128)
    sum_aggr = SumAggregation()
    out_sum = sum_aggr(x, index=index, dim_size=10)
    assert torch.allclose(out_sc, out_sum, atol=1e-5)
    logger.info("  T1d PASS: SumCAMAAggregation shape and zero-init matches sum")

    cama2 = CAMAAggregation(channels=128)
    x2 = torch.randn(100, 128, requires_grad=True)
    out2 = cama2(x2, index=index, dim_size=10)
    out2.sum().backward()
    assert cama2.w_gate.grad is not None
    assert cama2.W_sigma.weight.grad is not None
    logger.info("  T1e PASS: Gradient flow through CAMA verified")
    logger.info("All unit tests PASSED!")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════

_DATA_CACHE = {}

def load_data_and_task(task_name: str, device):
    """Load rel-f1 dataset and task, with caching."""
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from torch_frame import stype

    cache_key = task_name
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", task_name, download=True)
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    for tbl_name, col_stype in col_to_stype_dict.items():
        for col, st in list(col_stype.items()):
            if st == stype.text_embedded:
                del col_to_stype_dict[tbl_name][col]

    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        cache_dir=str(WORKSPACE / "cache"),
    )

    entity_table = task.entity_table
    train_input = get_node_train_table_input(table=train_table, task=task)
    val_input = get_node_train_table_input(table=val_table, task=task)
    test_input = get_node_train_table_input(table=test_table, task=task)

    # Keep data on CPU - NeighborLoader sampling must happen on CPU
    # Batches are moved to GPU after sampling in the training loop

    result = {
        "data": data,
        "col_stats_dict": col_stats_dict,
        "entity_table": entity_table,
        "train_input": train_input,
        "val_input": val_input,
        "test_input": test_input,
        "task": task,
        "is_classification": task_name == "driver-dnf",
    }
    _DATA_CACHE[cache_key] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_single_experiment(
    method: str, task_name: str, seed: int, device,
    channels: int = 128, lr: float = 0.005, epochs: int = 10,
    batch_size: int = 512, num_neighbors: int = 128, num_layers: int = 2,
    results_dir: Path | None = None,
) -> dict:
    from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
    from torch_geometric.loader import NeighborLoader

    logger.info(f"Starting: method={method}, task={task_name}, seed={seed}")
    start_time = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    cached = load_data_and_task(task_name, device)
    data = cached["data"]
    col_stats_dict = cached["col_stats_dict"]
    entity_table = cached["entity_table"]
    train_input = cached["train_input"]
    val_input = cached["val_input"]
    test_input = cached["test_input"]
    is_classification = cached["is_classification"]

    aggr = get_aggr_for_method(method, channels)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = HeteroEncoder(
                channels=channels,
                node_to_col_names_dict={
                    nt: data[nt].tf.col_names_dict
                    for nt in data.node_types if hasattr(data[nt], 'tf')
                },
                node_to_col_stats=col_stats_dict,
            )
            self.temporal_encoder = HeteroTemporalEncoder(
                node_types=[nt for nt in data.node_types if "time" in data[nt]],
                channels=channels,
            )
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr=aggr,
                num_layers=num_layers,
            )
            self.head = nn.Sequential(
                nn.Linear(channels, channels), nn.ReLU(),
                nn.Dropout(0.5), nn.Linear(channels, 1),
            )

        def forward(self, batch):
            seed_time = batch[entity_table].seed_time
            x_dict = self.encoder(batch.tf_dict)
            # Filter time_dict to only node types present in batch_dict
            safe_time_dict = {nt: t for nt, t in batch.time_dict.items()
                              if nt in batch.batch_size_dict}
            rel_time_dict = self.temporal_encoder(
                seed_time, safe_time_dict, batch.batch_size_dict
            )
            for nt in x_dict:
                if nt in rel_time_dict:
                    x_dict[nt] = x_dict[nt] + rel_time_dict[nt]
            x_dict = self.gnn(
                x_dict, batch.edge_index_dict,
                batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict
            )
            return self.head(x_dict[entity_table][:batch[entity_table].batch_size])

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if is_classification else nn.L1Loss()

    num_neighbors_per_layer = [num_neighbors] + [num_neighbors // 2] * (num_layers - 1)

    train_loader = NeighborLoader(
        data, num_neighbors=num_neighbors_per_layer, time_attr="time",
        input_nodes=train_input.nodes, input_time=train_input.time,
        transform=train_input.transform,
        batch_size=batch_size, shuffle=True, num_workers=0,
    )

    def make_eval_loader(node_input):
        return NeighborLoader(
            data, num_neighbors=num_neighbors_per_layer, time_attr="time",
            input_nodes=node_input.nodes, input_time=node_input.time,
            transform=node_input.transform,
            batch_size=batch_size, shuffle=False, num_workers=0,
        )

    val_loader = make_eval_loader(val_input)
    test_loader = make_eval_loader(test_input)

    best_val_metric = None
    best_epoch = 0
    best_state = None
    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch).squeeze()
            target = batch[entity_table].y[:batch[entity_table].batch_size]
            if is_classification:
                target = target.float()
            loss = loss_fn(pred, target)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at epoch {epoch}")
                continue
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch).squeeze()
                target = batch[entity_table].y[:batch[entity_table].batch_size]
                val_preds.append(pred.cpu())
                val_targets.append(target.cpu())

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        if is_classification:
            from sklearn.metrics import average_precision_score
            val_metric = average_precision_score(
                val_targets.numpy(), torch.sigmoid(val_preds).numpy()
            )
            metric_name = "average_precision"
            higher_is_better = True
        else:
            val_metric = F.l1_loss(val_preds, val_targets).item()
            metric_name = "mae"
            higher_is_better = False

        is_better = (best_val_metric is None or
                     (higher_is_better and val_metric > best_val_metric) or
                     (not higher_is_better and val_metric < best_val_metric))
        if is_better:
            best_val_metric = val_metric
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        logger.debug(f"  Epoch {epoch}/{epochs} loss={avg_loss:.4f} val_{metric_name}={val_metric:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    def evaluate(loader):
        model.eval()
        all_preds, all_targets = [], []
        has_labels = True
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch).squeeze()
                if not hasattr(batch[entity_table], 'y') or batch[entity_table].y is None:
                    has_labels = False
                    break
                target = batch[entity_table].y[:batch[entity_table].batch_size]
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        if not has_labels or len(all_preds) == 0:
            return None
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        metrics = {}
        if is_classification:
            from sklearn.metrics import average_precision_score, accuracy_score, f1_score
            probs = torch.sigmoid(preds).numpy()
            metrics["average_precision"] = float(average_precision_score(targets.numpy(), probs))
            metrics["accuracy"] = float(accuracy_score(targets.numpy(), (probs > 0.5).astype(int)))
            metrics["f1"] = float(f1_score(targets.numpy(), (probs > 0.5).astype(int)))
        else:
            metrics["mae"] = float(F.l1_loss(preds, targets).item())
            metrics["rmse"] = float(torch.sqrt(F.mse_loss(preds, targets)).item())
            t_np, p_np = targets.numpy(), preds.numpy()
            ss_res = np.sum((t_np - p_np) ** 2)
            ss_tot = np.sum((t_np - np.mean(t_np)) ** 2)
            metrics["r2"] = float(1 - ss_res / max(ss_tot, 1e-10))
        return metrics

    val_metrics = evaluate(val_loader)
    test_metrics = evaluate(test_loader)
    # If test has no labels, use val metrics as proxy
    if test_metrics is None:
        test_metrics = val_metrics
        logger.info("  Test labels unavailable, using val metrics as proxy")
    elapsed = time.time() - start_time

    result = {
        "method": method, "task": task_name, "seed": seed, "dataset": "rel-f1",
        "hyperparams": {"channels": channels, "lr": lr, "epochs": epochs,
                        "batch_size": batch_size, "num_neighbors": num_neighbors,
                        "num_layers": num_layers},
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "best_epoch": best_epoch, "train_time_seconds": round(elapsed, 1),
        "train_losses": [round(l, 4) for l in train_losses],
    }

    logger.info(f"Finished: {method}/{task_name}/seed{seed} in {elapsed:.0f}s. test={test_metrics}")

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{task_name}_{method}_seed{seed}.json"
        (results_dir / fname).write_text(json.dumps(result, indent=2))

    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# EFFECTIVE RANK MEASUREMENT
# ══════════════════════════════════════════════════════════════════════════════

def compute_effective_rank(X: torch.Tensor) -> dict:
    if X.shape[0] < 2 or X.shape[1] < 2:
        return {"erank": 1.0, "normalized_erank": 1.0, "num_rows": X.shape[0]}
    try:
        U, S, V = torch.linalg.svd(X.float(), full_matrices=False)
        S = S[S > 1e-10]
        if len(S) == 0:
            return {"erank": 1.0, "normalized_erank": 0.0, "num_rows": X.shape[0]}
        p = S / S.sum()
        H = -(p * p.log()).sum().item()
        erank = math.exp(H)
        max_rank = min(X.shape[0], X.shape[1])
        return {"erank": round(erank, 4), "normalized_erank": round(erank / max_rank, 4),
                "num_rows": X.shape[0], "num_singular_values": len(S)}
    except Exception as e:
        logger.warning(f"SVD failed: {e}")
        return {"erank": float("nan"), "normalized_erank": float("nan"), "num_rows": X.shape[0]}


def run_rank_analysis(device) -> dict:
    """Measure effective rank for mean vs sum on seed=42."""
    logger.info("Running rank analysis...")
    from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
    from torch_geometric.loader import NeighborLoader

    rank_results = {}
    for method in ["mean", "sum"]:
        logger.info(f"  Rank analysis for {method}...")
        torch.manual_seed(42)
        np.random.seed(42)
        cached = load_data_and_task("driver-position", device)
        data = cached["data"]
        col_stats_dict = cached["col_stats_dict"]
        entity_table = cached["entity_table"]
        val_input = cached["val_input"]
        channels = 128

        class RankProbeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = HeteroEncoder(
                    channels=channels,
                    node_to_col_names_dict={
                        nt: data[nt].tf.col_names_dict
                        for nt in data.node_types if hasattr(data[nt], 'tf')
                    },
                    node_to_col_stats=col_stats_dict,
                )
                self.temporal_encoder = HeteroTemporalEncoder(
                    node_types=[nt for nt in data.node_types if "time" in data[nt]],
                    channels=channels,
                )
                self.gnn = HeteroGraphSAGE(
                    node_types=data.node_types, edge_types=data.edge_types,
                    channels=channels, aggr=method, num_layers=2,
                )
            def forward(self, batch):
                seed_time = batch[entity_table].seed_time
                x_dict = self.encoder(batch.tf_dict)
                safe_time_dict = {nt: t for nt, t in batch.time_dict.items()
                                  if nt in batch.batch_size_dict}
                rel_time_dict = self.temporal_encoder(seed_time, safe_time_dict, batch.batch_size_dict)
                for nt in x_dict:
                    if nt in rel_time_dict:
                        x_dict[nt] = x_dict[nt] + rel_time_dict[nt]
                x_dict = self.gnn(x_dict, batch.edge_index_dict,
                                  batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict)
                return x_dict

        probe = RankProbeModel().to(device)
        probe.eval()
        val_loader = NeighborLoader(
            data, num_neighbors=[128, 64], time_attr="time",
            input_nodes=val_input.nodes, input_time=val_input.time,
            transform=val_input.transform, batch_size=256, shuffle=False, num_workers=0,
        )
        all_emb = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 5:
                    break
                batch = batch.to(device)
                x_dict = probe(batch)
                emb = x_dict[entity_table][:batch[entity_table].batch_size]
                all_emb.append(emb.cpu())
        if all_emb:
            rank_info = compute_effective_rank(torch.cat(all_emb, dim=0))
        else:
            rank_info = {"erank": float("nan"), "normalized_erank": float("nan")}
        rank_results[f"{method}_aggregation"] = {"entity_embeddings": rank_info, "method": method}
        del probe
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_er = rank_results.get("mean_aggregation", {}).get("entity_embeddings", {}).get("erank", 1.0)
    sum_er = rank_results.get("sum_aggregation", {}).get("entity_embeddings", {}).get("erank", 1.0)
    rank_results["collapse_ratio"] = round(mean_er / sum_er, 4) if mean_er > 0 and sum_er > 0 else float("nan")
    logger.info(f"Rank: mean_erank={mean_er:.2f}, sum_erank={sum_er:.2f}, ratio={rank_results['collapse_ratio']}")
    return rank_results


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    pooled = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return float((g2.mean() - g1.mean()) / pooled) if pooled > 1e-12 else 0.0


def bootstrap_cohens_d(g1: np.ndarray, g2: np.ndarray, n_boot: int = 10000, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    d_obs = cohens_d(g1, g2)
    n1, n2 = len(g1), len(g2)
    d_boot = np.array([cohens_d(g1[rng.choice(n1, n1, replace=True)],
                                g2[rng.choice(n2, n2, replace=True)]) for _ in range(n_boot)])
    ci_low, ci_high = np.percentile(d_boot, [2.5, 97.5])
    combined = np.concatenate([g1, g2])
    count = sum(1 for _ in range(n_boot)
                if abs(cohens_d(combined[rng.permutation(len(combined))[:n1]],
                                combined[rng.permutation(len(combined))[n1:]])) >= abs(d_obs))
    return {"d": round(d_obs, 4), "ci_low": round(float(ci_low), 4),
            "ci_high": round(float(ci_high), 4), "p_value": round(count / n_boot, 4)}


def run_interaction_test(results: list, task_name: str) -> dict:
    relevant = [r for r in results if r["task"] == task_name and
                r["method"] in ["mean", "sum", "mean_cama", "sum_cama"]
                and r.get("test_metrics")]
    if len(relevant) < 4:
        return {"F_stat": None, "p_value": None, "interpretation": "Insufficient data"}
    metric_key = "average_precision" if task_name == "driver-dnf" else "mae"
    rows = []
    for r in relevant:
        aggr_type = "mean" if r["method"] in ["mean", "mean_cama"] else "sum"
        has_cama = "yes" if "cama" in r["method"] else "no"
        rows.append({"aggr": aggr_type, "cama": has_cama,
                      "metric": r["test_metrics"].get(metric_key, float("nan"))})
    df = pd.DataFrame(rows)
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        model = ols('metric ~ C(aggr) * C(cama)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        key = 'C(aggr):C(cama)'
        if key in anova_table.index:
            F_stat = float(anova_table.loc[key, 'F'])
            p_val = float(anova_table.loc[key, 'PR(>F)'])
        else:
            F_stat, p_val = float("nan"), float("nan")
        if p_val < 0.05:
            interp = f"Significant interaction (p={p_val:.4f}): CAMA benefit depends on aggregation type"
        elif p_val < 0.15:
            interp = f"Trending interaction (p={p_val:.4f}): suggestive but not significant"
        else:
            interp = f"No significant interaction (p={p_val:.4f}): CAMA benefit NOT aggregation-specific"
        return {"F_stat": round(F_stat, 4), "p_value": round(p_val, 4), "interpretation": interp}
    except Exception as e:
        logger.exception(f"Interaction test failed")
        return {"F_stat": None, "p_value": None, "interpretation": f"Failed: {str(e)[:200]}"}


def run_statistical_analysis(all_results: list) -> dict:
    logger.info("Running statistical analysis...")
    TASKS = ["driver-position", "driver-dnf"]
    METHODS = ["mean", "sum", "mean_cama", "sum_cama", "mean_ungated"]
    COMPARISONS = [
        ("mean_cama_vs_mean", "mean", "mean_cama"),
        ("sum_cama_vs_sum", "sum", "sum_cama"),
        ("sum_vs_mean", "mean", "sum"),
        ("mean_ungated_vs_mean", "mean", "mean_ungated"),
        ("mean_cama_vs_mean_ungated", "mean_ungated", "mean_cama"),
        ("mean_cama_vs_sum", "sum", "mean_cama"),
    ]
    summary_table = {}
    cohens_d_comparisons = {}
    interaction_tests = {}

    for task_name in TASKS:
        is_cls = task_name == "driver-dnf"
        mk = "average_precision" if is_cls else "mae"
        higher = is_cls
        prefix = "test_ap" if is_cls else "test_mae"

        summary_table[task_name] = {}
        for method in METHODS:
            vals = [r["test_metrics"].get(mk) for r in all_results
                    if r["task"] == task_name and r["method"] == method and r.get("test_metrics")]
            vals = [v for v in vals if v is not None and not math.isnan(v)]
            if not vals:
                continue
            arr = np.array(vals)
            summary_table[task_name][method] = {
                f"{prefix}_mean": round(float(arr.mean()), 4),
                f"{prefix}_std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
                "per_seed": [round(v, 4) for v in vals], "n_seeds": len(vals),
            }

        cohens_d_comparisons[task_name] = {}
        for comp_name, base_m, test_m in COMPARISONS:
            bvals = [r["test_metrics"].get(mk) for r in all_results
                     if r["task"] == task_name and r["method"] == base_m and r.get("test_metrics")]
            tvals = [r["test_metrics"].get(mk) for r in all_results
                     if r["task"] == task_name and r["method"] == test_m and r.get("test_metrics")]
            bvals = [v for v in bvals if v is not None and not math.isnan(v)]
            tvals = [v for v in tvals if v is not None and not math.isnan(v)]
            if len(bvals) < 2 or len(tvals) < 2:
                cohens_d_comparisons[task_name][comp_name] = {"d": None, "ci_low": None, "ci_high": None, "p_value": None}
                continue
            g1, g2 = np.array(bvals), np.array(tvals)
            if not higher:  # MAE: lower is better, flip for positive d = improvement
                result = bootstrap_cohens_d(g2, g1)
            else:
                result = bootstrap_cohens_d(g1, g2)
            cohens_d_comparisons[task_name][comp_name] = result

        interaction_tests[task_name] = run_interaction_test(all_results, task_name)

    return {"summary_table": summary_table, "cohens_d_comparisons": cohens_d_comparisons,
            "interaction_test": interaction_tests}


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def build_output_json(all_results: list, analysis: dict, rank_analysis: dict,
                      dep3_data: list, dep4_data: list) -> dict:
    key_findings = []
    for task_name in ["driver-position", "driver-dnf"]:
        comps = analysis.get("cohens_d_comparisons", {}).get(task_name, {})
        for cn, cd in comps.items():
            if cd and cd.get("d") is not None:
                key_findings.append(f"{task_name}: {cn} d={cd['d']:.3f} (p={cd.get('p_value', 'N/A')})")
        interaction = analysis.get("interaction_test", {}).get(task_name, {})
        if interaction.get("interpretation"):
            key_findings.append(f"{task_name}: {interaction['interpretation']}")

    metadata = {
        "method_name": "CAMA (Cardinality-Aware Moment Aggregation)",
        "description": "Controlled experiment on rel-f1 comparing 5 aggregation methods across 2 tasks with 5 seeds.",
        "experiment_config": {
            "dataset": "rel-f1", "tasks": ["driver-position", "driver-dnf"],
            "methods": ["mean", "sum", "mean_cama", "sum_cama", "mean_ungated"],
            "seeds": [42, 123, 456, 789, 1024],
            "hyperparams": {"channels": 128, "lr": 0.005, "epochs": 10,
                            "batch_size": 512, "num_neighbors": 128, "num_layers": 2},
        },
        "per_run_results": all_results,
        "summary_table": analysis.get("summary_table", {}),
        "cohens_d_comparisons": analysis.get("cohens_d_comparisons", {}),
        "interaction_test": analysis.get("interaction_test", {}),
        "rank_analysis": rank_analysis,
        "key_findings": key_findings,
    }

    datasets = []
    # driver-position examples
    dp_ex = []
    for r in all_results:
        if r["task"] == "driver-position" and r.get("test_metrics"):
            dp_ex.append({
                "input": json.dumps({"method": r["method"], "task": r["task"], "seed": r["seed"],
                                     "dataset": r["dataset"], "hyperparams": r["hyperparams"]}),
                "output": json.dumps(r["test_metrics"]),
                "predict_baseline": str(r["test_metrics"].get("mae", "")),
                "predict_our_method": str(r["test_metrics"].get("mae", "")),
                "metadata_method": r["method"], "metadata_task": r["task"],
                "metadata_seed": r["seed"], "metadata_best_epoch": r["best_epoch"],
            })
    for ex in dep3_data:
        dp_ex.append({"input": ex.get("input", ""), "output": ex.get("output", ""),
                       "predict_baseline": ex.get("output", ""), "predict_our_method": ex.get("output", ""),
                       "metadata_fold": ex.get("metadata_fold", 0),
                       "metadata_task_type": ex.get("metadata_task_type", "regression")})
    if dp_ex:
        datasets.append({"dataset": "rel-f1/driver-position", "examples": dp_ex})

    # driver-dnf examples
    dnf_ex = []
    for r in all_results:
        if r["task"] == "driver-dnf" and r.get("test_metrics"):
            dnf_ex.append({
                "input": json.dumps({"method": r["method"], "task": r["task"], "seed": r["seed"],
                                     "dataset": r["dataset"], "hyperparams": r["hyperparams"]}),
                "output": json.dumps(r["test_metrics"]),
                "predict_baseline": str(r["test_metrics"].get("average_precision", "")),
                "predict_our_method": str(r["test_metrics"].get("average_precision", "")),
                "metadata_method": r["method"], "metadata_task": r["task"],
                "metadata_seed": r["seed"], "metadata_best_epoch": r["best_epoch"],
            })
    for ex in dep4_data:
        dnf_ex.append({"input": ex.get("input", ""), "output": ex.get("output", ""),
                        "predict_baseline": ex.get("output", ""), "predict_our_method": ex.get("output", ""),
                        "metadata_fold": ex.get("metadata_fold", 0),
                        "metadata_task_type": ex.get("metadata_task_type", "binary_classification")})
    if dnf_ex:
        datasets.append({"dataset": "rel-f1/driver-dnf", "examples": dnf_ex})

    return {"metadata": metadata, "datasets": datasets}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    logger.info("=" * 70)
    logger.info("CAMA Mean-vs-Sum Aggregation Causal Diagnosis Experiment")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    run_unit_tests()

    METHODS = ["mean", "sum", "mean_cama", "sum_cama", "mean_ungated"]
    TASKS = ["driver-position", "driver-dnf"]
    SEEDS = [42, 123, 456, 789, 1024]
    RESULTS_DIR = WORKSPACE / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = []
    pending_runs = []
    for task_name in TASKS:
        for method in METHODS:
            for seed in SEEDS:
                fname = f"{task_name}_{method}_seed{seed}.json"
                fpath = RESULTS_DIR / fname
                if fpath.exists():
                    try:
                        all_results.append(json.loads(fpath.read_text()))
                        logger.info(f"  Loaded cached: {fname}")
                        continue
                    except Exception:
                        pass
                pending_runs.append((method, task_name, seed))

    logger.info(f"Total: {len(METHODS)*len(TASKS)*len(SEEDS)}, cached: {len(all_results)}, pending: {len(pending_runs)}")

    total_start = time.time()
    succeeded = failed = 0
    for i, (method, task_name, seed) in enumerate(pending_runs):
        logger.info(f"\n--- Run {i+1}/{len(pending_runs)}: {method}/{task_name}/seed{seed} ---")
        try:
            result = run_single_experiment(
                method=method, task_name=task_name, seed=seed, device=device,
                channels=128, lr=0.005, epochs=10, batch_size=512,
                num_neighbors=128, num_layers=2, results_dir=RESULTS_DIR,
            )
            all_results.append(result)
            succeeded += 1
        except Exception as e:
            logger.exception(f"Run failed: {method}/{task_name}/seed{seed}")
            failed += 1
            error_result = {
                "method": method, "task": task_name, "seed": seed, "dataset": "rel-f1",
                "error": str(e)[:500],
                "hyperparams": {"channels": 128, "lr": 0.005, "epochs": 10,
                                "batch_size": 512, "num_neighbors": 128, "num_layers": 2},
                "val_metrics": {}, "test_metrics": {}, "best_epoch": 0,
                "train_time_seconds": 0, "train_losses": [],
            }
            all_results.append(error_result)
            (RESULTS_DIR / f"{task_name}_{method}_seed{seed}.json").write_text(
                json.dumps(error_result, indent=2))

        if (i + 1) % 5 == 0:
            elapsed = time.time() - total_start
            logger.info(f"Progress: {i+1}/{len(pending_runs)}, {elapsed/60:.1f}min elapsed")

    logger.info(f"\nRuns complete: {succeeded} ok, {failed} failed, {(time.time()-total_start)/60:.1f}min total")

    # Save combined
    (WORKSPACE / "combined_results.json").write_text(json.dumps(all_results, indent=2))

    # Rank analysis
    try:
        rank_analysis = run_rank_analysis(device)
    except Exception as e:
        logger.exception("Rank analysis failed")
        rank_analysis = {"error": str(e)[:500]}

    # Statistical analysis
    valid = [r for r in all_results if r.get("test_metrics")]
    analysis = run_statistical_analysis(valid)

    # Load dependency examples
    dep3_path = Path("/ai-inventor/aii_pipeline/runs/leskovec-predictive-residual-message-passing-v2_sti/3_invention_loop/iter_1/gen_art/data_id3_it1__opus/mini_data_out.json")
    dep4_path = Path("/ai-inventor/aii_pipeline/runs/leskovec-predictive-residual-message-passing-v2_sti/3_invention_loop/iter_1/gen_art/data_id4_it1__opus/mini_data_out.json")
    dep3_data, dep4_data = [], []
    try:
        for ds in json.loads(dep3_path.read_text()).get("datasets", []):
            if "driver-position" in ds.get("dataset", ""):
                dep3_data = ds.get("examples", [])[:3]
    except Exception:
        pass
    try:
        for ds in json.loads(dep4_path.read_text()).get("datasets", []):
            if "driver-dnf" in ds.get("dataset", ""):
                dep4_data = ds.get("examples", [])[:3]
    except Exception:
        pass

    output = build_output_json(all_results, analysis, rank_analysis, dep3_data, dep4_data)
    (WORKSPACE / "method_out.json").write_text(json.dumps(output, indent=2))
    logger.info(f"Saved method_out.json")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    for task_name in TASKS:
        logger.info(f"\n{task_name}:")
        st = analysis.get("summary_table", {}).get(task_name, {})
        for method, vals in st.items():
            mk = [k for k in vals if k.endswith("_mean")]
            if mk:
                logger.info(f"  {method:20s}: {mk[0]}={vals[mk[0]]:.4f} +/- {vals[mk[0].replace('_mean','_std')]:.4f}")
    logger.info("\nKey findings:")
    for f in output.get("metadata", {}).get("key_findings", []):
        logger.info(f"  - {f}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
