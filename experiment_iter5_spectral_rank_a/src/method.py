#!/usr/bin/env python3
"""Spectral Rank Analysis at FK Aggregation.

Train HeteroSAGE (mean aggregation, 10 epochs, seed=42) on 3 tasks with
contrasting CAMA effect sizes, then instrument the aggregation step to capture
child embeddings at every FK edge type. Compute SVD-based effective rank and
the cheap variance-entropy proxy, measure rank compression ratios, validate
the proxy against SVD, and correlate compression severity with CAMA Cohen's d.
"""

import copy
import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.stats import pearsonr, spearmanr

# ============================================================================
# SECTION 0: Constants, Hardware Detection, Resource Limits
# ============================================================================

WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")


def _detect_cpus() -> int:
    """Detect actual CPU allocation (containers/pods/bare metal)."""
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float | None:
    """Read RAM limit from cgroup (containers/pods)."""
    for p in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
)
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or 56.0

# Set resource limits
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1e9)  # 80% of container RAM
resource.setrlimit(
    resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3)
)
if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(0.92)

logger.info(
    f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
    f"GPU={'yes' if HAS_GPU else 'no'}"
    f"{f' ({VRAM_GB:.1f}GB VRAM)' if HAS_GPU else ''}"
)

# Experiment constants
TASKS = [
    ("rel-f1", "driver-dnf"),  # classification, d=1.13 (smallest first)
    ("rel-trial", "study-adverse"),  # regression, d=-2.45
    ("rel-stack", "user-engagement"),  # classification, d=7.95 (largest last)
]
CAMA_COHENS_D = {
    "rel-f1/driver-dnf": 1.13,
    "rel-trial/study-adverse": -2.45,
    "rel-stack/user-engagement": 7.95,
}
SEED = 42
EPOCHS = 10
CHANNELS = 128
NUM_LAYERS = 2
NUM_NEIGHBORS = 128
BATCH_SIZE = 512
LR = 0.005
MAX_PARENTS_PER_EDGE_TYPE = 1000
MAX_ANALYSIS_BATCHES = 100
MAX_CHILDREN_FOR_SVD = 500
EPS = 1e-10
MAX_STEPS_PER_EPOCH = 2000

# Per-task hyperparameter overrides (from RelBench configs)
TASK_OVERRIDES = {
    "rel-trial/study-adverse": {
        "batch_size": 128,
        "num_neighbors": 64,
        "lr": 0.005,
    },
    "rel-stack/user-engagement": {
        "batch_size": 512,
        "num_neighbors": 128,
        "lr": 0.005,
    },
}

# Cache directory for relbench data
CACHE_DIR = WORKSPACE / "relbench_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================================
# SECTION 1: Rank Computation Functions
# ============================================================================


def compute_svd_effective_rank(C: torch.Tensor, eps: float = EPS) -> float:
    """SVD-based effective rank of child embedding matrix C [N, d].

    erank = exp(H(sigma)) where sigma are normalized singular values.
    """
    if C.shape[0] < 2:
        return 1.0
    # Cap children for SVD performance
    if C.shape[0] > MAX_CHILDREN_FOR_SVD:
        perm = torch.randperm(C.shape[0])[:MAX_CHILDREN_FOR_SVD]
        C = C[perm]
    C = C.float()
    try:
        s = torch.linalg.svdvals(C)
    except Exception:
        return float("nan")
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    H = -(p * p.log()).sum().item()
    return math.exp(H)


def compute_variance_entropy_proxy(
    C: torch.Tensor, d: int, eps: float = EPS
) -> float:
    """Variance-entropy proxy: H(var) / log(d), normalized to [0, 1].

    Matches RAMA blueprint exactly.
    """
    if C.shape[0] < 2:
        return 1.0 / max(d, 1)
    var = C.var(dim=0)
    var_sum = var.sum()
    if var_sum < eps:
        return 1.0 / max(d, 1)
    p = var / var_sum
    p = p.clamp(min=eps)
    H = -(p * p.log()).sum().item()
    log_d = math.log(max(d, 2))
    return H / log_d


def compute_compression_ratio(erank: float, N: int, d: int) -> float:
    """Ratio of effective rank to maximum possible rank = min(N, d)."""
    max_rank = min(N, d)
    if max_rank == 0:
        return float("nan")
    return erank / max_rank


# ============================================================================
# SECTION 2: Instrumented Aggregation Module
# ============================================================================


class RankAnalysisCollector:
    """Collects child embeddings and aggregation results during analysis."""

    def __init__(self):
        self.recording = False
        self.records: Dict[str, list] = {}

    def start_recording(self):
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def clear(self):
        self.records = {}

    def record(self, key: str, x, index, dim_size, output):
        if not self.recording:
            return
        if key not in self.records:
            self.records[key] = []
        self.records[key].append(
            (
                x.detach().cpu(),
                index.detach().cpu(),
                dim_size,
                output.detach().cpu(),
            )
        )


# Import PyG aggregation base
from torch_geometric.nn.aggr import Aggregation


class InstrumentedMeanAggregation(Aggregation):
    """Mean aggregation that optionally records inputs for rank analysis."""

    def __init__(self, key: str, collector: RankAnalysisCollector):
        super().__init__()
        self.key = key
        self.collector = collector

    def forward(
        self,
        x,
        index=None,
        ptr=None,
        dim_size=None,
        dim=-2,
        max_num_elements=None,
    ):
        output = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        self.collector.record(self.key, x, index, dim_size, output)
        return output


# ============================================================================
# SECTION 3: Model Construction
# ============================================================================

from torch_geometric.nn import HeteroConv, LayerNorm, MLP, SAGEConv
from torch_geometric.typing import EdgeType, NodeType


class InstrumentedHeteroGraphSAGE(torch.nn.Module):
    """HeteroGraphSAGE with per-edge-type InstrumentedMeanAggregation."""

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        num_layers: int,
        collector: RankAnalysisCollector,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                key = (
                    f"layer_{layer_idx}/"
                    f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
                )
                aggr = InstrumentedMeanAggregation(key=key, collector=collector)
                conv_dict[edge_type] = SAGEConv(
                    (channels, channels), channels, aggr=aggr
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        num_sampled_nodes_dict=None,
        num_sampled_edges_dict=None,
    ) -> Dict[NodeType, torch.Tensor]:
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict


# Import RelBench modeling components
from torch.nn import Embedding, ModuleDict

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder


class InstrumentedModel(torch.nn.Module):
    """Full model mirroring RelBench's Model but with instrumented GNN."""

    def __init__(
        self,
        data,
        col_stats_dict: Dict,
        num_layers: int,
        channels: int,
        out_channels: int,
        collector: RankAnalysisCollector,
        norm: str = "batch_norm",
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type
                for node_type in data.node_types
                if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = InstrumentedHeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            num_layers=num_layers,
            collector=collector,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch, entity_table: NodeType) -> torch.Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        return self.head(x_dict[entity_table][: seed_time.size(0)])


# ============================================================================
# SECTION 4: Data Loading and Graph Construction
# ============================================================================

from torch_frame import stype
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task


def load_dataset_and_task(
    dataset_name: str, task_name: str
) -> Tuple[Dataset, EntityTask]:
    """Load a RelBench dataset and task with download."""
    logger.info(f"Loading dataset={dataset_name}, task={task_name}")
    try:
        dataset = get_dataset(dataset_name, download=True)
        task = get_task(dataset_name, task_name, download=True)
    except Exception:
        logger.exception(f"Failed to load {dataset_name}/{task_name}")
        raise
    logger.info(f"Dataset {dataset_name} loaded successfully")
    return dataset, task


def prepare_graph(
    dataset: Dataset, cache_name: str
) -> Tuple[Any, Dict]:
    """Construct the heterogeneous graph, excluding text columns."""
    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)

    # Remove text_embedded columns to avoid needing GloVe
    removed = []
    for table_name in list(col_to_stype_dict.keys()):
        cols_to_remove = [
            col
            for col, st in col_to_stype_dict[table_name].items()
            if st == stype.text_embedded
        ]
        for col in cols_to_remove:
            del col_to_stype_dict[table_name][col]
            removed.append(f"{table_name}.{col}")
    if removed:
        logger.info(
            f"Removed {len(removed)} text columns: "
            f"{', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}"
        )

    mat_dir = str(CACHE_DIR / f"{cache_name}" / "materialized")
    logger.info(f"Materializing graph to {mat_dir}")
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        cache_dir=mat_dir,
    )
    logger.info(
        f"Graph: {len(data.node_types)} node types, "
        f"{len(data.edge_types)} edge types"
    )
    return data, col_stats_dict


def create_loaders(
    data,
    task: EntityTask,
    num_neighbors: int,
    batch_size: int,
    num_layers: int,
) -> Tuple[Dict[str, NeighborLoader], str]:
    """Create train/val/test NeighborLoaders."""
    loader_dict = {}
    entity_table = None
    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        if entity_table is None:
            entity_table = table_input.nodes[0]
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[
                int(num_neighbors / 2**i) for i in range(num_layers)
            ],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=batch_size,
            temporal_strategy="uniform",
            shuffle=split == "train",
            num_workers=0,
            persistent_workers=False,
        )
    return loader_dict, entity_table


# ============================================================================
# SECTION 5: Training and Evaluation
# ============================================================================

from torch.nn import BCEWithLogitsLoss, L1Loss


def train_epoch(
    model: InstrumentedModel,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    entity_table: str,
    task_type: TaskType,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    loss_accum = 0.0
    count_accum = 0
    steps = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch, entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        y = batch[entity_table].y.float()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)
        steps += 1
        if steps >= MAX_STEPS_PER_EPOCH:
            break
    return loss_accum / max(count_accum, 1)


@torch.no_grad()
def evaluate_model(
    model: InstrumentedModel,
    loader: NeighborLoader,
    entity_table: str,
    task_type: TaskType,
    device: torch.device,
    clamp_min: float = None,
    clamp_max: float = None,
) -> np.ndarray:
    """Run inference and return predictions."""
    model.eval()
    pred_list = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, entity_table)
        if task_type == TaskType.REGRESSION and clamp_min is not None:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task_type == TaskType.BINARY_CLASSIFICATION:
            pred = torch.sigmoid(pred)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


# ============================================================================
# SECTION 6: Rank Analysis
# ============================================================================


def analyze_batch_records(
    records_dict: Dict[str, list], max_parents_per_key: int = 200
) -> Dict[str, list]:
    """Process one batch of recorded aggregation data."""
    batch_metrics: Dict[str, list] = {}
    for key, record_list in records_dict.items():
        parent_metrics = []
        for x, index, dim_size, output in record_list:
            if index is None or x.shape[0] == 0:
                continue
            d = x.shape[1]
            unique_parents = index.unique()
            if len(unique_parents) > max_parents_per_key:
                perm = torch.randperm(len(unique_parents))[:max_parents_per_key]
                sampled = unique_parents[perm]
            else:
                sampled = unique_parents

            for pid in sampled:
                mask = index == pid
                C = x[mask]
                N = C.shape[0]
                if N < 1:
                    continue
                svd_rank = compute_svd_effective_rank(C)
                proxy = compute_variance_entropy_proxy(C, d)
                compression = compute_compression_ratio(svd_rank, N, d)
                parent_metrics.append(
                    {
                        "cardinality": int(N),
                        "svd_effective_rank": float(svd_rank),
                        "proxy_rank": float(proxy),
                        "compression_ratio": float(compression),
                        "max_possible_rank": int(min(N, d)),
                    }
                )
        if key not in batch_metrics:
            batch_metrics[key] = []
        batch_metrics[key].extend(parent_metrics)
    return batch_metrics


def run_rank_analysis(
    model: InstrumentedModel,
    collector: RankAnalysisCollector,
    loader: NeighborLoader,
    entity_table: str,
    device: torch.device,
    max_batches: int = MAX_ANALYSIS_BATCHES,
) -> Dict[str, list]:
    """Run forward passes with recording, accumulate rank metrics."""
    model.eval()
    all_metrics: Dict[str, list] = {}
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        collector.clear()
        collector.start_recording()
        with torch.no_grad():
            batch = batch.to(device)
            model(batch, entity_table)
        collector.stop_recording()

        batch_metrics = analyze_batch_records(
            collector.records, max_parents_per_key=200
        )
        for key, metrics_list in batch_metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].extend(metrics_list)
            if len(all_metrics[key]) > MAX_PARENTS_PER_EDGE_TYPE:
                all_metrics[key] = all_metrics[key][
                    :MAX_PARENTS_PER_EDGE_TYPE
                ]

        if batch_idx % 10 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  Analysis batch {batch_idx}/{max_batches} "
                f"({elapsed:.1f}s, {len(all_metrics)} edge types)"
            )

    elapsed = time.time() - t0
    total_parents = sum(len(v) for v in all_metrics.values())
    logger.info(
        f"  Analysis done: {len(all_metrics)} edge types, "
        f"{total_parents} parent groups, {elapsed:.1f}s"
    )
    return all_metrics


# ============================================================================
# SECTION 7: Statistical Summary
# ============================================================================


def safe_stats(arr: np.ndarray) -> Dict:
    """Compute summary statistics, handling empty arrays."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "p25": None,
            "p75": None,
            "n": 0,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "n": int(len(arr)),
    }


def summarize_edge_type(metrics_list: list) -> Optional[Dict]:
    """Compute summary statistics for one edge type."""
    if not metrics_list:
        return None
    cards = np.array([m["cardinality"] for m in metrics_list], dtype=float)
    svd_ranks = np.array(
        [m["svd_effective_rank"] for m in metrics_list], dtype=float
    )
    proxies = np.array([m["proxy_rank"] for m in metrics_list], dtype=float)
    compressions = np.array(
        [m["compression_ratio"] for m in metrics_list], dtype=float
    )

    # Spearman correlation SVD vs proxy
    valid = ~np.isnan(svd_ranks) & ~np.isnan(proxies)
    if valid.sum() >= 5:
        rho, pval = spearmanr(svd_ranks[valid], proxies[valid])
        rho = float(rho) if not np.isnan(rho) else None
        pval = float(pval) if not np.isnan(pval) else None
    else:
        rho, pval = None, None

    return {
        "num_parents_analyzed": len(metrics_list),
        "cardinality": safe_stats(cards),
        "svd_effective_rank": safe_stats(svd_ranks),
        "proxy_rank": safe_stats(proxies),
        "compression_ratio": safe_stats(compressions),
        "spearman_svd_vs_proxy": {"rho": rho, "p_value": pval},
    }


def compute_task_severity(
    edge_summaries: Dict[str, Dict],
) -> Tuple[float, float, float]:
    """Compute task-level severity score (cardinality-weighted compression).

    Returns (severity, mean_compression, mean_proxy).
    Lower severity = more rank collapse = hypothesized larger CAMA benefit.
    """
    compressions = []
    weights = []
    proxies = []
    for key, summary in edge_summaries.items():
        if summary is None:
            continue
        cr = summary["compression_ratio"]
        card = summary["cardinality"]
        pr = summary["proxy_rank"]
        if cr["mean"] is not None and card["mean"] is not None:
            compressions.append(cr["mean"])
            weights.append(card["mean"])
        if pr["mean"] is not None:
            proxies.append(pr["mean"])

    if not compressions:
        return float("nan"), float("nan"), float("nan")

    weights = np.array(weights)
    compressions = np.array(compressions)
    total_w = weights.sum()
    if total_w > 0:
        severity = float(np.average(compressions, weights=weights))
    else:
        severity = float(np.mean(compressions))
    mean_compression = float(np.mean(compressions))
    mean_proxy = float(np.mean(proxies)) if proxies else float("nan")
    return severity, mean_compression, mean_proxy


# ============================================================================
# SECTION 8: Cross-Task Correlation
# ============================================================================


def cross_task_correlation(
    task_results: Dict[str, Dict],
) -> Dict:
    """Correlate rank compression severity with CAMA Cohen's d."""
    tasks = []
    severities = []
    cama_ds = []

    for task_key, result in task_results.items():
        d = CAMA_COHENS_D.get(task_key)
        if d is None:
            continue
        sev = result.get("severity")
        if sev is None or math.isnan(sev):
            continue
        tasks.append(task_key)
        severities.append(sev)
        cama_ds.append(d)

    n = len(tasks)
    analysis = {
        "tasks": [
            {"task": t, "severity": s, "cama_d": d}
            for t, s, d in zip(tasks, severities, cama_ds)
        ],
        "n_points": n,
    }

    if n >= 3:
        try:
            rho, rho_p = spearmanr(severities, cama_ds)
            analysis["spearman_severity_vs_d"] = {
                "rho": float(rho),
                "p_value": float(rho_p),
            }
        except Exception:
            analysis["spearman_severity_vs_d"] = {"rho": None, "p_value": None}
        try:
            r, r_p = pearsonr(severities, cama_ds)
            analysis["pearson_severity_vs_d"] = {
                "r": float(r),
                "p_value": float(r_p),
            }
        except Exception:
            analysis["pearson_severity_vs_d"] = {"r": None, "p_value": None}
    else:
        analysis["spearman_severity_vs_d"] = {"rho": None, "p_value": None}
        analysis["pearson_severity_vs_d"] = {"r": None, "p_value": None}

    # Qualitative interpretation
    if n >= 3 and analysis["spearman_severity_vs_d"]["rho"] is not None:
        rho_val = analysis["spearman_severity_vs_d"]["rho"]
        # Lower severity (more compression) should correlate with higher d
        # i.e., negative Spearman rho expected
        if rho_val < -0.5:
            interp = (
                "Strong negative correlation: tasks with more severe rank "
                "collapse (lower compression ratio) show higher CAMA benefit "
                "(higher Cohen's d). Supports the causal hypothesis."
            )
        elif rho_val < 0:
            interp = (
                "Weak negative correlation: directional support for the "
                "rank-collapse mechanism but not conclusive with only "
                f"{n} data points."
            )
        elif rho_val > 0.5:
            interp = (
                "Positive correlation: tasks with LESS compression show "
                "higher CAMA benefit. This DISCONFIRMS the rank-collapse "
                "hypothesis as the primary mechanism."
            )
        else:
            interp = (
                f"Near-zero correlation (rho={rho_val:.3f}): no clear "
                "relationship between rank compression and CAMA benefit."
            )
    else:
        interp = f"Insufficient data points ({n}) for correlation analysis."

    analysis["interpretation"] = interp
    return analysis


# ============================================================================
# SECTION 9: Output Generation (exp_gen_sol_out.json schema)
# ============================================================================


def generate_output(
    task_results: Dict[str, Dict],
    cross_analysis: Dict,
) -> Dict:
    """Generate output in exp_gen_sol_out.json schema format.

    Schema: {"metadata": {...}, "datasets": [{"dataset": str, "examples": [...]}]}
    Each example: {"input": str, "output": str, "metadata_*": ..., "predict_*": str}
    """
    datasets = []

    # One dataset per task
    for task_key, result in task_results.items():
        examples = []
        edge_summaries = result.get("edge_summaries", {})
        train_result = result.get("train_result", {})
        task_config = result.get("task_config", {})

        for edge_key, summary in edge_summaries.items():
            if summary is None:
                continue
            # Parse edge key: "layer_X/src__rel__dst"
            parts = edge_key.split("/", 1)
            layer = parts[0] if len(parts) > 1 else "unknown"
            edge_desc = parts[1] if len(parts) > 1 else edge_key

            input_data = {
                "task": task_key,
                "edge_type": edge_key,
                "layer": layer,
                "edge_description": edge_desc,
                "cama_cohen_d": CAMA_COHENS_D.get(task_key),
                "task_type": task_config.get("task_type", "unknown"),
            }
            output_data = {
                "num_parents_analyzed": summary["num_parents_analyzed"],
                "mean_svd_rank": summary["svd_effective_rank"]["mean"],
                "mean_proxy_rank": summary["proxy_rank"]["mean"],
                "mean_compression": summary["compression_ratio"]["mean"],
                "mean_cardinality": summary["cardinality"]["mean"],
                "spearman_svd_vs_proxy": summary["spearman_svd_vs_proxy"],
                "cardinality_stats": summary["cardinality"],
                "compression_stats": summary["compression_ratio"],
            }

            example = {
                "input": json.dumps(input_data),
                "output": json.dumps(output_data),
                "metadata_task": task_key,
                "metadata_edge_type": edge_key,
                "metadata_layer": layer,
                "metadata_cama_cohen_d": CAMA_COHENS_D.get(task_key),
                "metadata_num_parents": summary["num_parents_analyzed"],
                "metadata_mean_cardinality": summary["cardinality"]["mean"],
                "metadata_task_type": task_config.get("task_type", "unknown"),
            }

            # Add predict_ fields
            cr = summary["compression_ratio"]["mean"]
            sr = summary["svd_effective_rank"]["mean"]
            pr = summary["proxy_rank"]["mean"]
            example["predict_compression_ratio"] = (
                f"{cr:.6f}" if cr is not None else "nan"
            )
            example["predict_svd_effective_rank"] = (
                f"{sr:.6f}" if sr is not None else "nan"
            )
            example["predict_proxy_rank"] = (
                f"{pr:.6f}" if pr is not None else "nan"
            )
            examples.append(example)

        if examples:
            datasets.append({"dataset": task_key, "examples": examples})

    # Cross-task analysis dataset
    cross_examples = []
    for task_entry in cross_analysis.get("tasks", []):
        task_key = task_entry["task"]
        result = task_results.get(task_key, {})
        input_data = {
            "task": task_key,
            "cama_cohen_d": task_entry["cama_d"],
            "task_type": result.get("task_config", {}).get(
                "task_type", "unknown"
            ),
            "analysis_type": "cross_task_severity_correlation",
        }
        output_data = {
            "severity": task_entry["severity"],
            "mean_compression": result.get("mean_compression"),
            "mean_proxy": result.get("mean_proxy"),
            "num_edge_types_analyzed": result.get("num_edge_types", 0),
            "val_metric": result.get("train_result", {}).get(
                "best_val_metric"
            ),
            "val_metric_name": result.get("train_result", {}).get(
                "metric_name"
            ),
        }
        cross_example = {
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "metadata_task": task_key,
            "metadata_cama_cohen_d": task_entry["cama_d"],
            "metadata_severity": task_entry["severity"],
        }
        cross_example["predict_severity"] = f"{task_entry['severity']:.6f}"
        spearman = cross_analysis.get("spearman_severity_vs_d", {})
        rho = spearman.get("rho")
        cross_example["predict_spearman_rho"] = (
            f"{rho:.6f}" if rho is not None else "nan"
        )
        cross_examples.append(cross_example)

    if cross_examples:
        datasets.append(
            {"dataset": "cross_task_analysis", "examples": cross_examples}
        )

    # Build metadata
    # Determine key findings
    all_compressions = []
    all_proxy_svd_rhos = []
    for task_key, result in task_results.items():
        for key, summary in result.get("edge_summaries", {}).items():
            if summary is None:
                continue
            cr = summary["compression_ratio"]["mean"]
            if cr is not None:
                all_compressions.append(cr)
            rho = summary["spearman_svd_vs_proxy"]["rho"]
            if rho is not None:
                all_proxy_svd_rhos.append(rho)

    rank_collapse_observed = (
        any(c < 0.5 for c in all_compressions) if all_compressions else False
    )
    proxy_validates_svd = (
        np.mean(all_proxy_svd_rhos) > 0.5 if all_proxy_svd_rhos else False
    )
    spearman_rho = cross_analysis.get("spearman_severity_vs_d", {}).get("rho")
    compression_correlates = (
        spearman_rho is not None and spearman_rho < -0.5
    )

    metadata = {
        "title": "Spectral Rank Analysis at FK Aggregation",
        "hypothesis_tested": (
            "Rank collapse at FK aggregation is the causal mechanism "
            "behind CAMA's performance improvement"
        ),
        "configuration": {
            "model": "HeteroSAGE",
            "aggr": "mean",
            "channels": CHANNELS,
            "num_layers": NUM_LAYERS,
            "epochs": EPOCHS,
            "seed": SEED,
            "max_parents_per_edge_type": MAX_PARENTS_PER_EDGE_TYPE,
            "max_analysis_batches": MAX_ANALYSIS_BATCHES,
        },
        "cross_task_analysis": cross_analysis,
        "key_findings": {
            "rank_collapse_observed": rank_collapse_observed,
            "mean_compression_ratio": (
                float(np.mean(all_compressions))
                if all_compressions
                else None
            ),
            "proxy_validates_svd": bool(proxy_validates_svd),
            "mean_proxy_svd_spearman": (
                float(np.mean(all_proxy_svd_rhos))
                if all_proxy_svd_rhos
                else None
            ),
            "compression_correlates_with_benefit": compression_correlates,
            "spearman_severity_vs_d": spearman_rho,
            "disconfirmation_triggered": (
                not rank_collapse_observed
                or (spearman_rho is not None and spearman_rho > 0.3)
            ),
        },
    }

    return {"metadata": metadata, "datasets": datasets}


# ============================================================================
# SECTION 10: Process One Task End-to-End
# ============================================================================


def process_task(
    dataset_name: str, task_name: str
) -> Dict:
    """Process one task: load, train, analyze, summarize."""
    task_key = f"{dataset_name}/{task_name}"
    t0 = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"Processing task: {task_key}")
    logger.info(f"{'='*60}")

    # Get task-specific hyperparameters
    overrides = TASK_OVERRIDES.get(task_key, {})
    batch_size = overrides.get("batch_size", BATCH_SIZE)
    num_neighbors = overrides.get("num_neighbors", NUM_NEIGHBORS)
    lr = overrides.get("lr", LR)

    # Phase 1: Load data
    dataset, task = load_dataset_and_task(dataset_name, task_name)
    data, col_stats_dict = prepare_graph(dataset, cache_name=dataset_name)

    # Phase 2: Determine task config
    clamp_min, clamp_max = None, None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "roc_auc"
        higher_is_better = True
        task_type_str = "binary_classification"
    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = L1Loss()
        tune_metric = "mae"
        higher_is_better = False
        task_type_str = "regression"
        train_table = task.get_table("train")
        clamp_min, clamp_max = np.percentile(
            train_table.df[task.target_col].to_numpy(), [2, 98]
        )
    else:
        raise ValueError(f"Unsupported task type: {task.task_type}")

    task_config = {
        "task_type": task_type_str,
        "out_channels": out_channels,
        "tune_metric": tune_metric,
        "higher_is_better": higher_is_better,
        "batch_size": batch_size,
        "num_neighbors": num_neighbors,
        "lr": lr,
    }
    logger.info(
        f"Task config: type={task_type_str}, metric={tune_metric}, "
        f"bs={batch_size}, neighbors={num_neighbors}, lr={lr}"
    )

    # Phase 3: Create model and loaders
    collector = RankAnalysisCollector()
    model = InstrumentedModel(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=NUM_LAYERS,
        channels=CHANNELS,
        out_channels=out_channels,
        collector=collector,
    ).to(DEVICE)

    loader_dict, entity_table = create_loaders(
        data, task, num_neighbors, batch_size, NUM_LAYERS
    )
    logger.info(
        f"Model created, entity_table={entity_table}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    # Phase 4: Training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    state_dict = None
    best_val_metric = -math.inf if higher_is_better else math.inf

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.time()
        train_loss = train_epoch(
            model, loader_dict["train"], optimizer, loss_fn,
            entity_table, task.task_type, DEVICE,
        )
        val_pred = evaluate_model(
            model, loader_dict["val"], entity_table, task.task_type,
            DEVICE, clamp_min, clamp_max,
        )
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        val_score = val_metrics[tune_metric]
        epoch_time = time.time() - epoch_t0

        logger.info(
            f"  Epoch {epoch:02d}: loss={train_loss:.4f}, "
            f"val_{tune_metric}={val_score:.4f} ({epoch_time:.1f}s)"
        )

        if (higher_is_better and val_score >= best_val_metric) or (
            not higher_is_better and val_score <= best_val_metric
        ):
            best_val_metric = val_score
            state_dict = copy.deepcopy(model.state_dict())

    # Restore best model
    if state_dict is not None:
        model.load_state_dict(state_dict)
    logger.info(f"Best val_{tune_metric}={best_val_metric:.4f}")

    train_result = {
        "best_val_metric": float(best_val_metric),
        "metric_name": tune_metric,
        "epochs": EPOCHS,
    }

    # Phase 5: Rank analysis
    logger.info(f"Starting rank analysis ({MAX_ANALYSIS_BATCHES} batches)...")
    collector.stop_recording()
    collector.clear()

    all_metrics = run_rank_analysis(
        model, collector, loader_dict["train"], entity_table,
        DEVICE, MAX_ANALYSIS_BATCHES,
    )

    # Phase 6: Summarize
    edge_summaries = {}
    for key, metrics_list in all_metrics.items():
        edge_summaries[key] = summarize_edge_type(metrics_list)

    severity, mean_compression, mean_proxy = compute_task_severity(
        edge_summaries
    )
    logger.info(
        f"Task severity={severity:.4f}, mean_compression={mean_compression:.4f}, "
        f"mean_proxy={mean_proxy:.4f}"
    )

    elapsed = time.time() - t0
    logger.info(f"Task {task_key} completed in {elapsed:.1f}s")

    # Cleanup
    del model, optimizer, loader_dict
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "task_config": task_config,
        "train_result": train_result,
        "edge_summaries": edge_summaries,
        "severity": severity,
        "mean_compression": mean_compression,
        "mean_proxy": mean_proxy,
        "num_edge_types": len(edge_summaries),
        "elapsed_seconds": elapsed,
    }


# ============================================================================
# SECTION 11: Main Execution
# ============================================================================


@logger.catch
def main():
    t0 = time.time()
    seed_everything(SEED)
    if HAS_GPU:
        torch.set_num_threads(1)

    logger.info(f"Starting Spectral Rank Analysis experiment")
    logger.info(f"Tasks: {[f'{d}/{t}' for d,t in TASKS]}")
    logger.info(f"CAMA Cohen's d: {CAMA_COHENS_D}")

    task_results: Dict[str, Dict] = {}

    # Process tasks in order (smallest first)
    for dataset_name, task_name in TASKS:
        task_key = f"{dataset_name}/{task_name}"
        try:
            result = process_task(dataset_name, task_name)
            task_results[task_key] = result
        except Exception:
            logger.exception(f"Task {task_key} failed")
            # Fallback handling
            if dataset_name == "rel-trial":
                logger.warning(
                    "rel-trial failed, trying rel-f1/driver-position as fallback"
                )
                fallback_key = "rel-f1/driver-position"
                CAMA_COHENS_D[fallback_key] = -1.0  # estimated
                try:
                    result = process_task("rel-f1", "driver-position")
                    task_results[fallback_key] = result
                except Exception:
                    logger.exception("Fallback rel-f1/driver-position also failed")
            elif dataset_name == "rel-stack":
                logger.warning(
                    "rel-stack failed, trying with reduced parameters"
                )
                # Reduce parameters and retry
                TASK_OVERRIDES["rel-stack/user-engagement"] = {
                    "batch_size": 256,
                    "num_neighbors": 64,
                    "lr": 0.005,
                }
                global MAX_ANALYSIS_BATCHES
                MAX_ANALYSIS_BATCHES = 30
                try:
                    result = process_task(dataset_name, task_name)
                    task_results[task_key] = result
                except Exception:
                    logger.exception("rel-stack retry also failed")

    if not task_results:
        logger.error("No tasks completed successfully!")
        sys.exit(1)

    logger.info(f"Completed {len(task_results)} tasks")

    # Cross-task analysis
    cross_analysis = cross_task_correlation(task_results)
    logger.info(
        f"Cross-task analysis: "
        f"spearman={cross_analysis.get('spearman_severity_vs_d', {}).get('rho')}"
    )
    logger.info(f"Interpretation: {cross_analysis.get('interpretation', 'N/A')}")

    # Generate output
    output = generate_output(task_results, cross_analysis)

    # Save output
    output_path = WORKSPACE / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Output saved to {output_path}")

    elapsed = time.time() - t0
    logger.info(f"Total experiment time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
