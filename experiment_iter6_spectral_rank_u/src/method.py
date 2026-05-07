#!/usr/bin/env python3
"""Spectral Rank Under Sum vs Mean Aggregation.

Fork of iter-5 exp_id4_it5__opus/method.py. Trains HeteroSAGE with BOTH
sum and mean aggregation (10 epochs, seed=42) on 3 tasks with contrasting
CAMA effect sizes. Instruments the aggregation step to capture child
embeddings, computes SVD-based effective rank and variance-entropy proxy,
then compares rank preservation between aggregation types.

Core hypothesis: Sum aggregation preserves higher effective spectral rank
than mean at FK joins, explaining both sum's performance advantage AND why
CAMA helps mean but not sum/RelGNN.
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
AGGREGATION_TYPES = ["mean", "sum"]  # NEW: both conditions
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
# SECTION 2: Instrumented Aggregation Modules
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


class InstrumentedSumAggregation(Aggregation):
    """Sum aggregation that optionally records inputs for rank analysis."""

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
        output = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")
        self.collector.record(self.key, x, index, dim_size, output)
        return output


# ============================================================================
# SECTION 3: Model Construction (Parameterized by aggr_type)
# ============================================================================

from torch_geometric.nn import HeteroConv, LayerNorm, MLP, SAGEConv
from torch_geometric.typing import EdgeType, NodeType


class InstrumentedHeteroGraphSAGE(torch.nn.Module):
    """HeteroGraphSAGE with per-edge-type instrumented aggregation.

    Accepts aggr_type parameter to switch between mean and sum aggregation.
    """

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        num_layers: int,
        collector: RankAnalysisCollector,
        aggr_type: str = "mean",
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
                if aggr_type == "sum":
                    aggr = InstrumentedSumAggregation(key=key, collector=collector)
                else:
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
    """Full model mirroring RelBench's Model but with instrumented GNN.

    Accepts aggr_type to switch aggregation strategy.
    """

    def __init__(
        self,
        data,
        col_stats_dict: Dict,
        num_layers: int,
        channels: int,
        out_channels: int,
        collector: RankAnalysisCollector,
        aggr_type: str = "mean",
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
            aggr_type=aggr_type,
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
# SECTION 8: Process One (Task, Aggregation Type) Pair
# ============================================================================


def process_task(
    dataset_name: str,
    task_name: str,
    aggr_type: str = "mean",
    data_cache: Optional[Dict] = None,
) -> Dict:
    """Process one task with specified aggregation type.

    If data_cache is provided and contains (data, col_stats_dict, task) for
    this dataset, reuse them instead of reloading.
    """
    task_key = f"{dataset_name}/{task_name}"
    t0 = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"Processing task: {task_key} (aggr_type={aggr_type})")
    logger.info(f"{'='*60}")

    # Get task-specific hyperparameters
    overrides = TASK_OVERRIDES.get(task_key, {})
    batch_size = overrides.get("batch_size", BATCH_SIZE)
    num_neighbors = overrides.get("num_neighbors", NUM_NEIGHBORS)
    lr = overrides.get("lr", LR)

    # Phase 1: Load data (reuse cache if available)
    if data_cache and dataset_name in data_cache:
        logger.info(f"Reusing cached data for {dataset_name}")
        cached = data_cache[dataset_name]
        dataset = cached["dataset"]
        task = cached["task"]
        data = cached["data"]
        col_stats_dict = cached["col_stats_dict"]
    else:
        dataset, task = load_dataset_and_task(dataset_name, task_name)
        data, col_stats_dict = prepare_graph(dataset, cache_name=dataset_name)
        if data_cache is not None:
            data_cache[dataset_name] = {
                "dataset": dataset,
                "task": task,
                "data": data,
                "col_stats_dict": col_stats_dict,
            }

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
        "aggr_type": aggr_type,
    }
    logger.info(
        f"Task config: type={task_type_str}, metric={tune_metric}, "
        f"bs={batch_size}, neighbors={num_neighbors}, lr={lr}, aggr={aggr_type}"
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
        aggr_type=aggr_type,
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
    logger.info(f"Task {task_key} ({aggr_type}) completed in {elapsed:.1f}s")

    # Cleanup
    del model, optimizer, loader_dict, state_dict
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "aggr_type": aggr_type,
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
# SECTION 9: Cross-Aggregation Comparison (NEW)
# ============================================================================


def compare_aggregation_types(
    mean_results: Dict[str, Dict],
    sum_results: Dict[str, Dict],
) -> Tuple[Dict, Dict]:
    """Compare rank metrics between mean and sum aggregation per edge type.

    For each task that has results from BOTH aggregation types,
    match edge types and compute rank ratios.
    """
    comparison = {}
    for task_key in mean_results:
        if task_key not in sum_results:
            continue
        mean_edges = mean_results[task_key]["edge_summaries"]
        sum_edges = sum_results[task_key]["edge_summaries"]

        edge_comparisons = []
        for edge_key in mean_edges:
            if edge_key not in sum_edges:
                continue
            m_summary = mean_edges[edge_key]
            s_summary = sum_edges[edge_key]
            if m_summary is None or s_summary is None:
                continue

            m_svd = m_summary["svd_effective_rank"]["mean"]
            s_svd = s_summary["svd_effective_rank"]["mean"]
            m_proxy = m_summary["proxy_rank"]["mean"]
            s_proxy = s_summary["proxy_rank"]["mean"]
            m_comp = m_summary["compression_ratio"]["mean"]
            s_comp = s_summary["compression_ratio"]["mean"]
            m_card = m_summary["cardinality"]["mean"]

            # Skip if any key metric is None
            if any(v is None for v in [m_svd, s_svd, m_proxy, s_proxy, m_comp, s_comp]):
                continue

            # Skip trivial cardinality=1 edges (no aggregation happens)
            if m_card is not None and m_card <= 1.0:
                continue

            rank_ratio = s_svd / m_svd if m_svd > 0 else None
            proxy_ratio = s_proxy / m_proxy if m_proxy > 0 else None
            compression_diff = s_comp - m_comp  # positive = sum preserves more

            edge_comparisons.append({
                "edge_type": edge_key,
                "mean_cardinality": m_card,
                "mean_svd_rank": m_svd,
                "sum_svd_rank": s_svd,
                "svd_rank_ratio": rank_ratio,  # >1 means sum has higher rank
                "mean_proxy": m_proxy,
                "sum_proxy": s_proxy,
                "proxy_ratio": proxy_ratio,
                "mean_compression": m_comp,
                "sum_compression": s_comp,
                "compression_diff": compression_diff,
            })

        # Aggregate across non-trivial edge types
        if edge_comparisons:
            ratios = [e["svd_rank_ratio"] for e in edge_comparisons if e["svd_rank_ratio"] is not None]
            comp_diffs = [e["compression_diff"] for e in edge_comparisons if e["compression_diff"] is not None]
            comparison[task_key] = {
                "num_nontrivial_edges": len(edge_comparisons),
                "edge_comparisons": edge_comparisons,
                "aggregate": {
                    "mean_svd_rank_ratio": float(np.mean(ratios)) if ratios else None,
                    "median_svd_rank_ratio": float(np.median(ratios)) if ratios else None,
                    "mean_compression_diff": float(np.mean(comp_diffs)) if comp_diffs else None,
                    "fraction_sum_higher_rank": sum(1 for r in ratios if r > 1.0) / len(ratios) if ratios else None,
                    "cama_cohen_d": CAMA_COHENS_D.get(task_key),
                },
            }

    # Overall aggregate across all tasks
    all_ratios = []
    all_diffs = []
    for task_key, comp in comparison.items():
        agg = comp["aggregate"]
        if agg["mean_svd_rank_ratio"] is not None:
            all_ratios.append(agg["mean_svd_rank_ratio"])
        if agg["mean_compression_diff"] is not None:
            all_diffs.append(agg["mean_compression_diff"])

    overall = {
        "num_tasks_compared": len(comparison),
        "grand_mean_rank_ratio": float(np.mean(all_ratios)) if all_ratios else None,
        "grand_mean_compression_diff": float(np.mean(all_diffs)) if all_diffs else None,
        "hypothesis_supported": (
            float(np.mean(all_ratios)) > 1.0 if all_ratios else None
        ),  # sum should have HIGHER rank
        "interpretation": "",  # filled below
    }

    if all_ratios:
        grand_ratio = np.mean(all_ratios)
        if grand_ratio > 1.2:
            overall["interpretation"] = (
                f"Strong support: sum aggregation preserves {(grand_ratio-1)*100:.0f}% "
                "more effective rank than mean. Sum's superior rank preservation "
                "explains both its performance advantage AND why CAMA helps mean "
                "but is unnecessary for sum."
            )
        elif grand_ratio > 1.0:
            overall["interpretation"] = (
                f"Moderate support: sum preserves {(grand_ratio-1)*100:.0f}% more "
                "rank. Directionally consistent with hypothesis."
            )
        elif grand_ratio < 1.0:
            overall["interpretation"] = (
                "DISCONFIRMED: mean aggregation produces HIGHER effective rank "
                "than sum. The information-compression mechanism is not the "
                "primary explanation for sum's performance advantage."
            )
        else:
            overall["interpretation"] = "Inconclusive: negligible difference."
    else:
        overall["interpretation"] = "No tasks available for comparison."

    return comparison, overall


# ============================================================================
# SECTION 10: Output Generation (exp_gen_sol_out.json schema)
# ============================================================================


def generate_output(
    mean_results: Dict[str, Dict],
    sum_results: Dict[str, Dict],
    comparison: Dict,
    overall: Dict,
) -> Dict:
    """Generate method_out.json in required schema format."""
    datasets = []

    # Dataset per (aggr_type, task) pair: per-edge-type measurements
    for aggr_type, results in [("mean", mean_results), ("sum", sum_results)]:
        for task_key, result in results.items():
            examples = []
            edge_summaries = result.get("edge_summaries", {})
            task_config = result.get("task_config", {})

            for edge_key, summary in edge_summaries.items():
                if summary is None:
                    continue
                parts = edge_key.split("/", 1)
                layer = parts[0] if len(parts) > 1 else "unknown"
                edge_desc = parts[1] if len(parts) > 1 else edge_key

                input_data = {
                    "task": task_key,
                    "aggr_type": aggr_type,
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
                    "metadata_aggr_type": aggr_type,
                    "metadata_edge_type": edge_key,
                    "metadata_layer": layer,
                    "metadata_mean_cardinality": summary["cardinality"]["mean"],
                }

                cr = summary["compression_ratio"]["mean"]
                sr = summary["svd_effective_rank"]["mean"]
                example["predict_compression_ratio"] = (
                    f"{cr:.6f}" if cr is not None else "nan"
                )
                example["predict_svd_effective_rank"] = (
                    f"{sr:.6f}" if sr is not None else "nan"
                )
                examples.append(example)

            if examples:
                datasets.append({
                    "dataset": f"{aggr_type}/{task_key}",
                    "examples": examples,
                })

    # Cross-aggregation comparison dataset
    cross_examples = []
    for task_key, comp in comparison.items():
        agg = comp["aggregate"]
        input_data = {
            "task": task_key,
            "analysis_type": "sum_vs_mean_rank_comparison",
        }
        output_data = {
            "num_nontrivial_edges": comp["num_nontrivial_edges"],
            "mean_svd_rank_ratio": agg["mean_svd_rank_ratio"],
            "median_svd_rank_ratio": agg["median_svd_rank_ratio"],
            "mean_compression_diff": agg["mean_compression_diff"],
            "fraction_sum_higher_rank": agg["fraction_sum_higher_rank"],
        }
        cross_examples.append({
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "metadata_task": task_key,
            "metadata_cama_cohen_d": CAMA_COHENS_D.get(task_key),
            "predict_rank_ratio": (
                f"{agg['mean_svd_rank_ratio']:.6f}"
                if agg["mean_svd_rank_ratio"] is not None else "nan"
            ),
        })
    if cross_examples:
        datasets.append({
            "dataset": "cross_aggregation_comparison",
            "examples": cross_examples,
        })

    metadata = {
        "title": "Spectral Rank Under Sum vs Mean Aggregation",
        "hypothesis_tested": (
            "Sum aggregation preserves higher effective rank than mean at FK joins"
        ),
        "configuration": {
            "model": "HeteroSAGE",
            "channels": CHANNELS,
            "num_layers": NUM_LAYERS,
            "epochs": EPOCHS,
            "seed": SEED,
            "aggregation_types": AGGREGATION_TYPES,
            "max_parents_per_edge_type": MAX_PARENTS_PER_EDGE_TYPE,
            "max_analysis_batches": MAX_ANALYSIS_BATCHES,
        },
        "overall_comparison": overall,
        "key_findings": {
            "sum_preserves_more_rank": overall.get("hypothesis_supported"),
            "grand_mean_rank_ratio": overall.get("grand_mean_rank_ratio"),
            "grand_mean_compression_diff": overall.get("grand_mean_compression_diff"),
        },
    }
    return {"metadata": metadata, "datasets": datasets}


# ============================================================================
# SECTION 11: Smoke Test
# ============================================================================


def smoke_test_aggregations():
    """Quick smoke test for InstrumentedSumAggregation correctness."""
    logger.info("Running smoke test for InstrumentedSumAggregation...")

    collector = RankAnalysisCollector()
    collector.start_recording()

    # Test data: 20 vectors of dim 8, assigned to 5 groups
    torch.manual_seed(SEED)
    x = torch.randn(20, 8)
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    dim_size = 5

    # Test sum aggregation
    sum_aggr = InstrumentedSumAggregation(key="test_sum", collector=collector)
    sum_out = sum_aggr(x, index=index, dim_size=dim_size)

    assert sum_out.shape == (5, 8), f"Expected (5,8), got {sum_out.shape}"

    # Verify manually
    for g in range(5):
        mask = index == g
        expected = x[mask].sum(dim=0)
        actual = sum_out[g]
        diff = (expected - actual).abs().max().item()
        assert diff < 1e-5, f"Group {g}: max diff = {diff}"

    # Test mean aggregation
    collector_m = RankAnalysisCollector()
    collector_m.start_recording()
    mean_aggr = InstrumentedMeanAggregation(key="test_mean", collector=collector_m)
    mean_out = mean_aggr(x, index=index, dim_size=dim_size)

    assert mean_out.shape == (5, 8), f"Expected (5,8), got {mean_out.shape}"
    for g in range(5):
        mask = index == g
        expected = x[mask].mean(dim=0)
        actual = mean_out[g]
        diff = (expected - actual).abs().max().item()
        assert diff < 1e-5, f"Group {g}: max diff = {diff}"

    # Verify collector recorded data
    assert "test_sum" in collector.records, "Sum collector didn't record"
    assert len(collector.records["test_sum"]) == 1, "Expected 1 record"
    rec = collector.records["test_sum"][0]
    assert rec[0].shape == (20, 8), f"Recorded x shape: {rec[0].shape}"
    assert rec[1].shape == (20,), f"Recorded index shape: {rec[1].shape}"

    collector.stop_recording()

    logger.info("Smoke test PASSED: both aggregations correct, collector records data")


# ============================================================================
# SECTION 12: Main Execution
# ============================================================================


@logger.catch
def main():
    t0 = time.time()
    seed_everything(SEED)
    if HAS_GPU:
        torch.set_num_threads(1)

    logger.info("Starting Spectral Rank: Sum vs Mean Aggregation experiment")
    logger.info(f"Tasks: {[f'{d}/{t}' for d, t in TASKS]}")
    logger.info(f"Aggregation types: {AGGREGATION_TYPES}")
    logger.info(f"CAMA Cohen's d: {CAMA_COHENS_D}")

    # Run smoke test first
    smoke_test_aggregations()

    # Store results keyed by task_key -> result
    mean_results: Dict[str, Dict] = {}
    sum_results: Dict[str, Dict] = {}

    # Data cache: reuse loaded data/graph across aggregation types for same dataset
    data_cache: Dict[str, Dict] = {}

    # Process all 6 runs: 3 tasks x 2 aggregation types
    # Order: process both aggregation types per task to maximize data reuse
    for dataset_name, task_name in TASKS:
        task_key = f"{dataset_name}/{task_name}"
        task_start = time.time()

        for aggr_type in AGGREGATION_TYPES:
            try:
                result = process_task(
                    dataset_name, task_name,
                    aggr_type=aggr_type,
                    data_cache=data_cache,
                )
                if aggr_type == "mean":
                    mean_results[task_key] = result
                else:
                    sum_results[task_key] = result
            except Exception:
                logger.exception(f"Task {task_key} ({aggr_type}) failed")
                # Fallback handling
                if dataset_name == "rel-trial":
                    logger.warning(
                        f"rel-trial ({aggr_type}) failed, "
                        "trying rel-f1/driver-position as fallback"
                    )
                    fallback_key = "rel-f1/driver-position"
                    CAMA_COHENS_D[fallback_key] = -1.0
                    try:
                        result = process_task(
                            "rel-f1", "driver-position",
                            aggr_type=aggr_type,
                            data_cache=data_cache,
                        )
                        if aggr_type == "mean":
                            mean_results[fallback_key] = result
                        else:
                            sum_results[fallback_key] = result
                    except Exception:
                        logger.exception(
                            f"Fallback rel-f1/driver-position ({aggr_type}) also failed"
                        )
                elif dataset_name == "rel-stack":
                    logger.warning(
                        f"rel-stack ({aggr_type}) failed, "
                        "trying with reduced parameters"
                    )
                    TASK_OVERRIDES["rel-stack/user-engagement"] = {
                        "batch_size": 256,
                        "num_neighbors": 64,
                        "lr": 0.005,
                    }
                    global MAX_ANALYSIS_BATCHES
                    old_mab = MAX_ANALYSIS_BATCHES
                    MAX_ANALYSIS_BATCHES = 30
                    try:
                        result = process_task(
                            dataset_name, task_name,
                            aggr_type=aggr_type,
                            data_cache=data_cache,
                        )
                        if aggr_type == "mean":
                            mean_results[task_key] = result
                        else:
                            sum_results[task_key] = result
                    except Exception:
                        logger.exception(
                            f"rel-stack retry ({aggr_type}) also failed"
                        )
                    MAX_ANALYSIS_BATCHES = old_mab

        # Free data cache for this dataset to save memory
        # (unless the same dataset is used later -- rel-f1 might be reused)
        task_elapsed = time.time() - task_start
        logger.info(
            f"Dataset {dataset_name} both conditions done in {task_elapsed:.1f}s"
        )

        # Check time budget: if >80 min elapsed, skip remaining mean replications
        total_elapsed = time.time() - t0
        if total_elapsed > 80 * 60:
            logger.warning(
                f"Time pressure: {total_elapsed/60:.1f} min elapsed. "
                "May skip remaining tasks."
            )

        # Clean up data cache for this dataset to free memory
        if dataset_name in data_cache:
            del data_cache[dataset_name]
            gc.collect()
            torch.cuda.empty_cache()

    total_mean = len(mean_results)
    total_sum = len(sum_results)
    logger.info(f"Completed: {total_mean} mean + {total_sum} sum tasks")

    if not mean_results and not sum_results:
        logger.error("No tasks completed successfully!")
        # Produce minimal diagnostics output
        error_output = {
            "metadata": {
                "title": "Spectral Rank Under Sum vs Mean Aggregation",
                "error": "All tasks failed",
                "hardware": {
                    "cpus": NUM_CPUS,
                    "ram_gb": TOTAL_RAM_GB,
                    "gpu": HAS_GPU,
                    "vram_gb": VRAM_GB,
                },
            },
            "datasets": [{
                "dataset": "error",
                "examples": [{
                    "input": json.dumps({"error": "All tasks failed"}),
                    "output": json.dumps({"error": "No results"}),
                }],
            }],
        }
        output_path = WORKSPACE / "method_out.json"
        output_path.write_text(json.dumps(error_output, indent=2))
        sys.exit(1)

    # Cross-aggregation comparison
    comparison, overall = compare_aggregation_types(mean_results, sum_results)
    logger.info(
        f"Cross-aggregation comparison: "
        f"grand_mean_rank_ratio={overall.get('grand_mean_rank_ratio')}"
    )
    logger.info(f"Interpretation: {overall.get('interpretation', 'N/A')}")

    # Generate and save output
    output = generate_output(mean_results, sum_results, comparison, overall)

    output_path = WORKSPACE / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Output saved to {output_path}")

    elapsed = time.time() - t0
    logger.info(f"Total experiment time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
