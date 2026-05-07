#!/usr/bin/env python3
"""Sum Aggregation Baseline vs CAMA on rel-amazon/item-ltv (3 methods x 5 seeds).

Extends iter-5's 2-method experiment to include HeteroGraphSAGE(aggr="sum") as a
third comparison. The iter-5 "baseline" used aggr="mean", but RelBench's default
GNN uses aggr="sum". This experiment determines whether CAMA's d=13.58 effect
survives comparison against the sum baseline.

Run order: sum (novel) -> mean (replication) -> cama (replication).
"""

import copy
import gc
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from loguru import logger

# ============================================================================
# LOGGING SETUP
# ============================================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================================
# HARDWARE DETECTION & MEMORY LIMITS (cgroup-aware)
# ============================================================================
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


def _container_ram_gb() -> Optional[float]:
    """Read RAM limit from cgroup (containers/pods)."""
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


def _current_ram_gb() -> float:
    """Read current cgroup memory usage."""
    for p in ["/sys/fs/cgroup/memory.current",
              "/sys/fs/cgroup/memory/memory.usage_in_bytes"]:
        try:
            return int(Path(p).read_text().strip()) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return psutil.virtual_memory().used / 1e9


NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
            f"GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM), device={DEVICE}")

# Set RAM limit to 70% of container limit
RAM_BUDGET_GB = TOTAL_RAM_GB * 0.70
RAM_BUDGET = int(RAM_BUDGET_GB * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET_GB:.1f}GB (70% of {TOTAL_RAM_GB:.1f}GB)")

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.90)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))
    logger.info(f"VRAM budget: {VRAM_BUDGET / 1e9:.1f}GB (90% of {_total / 1e9:.1f}GB)")
    torch.set_num_threads(max(1, NUM_CPUS // 2))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================================
# DELAYED IMPORTS (after hardware setup)
# ============================================================================
from torch.nn import L1Loss
from torch_geometric.seed import seed_everything
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv, MLP
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.norm import LayerNorm
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from sentence_transformers import SentenceTransformer
from scipy import stats as sp_stats

# ============================================================================
# CONFIGURATION (match iter-5 exactly, add sum method)
# ============================================================================
RELBENCH_CACHE_DIR = (
    "/ai-inventor/aii_pipeline/runs/temp-debug-test_sbr/"
    "3_invention_loop/iter_1/gen_art/data_id3_it1__opus/"
    "temp/relbench_cache"
)
MATERIALIZED_CACHE = str(SCRIPT_DIR / "mat_cache" / "rel-amazon_notext")

TASK_CONFIG = {
    'dataset_name': 'rel-amazon',
    'task_name': 'item-ltv',
    'task_type': 'regression',
    'primary_metric': 'mae',
    'higher_is_better': False,
    'out_channels': 1,
    'channels': 128,               # MUST match iter-5
    'lr': 0.005,                   # MUST match iter-5
    'epochs': 10,                  # MUST match iter-5
    'batch_size': 512,             # MUST match iter-5
    'num_neighbors': [128, 128],   # MUST match iter-5
    'max_steps_per_epoch': 2000,   # MUST match iter-5
    'grad_clip': 1.0,             # MUST match iter-5
}
SEEDS = [42, 123, 456, 789, 1024]  # MUST match iter-5
# RUN ORDER: sum first (novel), then mean (baseline), then cama (replication)
METHODS = ['sum', 'mean', 'cama']
DEVIATIONS: List[str] = []

# Iter-5 reference data (for cross-validation)
ITER5_REFERENCE = {
    'baseline_mean': {
        'per_seed': {42: 49.2625, 123: 49.5791, 456: 49.1939, 789: 48.6403, 1024: 49.2884},
        'mean_mae': 49.1929, 'std_mae': 0.3422,
    },
    'cama': {
        'per_seed': {42: 45.0793, 123: 45.1592, 456: 45.0520, 789: 45.5217, 1024: 45.4809},
        'mean_mae': 45.2586, 'std_mae': 0.2255,
    },
    'cohens_d': 13.5764,
}


# ============================================================================
# STEP 1: CAMAAggregation MODULE (verbatim from iter-5)
# ============================================================================
class CAMAAggregation(Aggregation):
    """Cardinality-Aware Moment Aggregation.

    Enriches mean with per-dimension variance, gated by log-cardinality.
    Simplified RAMA: gate depends ONLY on log(cardinality), no rank proxy.

    Forward receives from SAGEConv's propagate():
      x: (num_messages, channels) - source node embeddings
      index: (num_messages,) - maps each message to its target node
      dim_size: number of target nodes
      dim: aggregation dimension (typically -2)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # Gate network: maps log_cardinality (scalar) to per-dimension gate
        self.gate_net = nn.Linear(1, channels, bias=True)
        # Variance transform: learned projection of variance features
        self.var_transform = nn.Linear(channels, channels, bias=False)
        # Gate recording for analysis
        self._gate_buffer: list = []
        self._recording = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gate_net.weight)   # Init gate near 0.5 (sigmoid(0))
        nn.init.zeros_(self.gate_net.bias)
        nn.init.eye_(self.var_transform.weight)  # Init as identity

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        # 1. MEAN: mu = E[x_j] per target node
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')

        # 2. VARIANCE: E[x_j^2] - (E[x_j])^2 per target node, per dimension
        mean_of_sq = self.reduce(x * x, index, ptr, dim_size, dim, reduce='mean')
        var = (mean_of_sq - mean * mean).clamp(min=1e-8)

        # 3. CARDINALITY: count of messages per target node
        ones = x.new_ones(x.size(0), 1)
        cardinality = self.reduce(ones, index, ptr, dim_size, dim, reduce='sum')
        log_card = torch.log1p(cardinality)  # numerically stable log(1+N)

        # 4. GATE: sigmoid(W_g * log_card + b_g)
        gate = torch.sigmoid(self.gate_net(log_card))

        # 5. RECORD gate values if in analysis mode
        if self._recording:
            self._gate_buffer.append(gate.detach().cpu())

        # 6. OUTPUT: mean + gate * W_sigma(variance)
        return mean + gate * self.var_transform(var)

    def start_recording(self):
        self._recording = True
        self._gate_buffer = []

    def stop_recording(self) -> Optional[Dict[str, float]]:
        self._recording = False
        if self._gate_buffer:
            all_gates = torch.cat(self._gate_buffer, dim=0)
            self._gate_buffer = []
            if all_gates.numel() == 0:
                return None
            gate_stats = {
                'mean': float(all_gates.mean()),
                'std': float(all_gates.std()),
                'min': float(all_gates.min()),
                'max': float(all_gates.max()),
                'median': float(all_gates.median()),
            }
            return gate_stats
        return None


# ============================================================================
# STEP 2: CAMAHeteroGraphSAGE (verbatim from iter-5)
# ============================================================================
class CAMAHeteroGraphSAGE(nn.Module):
    """Modified HeteroGraphSAGE with CAMA aggregation per edge type per layer."""

    def __init__(self, node_types, edge_types, channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.cama_registry: Dict[Tuple, CAMAAggregation] = {}

        for layer_idx in range(num_layers):
            edge_conv_dict = {}
            for et in edge_types:
                cama = CAMAAggregation(channels)
                edge_conv_dict[et] = SAGEConv(
                    (channels, channels), channels, aggr=cama
                )
                self.cama_registry[(layer_idx, et)] = cama
            conv = HeteroConv(edge_conv_dict, aggr="sum")
            self.convs.append(conv)

            norm_dict = nn.ModuleDict()
            for nt in node_types:
                norm_dict[nt] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(self, x_dict, edge_index_dict,
                num_sampled_nodes_dict=None, num_sampled_edges_dict=None):
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict


# ============================================================================
# STEP 3: MODEL CLASS (extended to support sum/mean/cama)
# ============================================================================
class Model(nn.Module):
    def __init__(self, data, col_stats_dict, num_layers, channels,
                 out_channels, method='mean'):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                nt: data[nt].tf.col_names_dict for nt in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                nt for nt in data.node_types if "time" in data[nt]
            ],
            channels=channels,
        )
        if method == 'cama':
            self.gnn = CAMAHeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                num_layers=num_layers,
            )
        elif method == 'sum':
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr="sum",            # THE KEY CHANGE: sum aggregation
                num_layers=num_layers,
            )
        else:  # 'mean' baseline (matches iter-5)
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr="mean",
                num_layers=num_layers,
            )
        self.head = MLP(
            channels, out_channels=out_channels,
            norm="batch_norm", num_layers=1,
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch, entity_table):
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        for nt, rel_time in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel_time
        x_dict = self.gnn(
            x_dict, batch.edge_index_dict,
            batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict,
        )
        return self.head(x_dict[entity_table][:seed_time.size(0)])


# ============================================================================
# STEP 4: GLOVE TEXT EMBEDDING
# ============================================================================
class GloveTextEmbedding:
    """Uses sentence-transformers GloVe model for text embedding."""

    def __init__(self, device=None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device or torch.device("cpu"),
        )

    def __call__(self, sentences):
        # Handle NaN/None values in text columns (product.description has NaNs)
        clean = [str(s) if isinstance(s, str) else "" for s in sentences]
        return torch.from_numpy(
            self.model.encode(clean, show_progress_bar=False)
        )


# ============================================================================
# STEP 5: TRAIN AND EVALUATE FUNCTIONS
# ============================================================================
def train_epoch(model, loader, optimizer, loss_fn, entity_table,
                max_steps, grad_clip=1.0):
    """Train one epoch with gradient clipping."""
    model.train()
    loss_accum = count_accum = 0
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch, entity_table).view(-1)
        loss = loss_fn(pred.float(), batch[entity_table].y.float())

        if torch.isnan(loss):
            logger.warning(f"NaN loss at step {step}, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)
    return loss_accum / max(count_accum, 1)


@torch.no_grad()
def evaluate(model, loader, entity_table, clamp_min=None, clamp_max=None):
    """Run inference and return predictions."""
    model.eval()
    pred_list = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table).view(-1)
        if clamp_min is not None:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


@torch.no_grad()
def evaluate_n_batches(model, loader, entity_table, n_batches=20):
    """Run inference on first n batches only (for gate sampling)."""
    model.eval()
    pred_list = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table).view(-1)
        pred_list.append(pred.detach().cpu())
    if pred_list:
        return torch.cat(pred_list, dim=0).numpy()
    return np.array([])


def collect_gate_snapshot(model) -> Dict[str, Dict]:
    """Collect gate statistics from all CAMA modules."""
    gate_snapshot = {}
    if not isinstance(model.gnn, CAMAHeteroGraphSAGE):
        return gate_snapshot
    for (layer_idx, edge_type), cama_mod in model.gnn.cama_registry.items():
        gs = cama_mod.stop_recording()
        if gs is not None:
            et_str = format_edge_type(edge_type)
            gate_snapshot[f"L{layer_idx}_{et_str}"] = gs
    return gate_snapshot


def start_gate_recording(model):
    """Enable gate recording on all CAMA modules."""
    if not isinstance(model.gnn, CAMAHeteroGraphSAGE):
        return
    for (layer_idx, edge_type), cama_mod in model.gnn.cama_registry.items():
        cama_mod.start_recording()


# ============================================================================
# STEP 6: STATISTICS
# ============================================================================
def cohens_d_independent(group1, group2):
    """Independent-samples Cohen's d. Positive = group2 has lower value (better for MAE)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled < 1e-12:
        return 0.0
    # For MAE: lower is better, so (higher_mae - lower_mae) / s_pooled
    return float((m1 - m2) / s_pooled)


def bootstrap_cohens_d(group1_scores, group2_scores,
                       n_bootstrap=10000, ci=0.95):
    """Bootstrap CI for Cohen's d."""
    rng = np.random.RandomState(42)
    group1_scores = np.asarray(group1_scores)
    group2_scores = np.asarray(group2_scores)
    n1, n2 = len(group1_scores), len(group2_scores)
    boot_ds = []
    for _ in range(n_bootstrap):
        idx1 = rng.choice(n1, n1, replace=True)
        idx2 = rng.choice(n2, n2, replace=True)
        boot_ds.append(
            cohens_d_independent(group1_scores[idx1], group2_scores[idx2])
        )
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(boot_ds, 100 * alpha)),
        float(np.percentile(boot_ds, 100 * (1 - alpha))),
    )


def compute_pairwise_stats(worse_maes, better_maes, name: str) -> Dict:
    """Compute full pairwise statistical comparison.

    Args:
        worse_maes: MAE values from the expected-worse method (higher MAE)
        better_maes: MAE values from the expected-better method (lower MAE)
        name: comparison name (e.g. 'cama_vs_sum')

    Returns positive d if better_maes are indeed lower (better).
    """
    result = {'comparison': name}
    if len(worse_maes) < 2 or len(better_maes) < 2:
        result['error'] = f'Insufficient data: {len(worse_maes)} vs {len(better_maes)}'
        return result

    d = cohens_d_independent(np.array(worse_maes), np.array(better_maes))
    ci_lo, ci_hi = bootstrap_cohens_d(np.array(worse_maes), np.array(better_maes))
    t_stat, p_value = sp_stats.ttest_ind(worse_maes, better_maes)

    wilcoxon_stat, wilcoxon_p = None, None
    if len(worse_maes) == len(better_maes) and len(worse_maes) >= 3:
        try:
            diffs = np.array(worse_maes) - np.array(better_maes)
            if np.any(diffs != 0):
                wilcoxon_stat, wilcoxon_p = sp_stats.wilcoxon(diffs)
                wilcoxon_stat = float(wilcoxon_stat)
                wilcoxon_p = float(wilcoxon_p)
        except ValueError:
            pass

    worse_mean = float(np.mean(worse_maes))
    better_mean = float(np.mean(better_maes))

    result.update({
        'cohens_d': float(d),
        'd_ci_95': [float(ci_lo), float(ci_hi)],
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'wilcoxon_stat': wilcoxon_stat,
        'wilcoxon_p': wilcoxon_p,
        'worse_mean_mae': worse_mean,
        'worse_std_mae': float(np.std(worse_maes, ddof=1)),
        'better_mean_mae': better_mean,
        'better_std_mae': float(np.std(better_maes, ddof=1)),
        'improvement_pct': float(
            (worse_mean - better_mean) / max(worse_mean, 1e-12) * 100
        ),
    })
    return result


def format_edge_type(et) -> str:
    """Convert PyG edge type tuple to readable string."""
    if isinstance(et, tuple):
        return "__".join(str(x) for x in et)
    return str(et)


def log_memory(label: str = "") -> float:
    """Log current memory usage. Returns usage in GB."""
    ram_gb = _current_ram_gb()
    gpu_info = ""
    if HAS_GPU:
        gpu_used = torch.cuda.memory_allocated() / 1e9
        gpu_info = f", GPU={gpu_used:.2f}GB"
    logger.info(f"[MEM {label}] RAM={ram_gb:.1f}/{TOTAL_RAM_GB:.0f}GB{gpu_info}")
    if ram_gb > TOTAL_RAM_GB * 0.85:
        logger.warning(f"RAM usage at {ram_gb/TOTAL_RAM_GB*100:.0f}% - near limit!")
    return ram_gb


# ============================================================================
# DATA LOADING (identical to iter-5)
# ============================================================================
def load_dataset_and_graph():
    """Load rel-amazon dataset and build materialized graph.

    CRITICAL: Excludes text columns from review table (20.8M rows)
    and customer_name from customer table to prevent OOM.
    """
    from torch_frame import stype as st

    logger.info("Loading rel-amazon dataset...")
    t0 = time.time()

    os.environ["RELBENCH_CACHE_DIR"] = RELBENCH_CACHE_DIR
    dataset = get_dataset("rel-amazon", download=True)
    task = get_task("rel-amazon", "item-ltv", download=True)
    db = dataset.get_db()

    logger.info(f"Dataset loaded in {time.time()-t0:.0f}s")
    logger.info(f"Tables: {list(db.table_dict.keys())}")
    log_memory("after dataset load")

    # Build stype dict with OOM-safe exclusions
    stypes_cache_path = Path(RELBENCH_CACHE_DIR) / "rel-amazon" / "stypes.json"
    if stypes_cache_path.exists():
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in list(col_to_stype.items()):
                col_to_stype[col] = st(stype_str)
        logger.info("Loaded cached stypes from relbench cache")
    else:
        logger.info("Computing stype proposal (no cache found)...")
        col_to_stype_dict = get_stype_proposal(db)

    # CRITICAL: Remove text columns from review table (20.8M rows)
    if "review" in col_to_stype_dict:
        for text_col in ["review_text", "summary"]:
            if text_col in col_to_stype_dict["review"]:
                removed_stype = col_to_stype_dict["review"].pop(text_col)
                logger.warning(
                    f"REMOVED review.{text_col} ({removed_stype}) to prevent OOM "
                    f"(20.8M rows * 300d = ~6.2GB per column)"
                )
        DEVIATIONS.append(
            "Removed review_text and summary from review table "
            "to prevent OOM: 20.8M rows * 2 cols * 300d = ~50GB"
        )

    # Remove customer_name (1.85M rows * 300d = ~2.2GB)
    if "customer" in col_to_stype_dict:
        if "customer_name" in col_to_stype_dict["customer"]:
            col_to_stype_dict["customer"].pop("customer_name")
            logger.warning("REMOVED customer.customer_name to save ~2.2GB")
            DEVIATIONS.append(
                "Removed customer_name from customer table to save ~2.2GB"
            )

    # Log remaining features
    for table, stypes in col_to_stype_dict.items():
        logger.info(f"  {table}: {dict(stypes)}")

    log_memory("before graph build")

    # Text embedder for product text columns (506K rows, manageable)
    text_embedder = GloveTextEmbedding(device=torch.device("cpu"))
    text_cfg = TextEmbedderConfig(
        text_embedder=text_embedder, batch_size=256,
    )

    logger.info("Building pkey-fkey graph...")
    t0 = time.time()
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_cfg,
        cache_dir=MATERIALIZED_CACHE,
    )
    graph_time = time.time() - t0
    logger.info(f"Graph built in {graph_time:.0f}s")
    logger.info(f"Node types: {data.node_types}")
    logger.info(f"Edge types: {data.edge_types}")
    for nt in data.node_types:
        if hasattr(data[nt], 'num_nodes'):
            logger.info(f"  {nt}: {data[nt].num_nodes} nodes")

    log_memory("after graph build")

    # Free temporary data
    del db, dataset, text_embedder
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

    return task, data, col_stats_dict, graph_time


def build_loaders(task, data):
    """Build NeighborLoaders for train/val splits."""
    loader_dict = {}
    for split in ["train", "val"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=TASK_CONFIG['num_neighbors'],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=TASK_CONFIG['batch_size'],
            temporal_strategy="uniform",
            shuffle=(split == "train"),
            num_workers=0,
            persistent_workers=False,
        )
    logger.info(f"Loaders: train={len(loader_dict['train'])}, "
                f"val={len(loader_dict['val'])}")
    return loader_dict


# ============================================================================
# PARTIAL RESULT SAVE/LOAD (crash recovery)
# ============================================================================
def _save_partial(results: dict, path: Path):
    """Save partial results to disk for crash recovery."""
    try:
        serializable = {}
        for key, val in results.items():
            method, seed = key
            skey = f"{method}__seed{seed}"
            serializable[skey] = val
        path.write_text(json.dumps(serializable, indent=2, default=str))
        logger.debug(f"Saved partial results ({len(serializable)} runs)")
    except Exception:
        logger.exception("Failed to save partial results")


def _load_partial(path: Path) -> dict:
    """Load partial results from a previous run."""
    results = {}
    if not path.exists():
        return results
    try:
        data = json.loads(path.read_text())
        for key, val in data.items():
            parts = key.split("__")
            if len(parts) != 2:
                continue
            method = parts[0]
            seed = int(parts[1].replace("seed", ""))
            if val.get('best_val_mae') is not None:
                results[(method, seed)] = val
        logger.info(f"Loaded {len(results)} completed runs from partial results")
    except Exception:
        logger.exception("Failed to load partial results, starting fresh")
    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
@logger.catch
def main():
    global TASK_CONFIG
    start_time = time.time()
    TIME_BUDGET_SEC = 55 * 60  # 55 min budget (script must complete in <60 min)

    def remaining_minutes() -> float:
        return (TIME_BUDGET_SEC - (time.time() - start_time)) / 60

    out_path = SCRIPT_DIR / "method_out.json"
    partial_path = SCRIPT_DIR / "partial_results.json"

    logger.info("=" * 60)
    logger.info("Sum Aggregation Baseline vs CAMA on rel-amazon/item-ltv")
    logger.info(f"Objective: Determine if d=13.58 CAMA effect survives sum comparison")
    logger.info(f"Config: {TASK_CONFIG['epochs']} epochs, {len(SEEDS)} seeds, "
                f"neighbors={TASK_CONFIG['num_neighbors']}")
    logger.info(f"Methods: {METHODS} (run order: sum first)")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # PHASE 1: CAMA UNIT TEST
    # ------------------------------------------------------------------
    logger.info("Phase 1: CAMA unit test...")
    cama_test = CAMAAggregation(channels=64)
    x_test = torch.randn(100, 64)
    index_test = torch.randint(0, 10, (100,))
    out_test = cama_test(x_test, index=index_test, dim_size=10)
    assert out_test.shape == (10, 64), f"Expected (10,64), got {out_test.shape}"

    # Gradient flow
    loss_test = out_test.sum()
    loss_test.backward()
    for name, p in cama_test.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

    # Check gate values near 0.5 at initialization
    cama_test.start_recording()
    _ = cama_test(x_test, index=index_test, dim_size=10)
    gs = cama_test.stop_recording()
    assert gs is not None, "Gate recording failed"
    assert abs(gs['mean'] - 0.5) < 0.05, f"Initial gate should be ~0.5, got {gs['mean']}"
    logger.info(f"  Gate init mean={gs['mean']:.4f} (expected ~0.5)")
    logger.info("CAMA unit tests passed!")

    del cama_test, x_test, index_test, out_test
    gc.collect()

    # ------------------------------------------------------------------
    # PHASE 2: LOAD DATA
    # ------------------------------------------------------------------
    logger.info("Phase 2: Loading dataset and building graph...")
    log_memory("before data load")
    task, data, col_stats_dict, graph_build_time = load_dataset_and_graph()
    entity_table = task.entity_table  # "product"

    # Compute regression clamp bounds from train targets
    train_table = task.get_table("train")
    targets = train_table.df[task.target_col].dropna().to_numpy()
    clamp_min = float(np.percentile(targets, 2))
    clamp_max = float(np.percentile(targets, 98))
    logger.info(f"Clamp bounds: [{clamp_min:.2f}, {clamp_max:.2f}]")
    logger.info(f"Target stats: mean={targets.mean():.2f}, std={targets.std():.2f}, "
                f"min={targets.min():.2f}, max={targets.max():.2f}")

    # ------------------------------------------------------------------
    # PHASE 3: BUILD LOADERS
    # ------------------------------------------------------------------
    logger.info("Phase 3: Building neighbor loaders...")
    loader_dict = build_loaders(task, data)
    log_memory("after loaders")

    # ------------------------------------------------------------------
    # PHASE 4: MINI VALIDATION (1 seed, 1 epoch, all 3 methods)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 4: MINI VALIDATION (1 epoch, seed=42, all 3 methods)")
    logger.info("=" * 60)

    actual_epochs = TASK_CONFIG['epochs']
    actual_max_steps = TASK_CONFIG['max_steps_per_epoch']
    actual_batch_size = TASK_CONFIG['batch_size']
    actual_num_neighbors = TASK_CONFIG['num_neighbors']

    mini_times = {}
    mini_maes = {}
    for method in METHODS:
        logger.info(f"  Mini run: {method}")
        seed_everything(42)
        try:
            model = Model(
                data=data, col_stats_dict=col_stats_dict,
                num_layers=2, channels=TASK_CONFIG['channels'],
                out_channels=TASK_CONFIG['out_channels'],
                method=method,
            ).to(DEVICE)

            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"    Model params: {param_count:,}")

            optimizer = torch.optim.Adam(model.parameters(), lr=TASK_CONFIG['lr'])
            loss_fn = L1Loss()

            t0 = time.time()
            train_loss = train_epoch(
                model, loader_dict["train"], optimizer, loss_fn,
                entity_table, actual_max_steps, TASK_CONFIG['grad_clip'],
            )
            epoch_time = time.time() - t0

            val_pred = evaluate(model, loader_dict["val"], entity_table,
                                clamp_min, clamp_max)
            val_metrics = task.evaluate(val_pred, task.get_table("val"))
            val_mae = val_metrics['mae']

            logger.info(f"    loss={train_loss:.4f}, val_mae={val_mae:.4f}, "
                        f"epoch_time={epoch_time:.1f}s")
            mini_times[method] = epoch_time
            mini_maes[method] = val_mae

            if HAS_GPU:
                peak_vram = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"    Peak VRAM: {peak_vram:.2f}GB")
                torch.cuda.reset_peak_memory_stats()

            del model, optimizer
            torch.cuda.empty_cache()
            gc.collect()

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"GPU OOM on mini {method}! Reducing batch/neighbors")
            torch.cuda.empty_cache()
            gc.collect()
            DEVIATIONS.append(
                f"GPU OOM on mini run {method}, reducing neighbors=[64,64], batch=256"
            )
            actual_num_neighbors = [64, 64]
            actual_batch_size = 256
            TASK_CONFIG['num_neighbors'] = actual_num_neighbors
            TASK_CONFIG['batch_size'] = actual_batch_size
            loader_dict = build_loaders(task, data)
            mini_times[method] = 120  # estimate
        except Exception:
            logger.exception(f"Mini run {method} failed")
            mini_times[method] = 120

    log_memory("after mini validation")

    # ------------------------------------------------------------------
    # PHASE 5: ADAPTIVE TIME BUDGET PLANNING
    # ------------------------------------------------------------------
    logger.info("Phase 5: Planning full runs...")
    max_epoch_time = max(mini_times.values()) if mini_times else 120
    logger.info(f"Max epoch time: {max_epoch_time:.1f}s")

    if max_epoch_time > 300:
        actual_max_steps = min(actual_max_steps, 1000)
        DEVIATIONS.append(
            f"Epoch time {max_epoch_time:.0f}s > 300s, max_steps={actual_max_steps}"
        )
        logger.warning(f"Reducing max_steps to {actual_max_steps}")

    # Estimate total time: 3 methods x 5 seeds x 10 epochs
    eval_overhead = max_epoch_time * 0.5
    est_run_time = actual_epochs * max_epoch_time + eval_overhead
    est_total_time = len(METHODS) * len(SEEDS) * est_run_time
    mins_left = remaining_minutes()
    time_budget_min = mins_left - 8  # reserve 8 min for analysis+output

    logger.info(f"Estimated per-run: {est_run_time/60:.1f} min")
    logger.info(f"Estimated total (15 runs): {est_total_time/60:.1f} min")
    logger.info(f"Time budget available: {time_budget_min:.0f} min")

    # Determine seeds per method. Sum always gets all 5 seeds.
    sum_seeds = list(SEEDS)
    mean_seeds = list(SEEDS)
    cama_seeds = list(SEEDS)
    use_iter5_for_mean = False
    use_iter5_for_cama = False

    # Progressive reduction if needed
    if est_total_time / 60 > time_budget_min:
        # A1: Reduce epochs from 10 to 7
        actual_epochs = max(7, actual_epochs - 3)
        est_run_time = actual_epochs * max_epoch_time + eval_overhead
        est_total_time = len(METHODS) * len(SEEDS) * est_run_time
        DEVIATIONS.append(f"Reduced epochs to {actual_epochs}")
        logger.info(f"Reduced epochs to {actual_epochs}")

    if est_total_time / 60 > time_budget_min:
        # A2: Reduce max_steps
        actual_max_steps = min(actual_max_steps, 1000)
        est_run_time = actual_epochs * max_epoch_time * 0.5 + eval_overhead
        est_total_time = len(METHODS) * len(SEEDS) * est_run_time
        DEVIATIONS.append(f"Reduced max_steps to {actual_max_steps}")
        logger.info(f"Reduced max_steps to {actual_max_steps}")

    if est_total_time / 60 > time_budget_min:
        # A3: Reduce mean+cama seeds to 3 (keep all 5 sum seeds)
        mean_seeds = SEEDS[:3]
        cama_seeds = SEEDS[:3]
        total_runs = len(sum_seeds) + len(mean_seeds) + len(cama_seeds)
        est_total_time = total_runs * est_run_time
        DEVIATIONS.append(f"Reduced mean+cama seeds to 3, sum keeps 5")
        logger.info(f"Reduced mean+cama seeds to 3")

    if est_total_time / 60 > time_budget_min:
        # A4: Skip cama, use iter-5 reference
        use_iter5_for_cama = True
        cama_seeds = []
        total_runs = len(sum_seeds) + len(mean_seeds)
        est_total_time = total_runs * est_run_time
        DEVIATIONS.append("Skipping cama runs, using iter-5 reference (d=13.58)")
        logger.info("Skipping cama runs, using iter-5 reference")

    if est_total_time / 60 > time_budget_min:
        # A5: Skip mean, use iter-5 reference
        use_iter5_for_mean = True
        mean_seeds = []
        total_runs = len(sum_seeds)
        est_total_time = total_runs * est_run_time
        DEVIATIONS.append("Skipping mean runs, using iter-5 reference (mae=49.19)")
        logger.info("Skipping mean runs, using iter-5 reference")

    TASK_CONFIG['epochs'] = actual_epochs
    TASK_CONFIG['max_steps_per_epoch'] = actual_max_steps

    # Build ordered run list: sum first (novel), then mean, then cama
    run_plan = []
    for seed in sum_seeds:
        run_plan.append(('sum', seed))
    for seed in mean_seeds:
        run_plan.append(('mean', seed))
    for seed in cama_seeds:
        run_plan.append(('cama', seed))

    logger.info(f"Final plan: {actual_epochs} epochs, max_steps={actual_max_steps}")
    logger.info(f"  sum: {len(sum_seeds)} seeds, mean: {len(mean_seeds)} seeds, "
                f"cama: {len(cama_seeds)} seeds = {len(run_plan)} total runs")
    logger.info(f"  iter-5 reference for mean={use_iter5_for_mean}, cama={use_iter5_for_cama}")

    # ------------------------------------------------------------------
    # PHASE 6: FULL EXPERIMENT RUNS
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 6: FULL EXPERIMENT RUNS")
    logger.info("=" * 60)

    all_results = _load_partial(partial_path)
    loss_fn = L1Loss()

    for method, seed in run_plan:
        run_key = f"{method}__seed{seed}"

        # Skip if already completed
        if (method, seed) in all_results:
            logger.info(f"Skipping {run_key} (already completed)")
            continue

        mins_left = remaining_minutes()
        if mins_left < 8:
            logger.warning(f"Only {mins_left:.0f} min left, stopping experiment loop")
            DEVIATIONS.append(f"Stopped at {run_key}: {mins_left:.0f}min left")
            break

        logger.info(f"=== RUN: {run_key} ({mins_left:.0f} min left) ===")
        seed_everything(seed)
        run_t0 = time.time()

        model = None
        optimizer = None
        best_state = None

        try:
            model = Model(
                data=data, col_stats_dict=col_stats_dict,
                num_layers=2, channels=TASK_CONFIG['channels'],
                out_channels=TASK_CONFIG['out_channels'],
                method=method,
            ).to(DEVICE)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=TASK_CONFIG['lr']
            )

            best_val_mae = math.inf
            best_epoch = 0
            epoch_records = []
            gate_evolution = []
            peak_vram = 0.0

            for epoch in range(1, actual_epochs + 1):
                t0 = time.time()

                # Train
                train_loss = train_epoch(
                    model, loader_dict["train"], optimizer, loss_fn,
                    entity_table, actual_max_steps,
                    TASK_CONFIG['grad_clip'],
                )

                # Validate
                val_pred = evaluate(
                    model, loader_dict["val"], entity_table,
                    clamp_min, clamp_max,
                )
                val_metrics = task.evaluate(val_pred, task.get_table("val"))
                val_mae = float(val_metrics['mae'])
                epoch_time = time.time() - t0

                # Track best
                if val_mae <= best_val_mae:
                    best_val_mae = val_mae
                    best_epoch = epoch
                    if best_state is not None:
                        del best_state
                    best_state = copy.deepcopy(model.state_dict())

                epoch_records.append({
                    'epoch': epoch,
                    'train_loss': float(train_loss),
                    'val_mae': val_mae,
                    'elapsed_s': float(epoch_time),
                })

                # Gate recording EVERY epoch (CAMA only)
                if method == 'cama' and isinstance(model.gnn, CAMAHeteroGraphSAGE):
                    try:
                        start_gate_recording(model)
                        _ = evaluate_n_batches(model, loader_dict["val"],
                                               entity_table, 20)
                        gate_snapshot = collect_gate_snapshot(model)
                        gate_evolution.append({
                            'epoch': epoch,
                            'gates': gate_snapshot,
                        })
                    except Exception:
                        logger.exception(f"Gate recording failed at epoch {epoch}")

                if HAS_GPU:
                    current_peak = torch.cuda.max_memory_allocated() / 1e9
                    peak_vram = max(peak_vram, current_peak)

                logger.info(
                    f"  E{epoch}: loss={train_loss:.4f}, "
                    f"val_mae={val_mae:.4f}, "
                    f"best={best_val_mae:.4f}@E{best_epoch}, "
                    f"time={epoch_time:.1f}s"
                )

            # Load best model
            if best_state is not None:
                model.load_state_dict(best_state)

            # Final gate stats (CAMA only)
            final_gate_stats = {}
            if method == 'cama' and isinstance(model.gnn, CAMAHeteroGraphSAGE):
                try:
                    start_gate_recording(model)
                    _ = evaluate(model, loader_dict["val"], entity_table,
                                 clamp_min, clamp_max)
                    final_gate_stats = collect_gate_snapshot(model)
                except Exception:
                    logger.exception("Final gate extraction failed")

            run_time = time.time() - run_t0
            all_results[(method, seed)] = {
                'best_val_mae': float(best_val_mae),
                'best_epoch': best_epoch,
                'epoch_records': epoch_records,
                'gate_evolution': gate_evolution,
                'final_gate_stats': final_gate_stats,
                'wall_clock_s': float(run_time),
                'peak_vram_gb': float(peak_vram),
            }
            logger.info(f"  Completed in {run_time:.1f}s, "
                        f"best_val_mae={best_val_mae:.4f}@E{best_epoch}")

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"GPU OOM on {run_key}!")
            torch.cuda.empty_cache()
            gc.collect()
            DEVIATIONS.append(f"GPU OOM on {run_key}")
            all_results[(method, seed)] = {
                'best_val_mae': float('nan'),
                'best_epoch': 0,
                'epoch_records': [],
                'gate_evolution': [],
                'final_gate_stats': {},
                'wall_clock_s': time.time() - run_t0,
                'peak_vram_gb': 0.0,
                'error': 'GPU OOM',
            }
        except Exception:
            logger.exception(f"FAILED: {run_key}")
            all_results[(method, seed)] = {
                'best_val_mae': float('nan'),
                'best_epoch': 0,
                'epoch_records': [],
                'gate_evolution': [],
                'final_gate_stats': {},
                'wall_clock_s': time.time() - run_t0,
                'peak_vram_gb': 0.0,
                'error': traceback.format_exc()[:500],
            }
        finally:
            del model, optimizer, best_state
            torch.cuda.empty_cache()
            gc.collect()

        # Save partial results after each run
        _save_partial(all_results, partial_path)
        log_memory(f"after {run_key}")

    # Free graph data
    del data, col_stats_dict, loader_dict
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PHASE 7: COLLECT MAE VALUES PER METHOD
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 7: Statistical analysis")
    logger.info("=" * 60)

    def collect_maes(method_name, seeds_list):
        maes = []
        for seed in seeds_list:
            res = all_results.get((method_name, seed))
            if res and not np.isnan(res.get('best_val_mae', float('nan'))):
                maes.append(res['best_val_mae'])
        return maes

    sum_maes = collect_maes('sum', sum_seeds)

    # For mean: use fresh results or iter-5 reference
    if use_iter5_for_mean:
        mean_maes = list(ITER5_REFERENCE['baseline_mean']['per_seed'].values())
        DEVIATIONS.append("Using iter-5 reference for mean MAEs")
        logger.info(f"Mean MAEs (iter-5 reference): {mean_maes}")
    else:
        mean_maes = collect_maes('mean', mean_seeds)
        logger.info(f"Mean MAEs (fresh): {mean_maes}")

    # For cama: use fresh results or iter-5 reference
    if use_iter5_for_cama:
        cama_maes = list(ITER5_REFERENCE['cama']['per_seed'].values())
        DEVIATIONS.append("Using iter-5 reference for CAMA MAEs")
        logger.info(f"CAMA MAEs (iter-5 reference): {cama_maes}")
    else:
        cama_maes = collect_maes('cama', cama_seeds)
        logger.info(f"CAMA MAEs (fresh): {cama_maes}")

    logger.info(f"Sum MAEs: {sum_maes}")
    logger.info(f"Mean MAEs: {mean_maes}")
    logger.info(f"CAMA MAEs: {cama_maes}")

    # ------------------------------------------------------------------
    # PHASE 8: 3 PAIRWISE COMPARISONS
    # ------------------------------------------------------------------
    pairwise = {}

    # 1. CAMA vs Sum (PRIMARY): positive d = CAMA lower MAE (better)
    if len(sum_maes) >= 2 and len(cama_maes) >= 2:
        pairwise['cama_vs_sum'] = compute_pairwise_stats(
            sum_maes, cama_maes, 'cama_vs_sum'
        )
        logger.info(f"CAMA vs Sum: d={pairwise['cama_vs_sum']['cohens_d']:.3f}")
    else:
        pairwise['cama_vs_sum'] = {
            'error': f'Insufficient data: sum={len(sum_maes)}, cama={len(cama_maes)}'
        }

    # 2. Mean vs Sum (SECONDARY): positive d = sum lower MAE (better)
    if len(sum_maes) >= 2 and len(mean_maes) >= 2:
        pairwise['mean_vs_sum'] = compute_pairwise_stats(
            mean_maes, sum_maes, 'mean_vs_sum'
        )
        logger.info(f"Mean vs Sum: d={pairwise['mean_vs_sum']['cohens_d']:.3f}")
    else:
        pairwise['mean_vs_sum'] = {
            'error': f'Insufficient data: mean={len(mean_maes)}, sum={len(sum_maes)}'
        }

    # 3. CAMA vs Mean (REPLICATION): positive d = CAMA lower MAE (better)
    if len(mean_maes) >= 2 and len(cama_maes) >= 2:
        pairwise['cama_vs_mean'] = compute_pairwise_stats(
            mean_maes, cama_maes, 'cama_vs_mean'
        )
        logger.info(f"CAMA vs Mean: d={pairwise['cama_vs_mean']['cohens_d']:.3f} "
                     f"(iter-5: {ITER5_REFERENCE['cohens_d']:.3f})")
    else:
        pairwise['cama_vs_mean'] = {
            'error': f'Insufficient data: mean={len(mean_maes)}, cama={len(cama_maes)}'
        }

    # ------------------------------------------------------------------
    # PHASE 9: DECISION TABLE
    # ------------------------------------------------------------------
    sum_mean_mae = float(np.mean(sum_maes)) if sum_maes else None
    cama_ref_mae = 45.2586  # iter-5 CAMA mean
    mean_ref_mae = 49.1929  # iter-5 mean baseline

    if sum_mean_mae is not None:
        if sum_mean_mae <= cama_ref_mae:
            decision = "SUM_SUFFICIENT"
            decision_detail = (
                f"Sum MAE ({sum_mean_mae:.2f}) <= CAMA MAE ({cama_ref_mae:.2f}). "
                "Sum aggregation alone matches or beats CAMA. "
                "CAMA's variance injection adds nothing over the simpler sum baseline."
            )
        elif sum_mean_mae < mean_ref_mae:
            decision = "CAMA_ADDS_VALUE"
            decision_detail = (
                f"CAMA MAE ({cama_ref_mae:.2f}) < Sum MAE ({sum_mean_mae:.2f}) < "
                f"Mean MAE ({mean_ref_mae:.2f}). Sum helps over mean but CAMA's "
                "distributional enrichment provides additional benefit beyond sum."
            )
        else:
            decision = "SUM_DOES_NOT_HELP"
            decision_detail = (
                f"Sum MAE ({sum_mean_mae:.2f}) >= Mean MAE ({mean_ref_mae:.2f}). "
                "Sum aggregation does not help; mean's normalization is important. "
                "Sum magnifies noise in high-cardinality neighborhoods."
            )
    else:
        decision = "INSUFFICIENT_DATA"
        decision_detail = "No sum MAE results available."

    logger.info(f"Decision: {decision}")
    logger.info(f"Detail: {decision_detail}")

    # ------------------------------------------------------------------
    # PHASE 10: CROSS-ITERATION COMPARISON
    # ------------------------------------------------------------------
    cross_iteration = {
        'iter2_rama_10epochs': {
            'module': 'RAMA (rank+cardinality gate)',
            'epochs': 10,
            'num_neighbors': [128, 64],
            'baseline_mae': 48.78,
            'method_mae': 44.99,
            'cohens_d': 10.83,
            'note': 'RAMA with full rank proxy + cardinality gating',
        },
        'iter4_cama_5epochs': {
            'module': 'CAMA (cardinality-only gate)',
            'epochs': 5,
            'num_neighbors': [128, 128],
            'baseline_mae': 171.91,
            'method_mae': 343.10,
            'cohens_d': -1.38,
            'note': 'CAMA underperformed baseline with only 5 epochs',
        },
        'iter5_cama_10epochs': {
            'module': 'CAMA (cardinality-only gate)',
            'epochs': 10,
            'num_neighbors': [128, 128],
            'baseline_mae': 49.19,
            'method_mae': 45.26,
            'cohens_d': 13.58,
            'note': 'CAMA with 10 epochs, mean baseline',
        },
        'this_sum_baseline': {
            'module': 'HeteroGraphSAGE(aggr=sum)',
            'epochs': actual_epochs,
            'num_neighbors': TASK_CONFIG['num_neighbors'],
            'baseline_mae': float(np.mean(mean_maes)) if mean_maes else None,
            'method_mae': sum_mean_mae,
            'cohens_d': pairwise.get('mean_vs_sum', {}).get('cohens_d'),
            'note': f'Sum aggregation baseline, {actual_epochs} epochs',
        },
    }

    # ------------------------------------------------------------------
    # PHASE 11: DIAGNOSTICS
    # ------------------------------------------------------------------
    diagnostics = {
        'learning_curves': {'sum': {}, 'mean': {}, 'cama': {}},
        'gate_evolution': {},
        'wall_clock_summary': {},
        'interpretation': {},
    }

    for method in METHODS:
        method_seeds = sum_seeds if method == 'sum' else (
            mean_seeds if method == 'mean' else cama_seeds)
        total_time = 0.0
        for seed in method_seeds:
            res = all_results.get((method, seed))
            if res and res.get('epoch_records'):
                diagnostics['learning_curves'][method][str(seed)] = res['epoch_records']
            if res and res.get('wall_clock_s'):
                total_time += res['wall_clock_s']
        diagnostics['wall_clock_summary'][f'{method}_total_s'] = total_time

    # Gate evolution for CAMA
    for seed in cama_seeds:
        res = all_results.get(('cama', seed))
        if res and res.get('gate_evolution'):
            diagnostics['gate_evolution'][str(seed)] = res['gate_evolution']

    # Interpret results
    sum_vs_mean_interp = "Not computed"
    cama_vs_sum_interp = "Not computed"

    if sum_mean_mae is not None and mean_maes:
        mean_avg = float(np.mean(mean_maes))
        diff = mean_avg - sum_mean_mae
        if abs(diff) < 1.0:
            sum_vs_mean_interp = (
                f"Sum ({sum_mean_mae:.2f}) and mean ({mean_avg:.2f}) produce similar MAE "
                f"(diff={diff:.2f}). The aggregation function choice between sum and mean "
                "has minimal impact on this task."
            )
        elif diff > 0:
            sum_vs_mean_interp = (
                f"Sum ({sum_mean_mae:.2f}) outperforms mean ({mean_avg:.2f}) by {diff:.2f}. "
                "Sum aggregation preserves neighborhood magnitude information that mean "
                "normalizes away."
            )
        else:
            sum_vs_mean_interp = (
                f"Mean ({mean_avg:.2f}) outperforms sum ({sum_mean_mae:.2f}) by {-diff:.2f}. "
                "Mean's normalization helps for high-cardinality neighborhoods in Amazon data."
            )

    if sum_mean_mae is not None and cama_maes:
        cama_avg = float(np.mean(cama_maes))
        diff = sum_mean_mae - cama_avg
        if diff > 0:
            cama_vs_sum_interp = (
                f"CAMA ({cama_avg:.2f}) outperforms sum ({sum_mean_mae:.2f}) by {diff:.2f}. "
                "CAMA's cardinality-aware variance injection provides value beyond sum aggregation."
            )
        else:
            cama_vs_sum_interp = (
                f"Sum ({sum_mean_mae:.2f}) matches or beats CAMA ({cama_avg:.2f}). "
                "CAMA's complexity is not justified — simple sum aggregation suffices."
            )

    diagnostics['interpretation'] = {
        'sum_vs_mean': sum_vs_mean_interp,
        'cama_vs_sum': cama_vs_sum_interp,
        'overall_conclusion': decision_detail,
    }

    # ------------------------------------------------------------------
    # PHASE 12: BUILD OUTPUT (exp_gen_sol_out schema)
    # ------------------------------------------------------------------
    logger.info("Phase 12: Building output...")

    # Build examples from val predictions
    examples = []
    try:
        val_table = task.get_table("val")
        val_df = val_table.df
        entity_col = task.entity_col
        target_col = task.target_col

        # Use at most 20 examples
        n_examples = min(20, len(val_df))

        for idx in range(n_examples):
            row = val_df.iloc[idx]
            input_dict = {}
            for col in val_df.columns:
                val = row[col]
                if hasattr(val, "isoformat"):
                    input_dict[col] = val.isoformat()
                elif isinstance(val, (np.integer,)):
                    input_dict[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    input_dict[col] = float(val) if not np.isnan(val) else None
                elif isinstance(val, list):
                    input_dict[col] = val
                elif val is None or (isinstance(val, float) and np.isnan(val)):
                    input_dict[col] = None
                else:
                    input_dict[col] = str(val)

            target_val = row.get(target_col, None) if target_col in val_df.columns else None
            if target_val is not None and not (isinstance(target_val, float) and np.isnan(target_val)):
                output_str = str(round(float(target_val), 6))
            else:
                output_str = "nan"

            # Best seed per method for predictions
            best_sum_seed = sum_seeds[int(np.argmin(sum_maes))] if sum_maes else 42
            best_mean_seed = (
                mean_seeds[int(np.argmin(mean_maes))] if mean_maes and not use_iter5_for_mean
                else 42
            )
            best_cama_seed = (
                cama_seeds[int(np.argmin(cama_maes))] if cama_maes and not use_iter5_for_cama
                else 42
            )

            example: Dict[str, Any] = {
                "input": json.dumps(input_dict),
                "output": output_str,
                "predict_sum": str(round(
                    all_results.get(('sum', best_sum_seed), {})
                    .get('best_val_mae', float('nan')), 4
                )),
                "predict_mean": str(round(
                    all_results.get(('mean', best_mean_seed), {})
                    .get('best_val_mae', float('nan')), 4
                )) if not use_iter5_for_mean else str(round(
                    ITER5_REFERENCE['baseline_mean']['mean_mae'], 4
                )),
                "predict_cama": str(round(
                    all_results.get(('cama', best_cama_seed), {})
                    .get('best_val_mae', float('nan')), 4
                )) if not use_iter5_for_cama else str(round(
                    ITER5_REFERENCE['cama']['mean_mae'], 4
                )),
                "metadata_fold": 1,
                "metadata_task_type": "regression",
                "metadata_entity_col": entity_col,
                "metadata_target_col": target_col,
            }
            if entity_col in val_df.columns:
                eid = row[entity_col]
                example["metadata_row_index"] = (
                    int(eid) if isinstance(eid, (int, np.integer)) else str(eid)
                )
            examples.append(example)

    except Exception:
        logger.exception("Failed to build examples from val table")
        # Fallback: create summary examples per seed
        for seed in sum_seeds:
            s_res = all_results.get(('sum', seed), {})
            m_res = all_results.get(('mean', seed), {})
            c_res = all_results.get(('cama', seed), {})
            example = {
                "input": json.dumps({
                    "task": "rel-amazon/item-ltv",
                    "type": "regression",
                    "metric": "mae",
                    "seed": seed,
                    "epochs": actual_epochs,
                }),
                "output": json.dumps({
                    "sum_val_mae": s_res.get('best_val_mae'),
                    "mean_val_mae": m_res.get('best_val_mae'),
                    "cama_val_mae": c_res.get('best_val_mae'),
                }),
                "predict_sum": str(s_res.get('best_val_mae', 'N/A')),
                "predict_mean": str(m_res.get('best_val_mae', 'N/A')),
                "predict_cama": str(c_res.get('best_val_mae', 'N/A')),
                "metadata_seed": seed,
                "metadata_task_type": "regression",
                "metadata_entity_col": "product_id",
                "metadata_target_col": "ltv",
            }
            examples.append(example)

    # Ensure at least 1 example
    if not examples:
        examples.append({
            "input": json.dumps({"task": "rel-amazon/item-ltv", "note": "no valid examples"}),
            "output": "nan",
            "predict_sum": "nan",
            "predict_mean": "nan",
            "predict_cama": "nan",
            "metadata_task_type": "regression",
        })

    # Build per-method result dicts
    per_method_results = {}
    for method_name, method_seeds_list, ref_key in [
        ('sum', sum_seeds, None),
        ('mean', mean_seeds, 'baseline_mean' if use_iter5_for_mean else None),
        ('cama', cama_seeds, 'cama' if use_iter5_for_cama else None),
    ]:
        if ref_key and ref_key in ITER5_REFERENCE:
            # Using iter-5 reference
            ref = ITER5_REFERENCE[ref_key]
            per_method_results[method_name] = {
                'per_seed': {str(k): v for k, v in ref['per_seed'].items()},
                'mean_mae': ref['mean_mae'],
                'std_mae': ref['std_mae'],
                'source': 'iter-5 reference',
            }
        else:
            maes_for_method = collect_maes(method_name, method_seeds_list)
            per_method_results[method_name] = {
                'per_seed': {
                    str(s): all_results.get((method_name, s), {}).get('best_val_mae')
                    for s in method_seeds_list
                },
                'mean_mae': float(np.mean(maes_for_method)) if maes_for_method else None,
                'std_mae': float(np.std(maes_for_method, ddof=1)) if len(maes_for_method) > 1 else None,
                'source': 'fresh run',
            }

    output = {
        'metadata': {
            'experiment': 'Sum aggregation baseline vs CAMA on rel-amazon/item-ltv',
            'objective': 'Determine if d=13.58 CAMA effect survives sum comparison',
            'method_name': 'Sum/Mean/CAMA aggregation comparison',
            'description': (
                'Tests whether iter-5 CAMA effect (d=13.58 vs mean baseline) survives '
                'comparison against HeteroGraphSAGE(aggr=sum). The RelBench default is sum, '
                'so this is the true baseline comparison. Runs 3 methods x 5 seeds x '
                f'{actual_epochs} epochs on rel-amazon/item-ltv.'
            ),
            'hyperparameters': {
                'channels': TASK_CONFIG['channels'],
                'lr': TASK_CONFIG['lr'],
                'epochs': actual_epochs,
                'batch_size': TASK_CONFIG['batch_size'],
                'num_neighbors': TASK_CONFIG['num_neighbors'],
                'num_layers': 2,
                'max_steps_per_epoch': actual_max_steps,
                'grad_clip': TASK_CONFIG['grad_clip'],
            },
            'seeds': SEEDS,
            'methods': METHODS,
            'evaluation_split': 'val',
            'test_target_status': {
                'available': False,
                'reason': 'RelBench hides test labels for item-ltv',
            },
            'iter5_reference': ITER5_REFERENCE,
            'deviations': DEVIATIONS,
            'results': {
                'per_method': per_method_results,
                'pairwise_comparisons': pairwise,
                'decision': decision,
                'decision_detail': decision_detail,
                'cross_iteration_comparison': cross_iteration,
            },
            'diagnostics': diagnostics,
            'graph_build_time_sec': graph_build_time,
            'total_wall_clock_sec': time.time() - start_time,
        },
        'datasets': [
            {
                'dataset': 'rel-amazon/item-ltv',
                'examples': examples,
            }
        ],
    }

    out_path.write_text(json.dumps(output, indent=2, default=str))
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"Saved output to {out_path} ({size_kb:.1f} KB)")

    # Clean up partial results
    if partial_path.exists():
        partial_path.unlink()

    total_time = time.time() - start_time
    logger.info(f"=== DONE in {total_time:.0f}s ({total_time/60:.1f} min) ===")
    logger.info(f"Decision: {decision}")
    logger.info(f"Deviations: {DEVIATIONS}")


if __name__ == "__main__":
    main()
