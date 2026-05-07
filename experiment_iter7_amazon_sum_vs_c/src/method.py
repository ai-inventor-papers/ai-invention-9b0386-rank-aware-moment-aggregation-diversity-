#!/usr/bin/env python3
"""Amazon Sum-vs-CAMA Fix + Trial Classification: Custom SumAggregation Bypassing pyg-lib Fused Path.

Two-part GPU experiment resolving iter-6's critical evidence gaps:
(A) Amazon item-ltv sum vs mean vs CAMA comparison (failed in iter-6 due to
    pyg-lib/spmm), fixed via a custom SumAggregation subclass that bypasses
    the fused message_and_aggregate path.
(B) rel-trial/study-outcome classification with sum baseline (skipped in iter-6).

30 total runs (3 methods x 5 seeds x 2 tasks) within 80-min GPU budget.
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
# MONKEY-PATCH: Disable pyg-lib C++ sampler to use Python fallback.
# pyg-lib's C++ hetero_neighbor_sample doesn't properly support temporal
# disjoint sampling. Force the pure-Python/torch-scatter fallback which
# handles temporal + disjoint correctly.
# ============================================================================
try:
    import torch_geometric.sampler.neighbor_sampler as _ns_module
    if hasattr(_ns_module, 'pyg_lib'):
        _ns_module.pyg_lib = None
        logger.info("Disabled pyg-lib C++ sampler, using Python fallback")
    import torch_geometric.sampler.utils as _sampler_utils
    if hasattr(_sampler_utils, 'pyg_lib'):
        _sampler_utils.pyg_lib = None
    if hasattr(_ns_module, 'torch_sparse'):
        _ns_module.torch_sparse = None
        logger.info("Also disabled torch-sparse sampler")
except Exception as e:
    logger.warning(f"Could not disable pyg-lib sampler: {e}")
    try:
        from torch_geometric.sampler.neighbor_sampler import NeighborSampler as _NS
        _NS.disjoint = property(
            lambda self: self._disjoint,
            lambda self, v: setattr(self, '_disjoint', v)
        )
        logger.info("Applied NeighborSampler disjoint monkey-patch as fallback")
    except Exception as e2:
        logger.warning(f"Could not apply fallback patch: {e2}")

# ============================================================================
# DELAYED IMPORTS (after hardware setup and monkey-patch)
# ============================================================================
from torch.nn import BCEWithLogitsLoss, L1Loss
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
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from sentence_transformers import SentenceTransformer
from scipy import stats as sp_stats

# ============================================================================
# CONFIGURATION
# ============================================================================
SEEDS = [42, 123, 456, 789, 1024]
RELBENCH_CACHE_DIR = (
    "/ai-inventor/aii_pipeline/runs/temp-debug-test_sbr/"
    "3_invention_loop/iter_1/gen_art/data_id3_it1__opus/"
    "temp/relbench_cache"
)
os.environ["RELBENCH_CACHE_DIR"] = RELBENCH_CACHE_DIR

METHOD_NAMES = ['sum', 'mean', 'cama']
TIME_BUDGET_SEC = 80 * 60  # 80 minutes total
DEVIATIONS: List[str] = []

PART_A_CONFIG = {
    'dataset': 'rel-amazon',
    'task': 'item-ltv',
    'type': 'regression',
    'metric': 'mae',
    'higher_is_better': False,
    'out_channels': 1,
    'channels': 128,
    'lr': 0.005,
    'epochs': 10,
    'batch_size': 512,
    'num_neighbors': [128, 128],
    'num_layers': 2,
    'max_steps_per_epoch': 2000,
    'grad_clip': 1.0,
    'exclude_cols': {
        'review': ['review_text', 'summary'],
        'customer': ['customer_name'],
    },
}

PART_B_CONFIG = {
    'dataset': 'rel-trial',
    'task': 'study-outcome',
    'type': 'binary_classification',
    'metric': 'average_precision',
    'higher_is_better': True,
    'out_channels': 1,
    'channels': 128,
    'lr': 0.005,
    'epochs': 10,
    'batch_size': 512,
    'num_neighbors': [128, 128],
    'num_layers': 2,
    'max_steps_per_epoch': 2000,
    'grad_clip': 1.0,
    'exclude_cols': {},
}

# Iter-5 reference values for cross-iteration validation (Amazon item-ltv MAE)
ITER5_MEAN_MAES = [49.2625, 49.5791, 49.1939, 48.6403, 49.2884]
ITER5_CAMA_MAES = [45.0793, 45.1592, 45.0520, 45.5217, 45.4809]


# ============================================================================
# STEP 4: CUSTOM SAFE SUM AGGREGATION - THE CRITICAL FIX
# ============================================================================
class SafeSumAggregation(Aggregation):
    """Sum aggregation as a custom Aggregation subclass.

    CRITICAL: By being an Aggregation *object* (not a string), when passed
    to SAGEConv(aggr=SafeSumAggregation()), PyG's MessagePassing checks
    isinstance(self.aggr, str) -> False -> self.fuse = False -> bypasses
    message_and_aggregate() which calls spmm() requiring pyg-lib.
    Instead uses the safe: message() -> aggregate() -> aggr_module.forward() path.
    """

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')


class PureTorchSumAggregation(Aggregation):
    """Fallback sum aggregation using pure PyTorch (no torch_scatter needed)."""

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        if dim_size is None:
            dim_size = int(index.max()) + 1 if index is not None and index.numel() > 0 else 0
        out = torch.zeros(dim_size, x.size(-1), device=x.device, dtype=x.dtype)
        if index is not None:
            return out.index_add_(0, index, x)
        return out


# ============================================================================
# STEP 5: CAMA AGGREGATION
# ============================================================================
class CAMAAggregation(Aggregation):
    """Cardinality-Aware Moment Aggregation.

    Enriches mean with per-dimension variance, gated by log-cardinality.
    Simplified RAMA: gate depends ONLY on log(cardinality), no rank proxy.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.gate_net = nn.Linear(1, channels, bias=True)
        self.var_transform = nn.Linear(channels, channels, bias=False)
        self._gate_buffer: list = []
        self._recording = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gate_net.weight)
        nn.init.zeros_(self.gate_net.bias)
        nn.init.eye_(self.var_transform.weight)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        mean_of_sq = self.reduce(x * x, index, ptr, dim_size, dim, reduce='mean')
        var = (mean_of_sq - mean * mean).clamp(min=1e-8)

        ones = x.new_ones(x.size(0), 1)
        cardinality = self.reduce(ones, index, ptr, dim_size, dim, reduce='sum')
        log_card = torch.log1p(cardinality)

        gate = torch.sigmoid(self.gate_net(log_card))

        if self._recording:
            self._gate_buffer.append(gate.detach().cpu())

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
            return {
                'mean': float(all_gates.mean()),
                'std': float(all_gates.std()),
                'min': float(all_gates.min()),
                'max': float(all_gates.max()),
                'median': float(all_gates.median()),
            }
        return None


# ============================================================================
# STEP 6: CUSTOM HeteroGraphSAGE
# ============================================================================
class CustomHeteroGraphSAGE(nn.Module):
    """HeteroGraphSAGE that deep-copies non-string Aggregation objects
    for each SAGEConv, so each edge type gets its own instance."""

    def __init__(self, node_types, edge_types, channels, aggr, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.cama_registry: Dict[Tuple, CAMAAggregation] = {}

        for layer_idx in range(num_layers):
            edge_conv_dict = {}
            for et in edge_types:
                if isinstance(aggr, str):
                    conv_aggr = aggr
                else:
                    conv_aggr = copy.deepcopy(aggr)
                    if isinstance(conv_aggr, CAMAAggregation):
                        self.cama_registry[(layer_idx, et)] = conv_aggr
                edge_conv_dict[et] = SAGEConv(
                    (channels, channels), channels, aggr=conv_aggr
                )
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
# STEP 7: MODEL CLASS
# ============================================================================
class Model(nn.Module):
    def __init__(self, data, col_stats_dict, num_layers, channels,
                 out_channels, aggr):
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
        self.gnn = CustomHeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
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
# STEP 8: GLOVE TEXT EMBEDDING
# ============================================================================
class GloveTextEmbedding:
    """Uses sentence-transformers GloVe model for text embedding."""

    def __init__(self, device=None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device or torch.device("cpu"),
        )

    def __call__(self, sentences):
        clean = []
        for s in sentences:
            if s is None or (isinstance(s, float) and math.isnan(s)):
                clean.append("")
            else:
                clean.append(str(s))
        return torch.from_numpy(
            self.model.encode(clean, show_progress_bar=False)
        )


# ============================================================================
# STEP 9: TRAINING AND EVALUATION
# ============================================================================
def train_epoch_regression(model, loader, optimizer, entity_table,
                           max_steps, grad_clip=1.0):
    """Train one epoch of regression with L1Loss."""
    model.train()
    loss_fn = L1Loss()
    loss_accum = count_accum = 0
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch, entity_table).view(-1)
        target = batch[entity_table].y.float()
        loss = loss_fn(pred, target)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss at step {step}, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)
    return loss_accum / max(count_accum, 1)


def train_epoch_classification(model, loader, optimizer, entity_table,
                               max_steps, grad_clip=1.0):
    """Train one epoch of binary classification with BCEWithLogitsLoss."""
    model.train()
    loss_fn = BCEWithLogitsLoss()
    loss_accum = count_accum = 0
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch, entity_table).view(-1)
        target = batch[entity_table].y.float()
        loss = loss_fn(pred, target)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss at step {step}, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)
    return loss_accum / max(count_accum, 1)


@torch.no_grad()
def evaluate_regression(model, loader, entity_table, task_obj, split="val"):
    """Evaluate regression: return predictions and MAE via task.evaluate()."""
    model.eval()
    pred_list = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table).view(-1)
        pred_list.append(pred.detach().cpu())
    if not pred_list:
        return float('inf'), np.array([])
    all_preds = torch.cat(pred_list, dim=0).numpy()
    # Clamp predictions to reasonable range
    all_preds = np.clip(all_preds, 0, 500_000)
    table = task_obj.get_table(split)
    metrics = task_obj.evaluate(all_preds, table)
    mae = float(metrics.get('mae', float('inf')))
    return mae, all_preds


@torch.no_grad()
def evaluate_classification(model, loader, entity_table, task_obj, split="val"):
    """Evaluate classification: return predictions and AP via task.evaluate()."""
    model.eval()
    pred_list = []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch, entity_table).view(-1)
        probs = torch.sigmoid(logits)
        pred_list.append(probs.detach().cpu())
    if not pred_list:
        return 0.0, np.array([])
    all_preds = torch.cat(pred_list, dim=0).numpy()
    table = task_obj.get_table(split)
    metrics = task_obj.evaluate(all_preds, table)
    ap = float(metrics.get('average_precision', 0.0))
    return ap, all_preds


# ============================================================================
# STEP 10: STATISTICAL ANALYSIS
# ============================================================================
def cohens_d_independent(group1, group2):
    """Pooled-std Cohen's d. Positive = group2 has higher value."""
    g1, g2 = np.asarray(group1, dtype=float), np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled < 1e-12:
        return 0.0
    return float((m2 - m1) / s_pooled)


def bootstrap_cohens_d(group1, group2, n_boot=10000, alpha=0.05):
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.RandomState(42)
    g1, g2 = np.asarray(group1), np.asarray(group2)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return (0.0, 0.0)
    boot_ds = []
    for _ in range(n_boot):
        idx1 = rng.choice(n1, n1, replace=True)
        idx2 = rng.choice(n2, n2, replace=True)
        boot_ds.append(cohens_d_independent(g1[idx1], g2[idx2]))
    return (
        float(np.percentile(boot_ds, 100 * alpha / 2)),
        float(np.percentile(boot_ds, 100 * (1 - alpha / 2))),
    )


def compute_pairwise_stats(scores_a, scores_b, name_a, name_b, higher_is_better=True):
    """Full pairwise comparison: Cohen's d, t-test, Wilcoxon, bootstrap CI.
    For MAE (lower=better): positive d means B has lower (better) MAE.
    For AP (higher=better): positive d means B has higher (better) AP.
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if higher_is_better:
        d = cohens_d_independent(a, b)  # positive = B higher = B better
    else:
        d = cohens_d_independent(b, a)  # positive = B lower = B better (for MAE)

    ci_lo, ci_hi = bootstrap_cohens_d(
        b if not higher_is_better else a,
        a if not higher_is_better else b,
    )

    t_stat, p_val_t = None, None
    if len(a) == len(b) and len(a) >= 3:
        try:
            t_stat, p_val_t = sp_stats.ttest_rel(a, b)
            t_stat, p_val_t = float(t_stat), float(p_val_t)
        except Exception:
            pass

    p_val_w = None
    if len(a) == len(b) and len(a) >= 5:
        try:
            _, p_val_w = sp_stats.wilcoxon(a, b)
            p_val_w = float(p_val_w)
        except Exception:
            pass

    return {
        'comparison': f"{name_a}_vs_{name_b}",
        'cohens_d': d,
        'd_ci_95': [ci_lo, ci_hi],
        't_statistic': t_stat,
        'p_value_ttest': p_val_t,
        'p_value_wilcoxon': p_val_w,
        f'{name_a}_mean': float(np.mean(a)),
        f'{name_a}_std': float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        f'{name_b}_mean': float(np.mean(b)),
        f'{name_b}_std': float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
        'n_seeds': min(len(a), len(b)),
        'interpretation': (
            f"{name_b} {'better' if d > 0 else 'worse'} than {name_a} "
            f"by |d|={abs(d):.3f}"
        ),
    }


# ============================================================================
# GATE & MEMORY HELPERS
# ============================================================================
def format_edge_type(et) -> str:
    if isinstance(et, tuple):
        return "__".join(str(x) for x in et)
    return str(et)


def start_gate_recording(model):
    if hasattr(model.gnn, 'cama_registry'):
        for cama_mod in model.gnn.cama_registry.values():
            cama_mod.start_recording()


def collect_gate_snapshot(model) -> Dict[str, Dict]:
    gate_snapshot = {}
    if not hasattr(model.gnn, 'cama_registry'):
        return gate_snapshot
    for (layer_idx, edge_type), cama_mod in model.gnn.cama_registry.items():
        gs = cama_mod.stop_recording()
        if gs is not None:
            et_str = format_edge_type(edge_type)
            gate_snapshot[f"L{layer_idx}_{et_str}"] = gs
    return gate_snapshot


def log_memory(label: str = "") -> float:
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
# DATA LOADING
# ============================================================================
def load_task_data(dataset_name: str, task_name: str,
                   exclude_cols: Optional[Dict[str, List[str]]] = None):
    """Load RelBench dataset and build materialized hetero graph."""
    from torch_frame import stype as st

    logger.info(f"Loading {dataset_name}/{task_name}...")
    t0 = time.time()

    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)
    db = dataset.get_db()

    logger.info(f"Dataset loaded in {time.time()-t0:.0f}s")
    logger.info(f"Tables: {list(db.table_dict.keys())}")
    log_memory(f"after {dataset_name} load")

    # Build stype dict
    stypes_cache_path = Path(RELBENCH_CACHE_DIR) / dataset_name / "stypes.json"
    if stypes_cache_path.exists():
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in list(col_to_stype.items()):
                col_to_stype[col] = st(stype_str)
        logger.info(f"Loaded cached stypes for {dataset_name}")
    else:
        logger.info(f"Computing stype proposal for {dataset_name}...")
        col_to_stype_dict = get_stype_proposal(db)
        try:
            stypes_cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_out = {}
            for table, col_to_stype in col_to_stype_dict.items():
                cache_out[table] = {col: str(st_val) for col, st_val in col_to_stype.items()}
            stypes_cache_path.write_text(json.dumps(cache_out, indent=2))
            logger.info(f"Cached stypes for {dataset_name}")
        except Exception:
            logger.warning("Failed to cache stypes")

    # Exclude specific columns (for Amazon text columns that cause OOM)
    if exclude_cols:
        for table_name, cols_to_remove in exclude_cols.items():
            if table_name in col_to_stype_dict:
                for col_name in cols_to_remove:
                    if col_name in col_to_stype_dict[table_name]:
                        removed = col_to_stype_dict[table_name].pop(col_name)
                        logger.warning(
                            f"EXCLUDED {table_name}.{col_name} ({removed}) per config"
                        )

    # Also remove large text columns for safety
    for table_name in list(col_to_stype_dict.keys()):
        for col_name in list(col_to_stype_dict[table_name].keys()):
            stype_val = col_to_stype_dict[table_name][col_name]
            if str(stype_val) == "text_embedded" or stype_val == st.text_embedded:
                table_obj = db.table_dict.get(table_name)
                if table_obj is not None and len(table_obj.df) > 1_000_000:
                    removed = col_to_stype_dict[table_name].pop(col_name)
                    logger.warning(
                        f"REMOVED {table_name}.{col_name} ({removed}) - "
                        f"table has {len(table_obj.df)} rows, OOM risk"
                    )
                    DEVIATIONS.append(
                        f"Removed {table_name}.{col_name} text column "
                        f"({len(table_obj.df)} rows) to prevent OOM"
                    )

    # Check if any text columns remain
    has_text = any(
        str(v) == "text_embedded" or v == st.text_embedded
        for t_cols in col_to_stype_dict.values()
        for v in t_cols.values()
    )

    for table, stypes in col_to_stype_dict.items():
        logger.debug(f"  {table}: {dict(stypes)}")

    log_memory(f"before {dataset_name} graph build")

    text_cfg = None
    if has_text:
        text_embedder = GloveTextEmbedding(device=torch.device("cpu"))
        text_cfg = TextEmbedderConfig(text_embedder=text_embedder, batch_size=256)

    mat_cache = str(SCRIPT_DIR / "mat_cache" / dataset_name.replace("-", "_"))

    logger.info(f"Building pkey-fkey graph for {dataset_name}...")
    t0 = time.time()
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_cfg,
        cache_dir=mat_cache,
    )
    graph_time = time.time() - t0
    logger.info(f"Graph built in {graph_time:.0f}s")
    logger.info(f"Node types: {data.node_types}")
    logger.info(f"Edge types: {data.edge_types}")
    for nt in data.node_types:
        if hasattr(data[nt], 'num_nodes'):
            logger.info(f"  {nt}: {data[nt].num_nodes} nodes")

    log_memory(f"after {dataset_name} graph build")

    del db, dataset
    if text_cfg is not None:
        del text_embedder
    gc.collect()

    return task, data, col_stats_dict


def build_loaders(task, data, hp: dict):
    """Build NeighborLoaders for train/val/test splits."""
    loader_dict = {}
    entity_table = task.entity_table
    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=hp['num_neighbors'],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=hp['batch_size'],
            temporal_strategy="uniform",
            shuffle=(split == "train"),
            num_workers=0,
            persistent_workers=False,
        )
    logger.info(f"Loaders: train={len(loader_dict['train'])}, "
                f"val={len(loader_dict['val'])}, test={len(loader_dict['test'])}")
    return loader_dict, entity_table


# ============================================================================
# PARTIAL RESULTS (crash recovery)
# ============================================================================
def _save_partial(results: dict, path: Path):
    try:
        serializable = {}
        for key, val in results.items():
            if isinstance(key, tuple):
                skey = f"{key[0]}__{key[1]}__{key[2]}"
            else:
                skey = str(key)
            serializable[skey] = val
        path.write_text(json.dumps(serializable, indent=2, default=str))
        logger.debug(f"Saved partial results ({len(serializable)} runs)")
    except Exception:
        logger.exception("Failed to save partial results")


def _load_partial(path: Path) -> dict:
    results = {}
    if not path.exists():
        return results
    try:
        data = json.loads(path.read_text())
        for key, val in data.items():
            parts = key.split("__")
            if len(parts) != 3:
                continue
            method = parts[0]
            task_key = parts[1]
            seed = int(parts[2])
            if isinstance(val, dict) and val.get('best_val_metric') is not None:
                bvm = val['best_val_metric']
                if isinstance(bvm, (int, float)) and not np.isnan(bvm):
                    results[(method, task_key, seed)] = val
        logger.info(f"Loaded {len(results)} completed runs from partial results")
    except Exception:
        logger.exception("Failed to load partial results, starting fresh")
    return results


# ============================================================================
# CREATE AGGREGATION HELPER
# ============================================================================
def create_aggr(method_name: str, channels: int):
    """Create the appropriate aggregation for a method name."""
    if method_name == 'sum':
        return SafeSumAggregation()
    elif method_name == 'mean':
        return 'mean'  # String is fine for mean - no spmm issues
    elif method_name == 'cama':
        return CAMAAggregation(channels)
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ============================================================================
# SINGLE EXPERIMENT RUN
# ============================================================================
def run_single_experiment(
    task_obj, data, col_stats_dict, entity_table,
    loader_dict, method_name, seed, hp, task_type='regression'
) -> Dict[str, Any]:
    """Run one full training loop for a method/seed combination."""
    seed_everything(seed)

    aggr = create_aggr(method_name, hp['channels'])

    model = Model(
        data=data, col_stats_dict=col_stats_dict,
        num_layers=hp['num_layers'], channels=hp['channels'],
        out_channels=1, aggr=aggr,
    ).to(DEVICE)

    # CRITICAL: Verify fuse=False for non-string aggregations
    if method_name in ('sum', 'cama'):
        for conv_layer in model.gnn.convs:
            for et_key, conv in conv_layer.convs.items():
                if hasattr(conv, 'fuse') and conv.fuse:
                    logger.error(
                        f"FUSE IS TRUE for {et_key} with {method_name}! "
                        f"This will trigger spmm. Attempting force-disable."
                    )
                    conv.fuse = False

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model params: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])

    is_regression = (task_type == 'regression')
    if is_regression:
        best_val_metric = float('inf')  # Lower MAE is better
    else:
        best_val_metric = -float('inf')  # Higher AP is better

    best_epoch = 0
    best_state = None
    epoch_records = []
    gate_evolution = []

    for epoch in range(1, hp['epochs'] + 1):
        t0 = time.time()

        # Train
        if is_regression:
            train_loss = train_epoch_regression(
                model, loader_dict["train"], optimizer, entity_table,
                hp['max_steps_per_epoch'], hp['grad_clip'],
            )
        else:
            train_loss = train_epoch_classification(
                model, loader_dict["train"], optimizer, entity_table,
                hp['max_steps_per_epoch'], hp['grad_clip'],
            )

        # Validate
        if is_regression:
            val_metric, _ = evaluate_regression(
                model, loader_dict["val"], entity_table, task_obj, "val"
            )
        else:
            val_metric, _ = evaluate_classification(
                model, loader_dict["val"], entity_table, task_obj, "val"
            )

        epoch_time = time.time() - t0

        # Track best
        improved = False
        if is_regression:
            if val_metric <= best_val_metric:
                improved = True
        else:
            if val_metric >= best_val_metric:
                improved = True

        if improved:
            best_val_metric = val_metric
            best_epoch = epoch
            if best_state is not None:
                del best_state
            best_state = copy.deepcopy(model.state_dict())

        epoch_records.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_metric': float(val_metric),
            'elapsed_s': float(epoch_time),
        })

        # Gate recording for CAMA at key epochs
        if method_name == 'cama' and epoch in (1, hp['epochs'] // 2, hp['epochs']):
            try:
                start_gate_recording(model)
                if is_regression:
                    evaluate_regression(
                        model, loader_dict["val"], entity_table, task_obj, "val"
                    )
                else:
                    evaluate_classification(
                        model, loader_dict["val"], entity_table, task_obj, "val"
                    )
                gate_snapshot = collect_gate_snapshot(model)
                gate_evolution.append({'epoch': epoch, 'gates': gate_snapshot})
            except Exception:
                logger.warning(f"Gate recording failed at epoch {epoch}")

        peak_vram = 0.0
        if HAS_GPU:
            peak_vram = torch.cuda.max_memory_allocated() / 1e9

        metric_name = "mae" if is_regression else "ap"
        logger.info(
            f"  E{epoch}: loss={train_loss:.4f}, val_{metric_name}={val_metric:.4f}, "
            f"best={best_val_metric:.4f}@E{best_epoch}, time={epoch_time:.1f}s"
        )

    # Load best model for test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metric = None
    try:
        if is_regression:
            test_metric, _ = evaluate_regression(
                model, loader_dict["test"], entity_table, task_obj, "test"
            )
        else:
            test_metric, _ = evaluate_classification(
                model, loader_dict["test"], entity_table, task_obj, "test"
            )
        logger.info(f"  Test {metric_name}: {test_metric:.4f}")
    except Exception as e:
        logger.info(f"  Test labels unavailable: {e}")

    # Final gate stats for CAMA
    final_gate_stats = {}
    if method_name == 'cama' and hasattr(model.gnn, 'cama_registry'):
        try:
            start_gate_recording(model)
            if is_regression:
                evaluate_regression(
                    model, loader_dict["val"], entity_table, task_obj, "val"
                )
            else:
                evaluate_classification(
                    model, loader_dict["val"], entity_table, task_obj, "val"
                )
            final_gate_stats = collect_gate_snapshot(model)
        except Exception:
            logger.warning("Final gate extraction failed")

    result = {
        'method': method_name,
        'seed': seed,
        'best_val_metric': float(best_val_metric),
        'best_epoch': best_epoch,
        'test_metric': float(test_metric) if test_metric is not None else None,
        'epoch_records': epoch_records,
        'gate_evolution': gate_evolution,
        'final_gate_stats': final_gate_stats,
        'peak_vram_gb': float(peak_vram) if HAS_GPU else 0.0,
        'param_count': param_count,
    }

    del model, optimizer, best_state
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return result


# ============================================================================
# RUN TASK EXPERIMENTS
# ============================================================================
def run_task_experiments(
    task_cfg: dict,
    all_results: dict,
    partial_path: Path,
    remaining_minutes_fn,
    task_budget_min: float,
) -> Dict[str, List[float]]:
    """Run all method/seed combos for one task. Returns {method: [val_metrics]}."""

    dataset_name = task_cfg['dataset']
    task_name = task_cfg['task']
    task_type = task_cfg['type']
    task_key = f"{dataset_name}__{task_name}"
    hp = {k: task_cfg[k] for k in [
        'channels', 'lr', 'epochs', 'batch_size', 'num_neighbors',
        'num_layers', 'max_steps_per_epoch', 'grad_clip',
    ]}

    logger.info("=" * 60)
    logger.info(f"TASK: {task_key} ({remaining_minutes_fn():.0f} min remaining)")
    logger.info("=" * 60)

    # Load data
    try:
        task_obj, data, col_stats_dict = load_task_data(
            dataset_name, task_name,
            exclude_cols=task_cfg.get('exclude_cols'),
        )
    except Exception:
        logger.exception(f"Failed to load {task_key}")
        DEVIATIONS.append(f"Failed to load {task_key}")
        return {}

    # Build loaders
    try:
        loader_dict, entity_table = build_loaders(task_obj, data, hp)
    except torch.cuda.OutOfMemoryError:
        logger.warning(f"OOM building loaders for {task_key}, reducing params")
        torch.cuda.empty_cache()
        gc.collect()
        hp['batch_size'] = 256
        hp['num_neighbors'] = [64, 64]
        DEVIATIONS.append(
            f"Reduced batch_size=256, neighbors=[64,64] for {task_key} due to OOM"
        )
        try:
            loader_dict, entity_table = build_loaders(task_obj, data, hp)
        except Exception:
            logger.exception(f"Failed even with reduced params for {task_key}")
            DEVIATIONS.append(f"Completely failed to build loaders for {task_key}")
            del data, col_stats_dict
            gc.collect()
            return {}

    # Timing calibration: 1 epoch of mean/seed=42
    logger.info("Timing calibration: 1 epoch mean/seed=42...")
    calib_t0 = time.time()
    calib_time = 120  # default estimate
    try:
        seed_everything(42)
        calib_model = Model(
            data=data, col_stats_dict=col_stats_dict,
            num_layers=hp['num_layers'], channels=hp['channels'],
            out_channels=1, aggr='mean',
        ).to(DEVICE)
        calib_opt = torch.optim.Adam(calib_model.parameters(), lr=hp['lr'])

        if task_type == 'regression':
            calib_loss = train_epoch_regression(
                calib_model, loader_dict["train"], calib_opt, entity_table,
                hp['max_steps_per_epoch'], hp['grad_clip'],
            )
            val_metric, _ = evaluate_regression(
                calib_model, loader_dict["val"], entity_table, task_obj, "val"
            )
        else:
            calib_loss = train_epoch_classification(
                calib_model, loader_dict["train"], calib_opt, entity_table,
                hp['max_steps_per_epoch'], hp['grad_clip'],
            )
            val_metric, _ = evaluate_classification(
                calib_model, loader_dict["val"], entity_table, task_obj, "val"
            )

        calib_time = time.time() - calib_t0
        metric_name = "mae" if task_type == 'regression' else "ap"
        logger.info(
            f"  Calibration: 1 epoch in {calib_time:.1f}s, "
            f"val_{metric_name}={val_metric:.4f}, loss={calib_loss:.4f}"
        )

        if HAS_GPU:
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"  Peak VRAM: {peak_vram:.2f}GB")
            torch.cuda.reset_peak_memory_stats()

        del calib_model, calib_opt
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM during calibration, reducing params")
        torch.cuda.empty_cache()
        gc.collect()
        hp['batch_size'] = min(hp['batch_size'], 256)
        hp['num_neighbors'] = [64, 64]
        DEVIATIONS.append(f"OOM calibration for {task_key}, reduced params")
        loader_dict, entity_table = build_loaders(task_obj, data, hp)
    except Exception:
        logger.exception("Calibration failed")

    # Time budget planning
    eval_overhead = calib_time * 0.5
    est_run_time = hp['epochs'] * calib_time + eval_overhead
    task_seeds = list(SEEDS)
    est_task_time = len(METHOD_NAMES) * len(task_seeds) * est_run_time

    logger.info(f"  Est per-run: {est_run_time/60:.1f} min")
    logger.info(f"  Est total for task: {est_task_time/60:.1f} min")
    logger.info(f"  Task budget: {task_budget_min:.0f} min")

    calib_epoch_time = calib_time

    # Adaptive reduction
    if est_task_time / 60 > task_budget_min:
        old_steps = hp['max_steps_per_epoch']
        hp['max_steps_per_epoch'] = 500
        ratio = hp['max_steps_per_epoch'] / max(old_steps, 1)
        calib_epoch_time *= ratio
        est_run_time = hp['epochs'] * calib_epoch_time + calib_epoch_time * 0.5
        est_task_time = len(METHOD_NAMES) * len(task_seeds) * est_run_time
        DEVIATIONS.append(f"Reduced max_steps to {hp['max_steps_per_epoch']} for {task_key}")
        logger.info(f"  Reduced max_steps to {hp['max_steps_per_epoch']}")

    if est_task_time / 60 > task_budget_min:
        hp['num_neighbors'] = [64, 64]
        calib_epoch_time *= 0.6
        est_run_time = hp['epochs'] * calib_epoch_time + calib_epoch_time * 0.5
        est_task_time = len(METHOD_NAMES) * len(task_seeds) * est_run_time
        DEVIATIONS.append(f"Reduced num_neighbors to [64,64] for {task_key}")
        logger.info("  Reduced num_neighbors to [64,64]")
        loader_dict, entity_table = build_loaders(task_obj, data, hp)

    if est_task_time / 60 > task_budget_min:
        task_seeds = SEEDS[:3]
        est_task_time = len(METHOD_NAMES) * len(task_seeds) * est_run_time
        DEVIATIONS.append(f"Reduced seeds to {len(task_seeds)} for {task_key}")
        logger.info(f"  Reduced seeds to {len(task_seeds)}")

    if est_task_time / 60 > task_budget_min:
        hp['epochs'] = max(5, hp['epochs'] - 3)
        est_run_time = hp['epochs'] * calib_epoch_time + calib_epoch_time * 0.5
        est_task_time = len(METHOD_NAMES) * len(task_seeds) * est_run_time
        DEVIATIONS.append(f"Reduced epochs to {hp['epochs']} for {task_key}")
        logger.info(f"  Reduced epochs to {hp['epochs']}")

    logger.info(
        f"  Final plan: {hp['epochs']} epochs, {len(task_seeds)} seeds, "
        f"max_steps={hp['max_steps_per_epoch']}, neighbors={hp['num_neighbors']}"
    )

    # Run experiments: sum first (highest priority novel data), then mean, then cama
    task_method_scores: Dict[str, List[float]] = {m: [] for m in METHOD_NAMES}
    time_exhausted = False

    for seed in task_seeds:
        if time_exhausted:
            break
        for method_name in METHOD_NAMES:
            run_key = (method_name, task_key, seed)

            if run_key in all_results:
                res = all_results[run_key]
                val_m = res.get('best_val_metric')
                if val_m is not None and isinstance(val_m, (int, float)) and not np.isnan(val_m):
                    task_method_scores[method_name].append(val_m)
                logger.info(f"Skipping {run_key} (already completed, metric={val_m})")
                continue

            mins_left = remaining_minutes_fn()
            if mins_left < 8:
                logger.warning(f"Only {mins_left:.0f} min left, stopping runs")
                DEVIATIONS.append(
                    f"Stopped at {method_name}/{task_key}/seed{seed}: "
                    f"{mins_left:.0f}min left"
                )
                time_exhausted = True
                break

            logger.info(
                f"=== RUN: {method_name}/{task_key}/seed{seed} "
                f"({mins_left:.0f} min left) ==="
            )
            run_t0 = time.time()

            try:
                result = run_single_experiment(
                    task_obj, data, col_stats_dict, entity_table,
                    loader_dict, method_name, seed, hp, task_type,
                )
                result['task_key'] = task_key
                all_results[run_key] = result

                val_m = result['best_val_metric']
                if not np.isnan(val_m):
                    task_method_scores[method_name].append(val_m)

                run_time = time.time() - run_t0
                metric_name = "mae" if task_type == 'regression' else "ap"
                logger.info(
                    f"  Completed in {run_time:.1f}s, "
                    f"best_val_{metric_name}={val_m:.4f}@E{result['best_epoch']}"
                )

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"GPU OOM on {method_name}/{task_key}/seed{seed}")
                torch.cuda.empty_cache()
                gc.collect()
                DEVIATIONS.append(f"GPU OOM on {method_name}/{task_key}/seed{seed}")
                all_results[run_key] = {
                    'method': method_name, 'seed': seed,
                    'task_key': task_key,
                    'best_val_metric': float('nan'),
                    'best_epoch': 0, 'error': 'GPU OOM',
                }
            except Exception:
                logger.exception(f"FAILED: {method_name}/{task_key}/seed{seed}")
                all_results[run_key] = {
                    'method': method_name, 'seed': seed,
                    'task_key': task_key,
                    'best_val_metric': float('nan'),
                    'best_epoch': 0,
                    'error': traceback.format_exc()[:500],
                }

            _save_partial(all_results, partial_path)
            log_memory(f"after {method_name}/seed{seed}")

    # Log task summary
    logger.info(f"\n--- Task Summary: {task_key} ---")
    for m in METHOD_NAMES:
        scores = task_method_scores.get(m, [])
        if scores:
            logger.info(
                f"  {m}: mean={np.mean(scores):.4f}, "
                f"std={np.std(scores, ddof=1) if len(scores) > 1 else 0:.4f}, "
                f"n={len(scores)}, values={[f'{s:.4f}' for s in scores]}"
            )
        else:
            logger.info(f"  {m}: no results")

    # Free task data
    del data, col_stats_dict, loader_dict, task_obj
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

    return task_method_scores


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
@logger.catch
def main():
    start_time = time.time()

    def remaining_minutes() -> float:
        return (TIME_BUDGET_SEC - (time.time() - start_time)) / 60

    out_path = SCRIPT_DIR / "method_out.json"
    partial_path = SCRIPT_DIR / "partial_results.json"

    logger.info("=" * 60)
    logger.info("Amazon Sum-vs-CAMA Fix + Trial Classification")
    logger.info(f"Methods: {METHOD_NAMES}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"PART A: {PART_A_CONFIG['dataset']}/{PART_A_CONFIG['task']} (regression)")
    logger.info(f"PART B: {PART_B_CONFIG['dataset']}/{PART_B_CONFIG['task']} (classification)")
    logger.info(f"Time budget: {TIME_BUDGET_SEC/60:.0f} minutes")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # PHASE 0: UNIT TESTS
    # ------------------------------------------------------------------
    logger.info("Phase 0: Unit tests...")

    # Test 1: SafeSumAggregation
    logger.info("  Test 1: SafeSumAggregation...")
    sum_agg = SafeSumAggregation()
    x_test = torch.randn(100, 128, requires_grad=True)
    index_test = torch.randint(0, 10, (100,))
    out_test = sum_agg(x_test, index=index_test, dim_size=10)
    assert out_test.shape == (10, 128), f"Expected (10,128), got {out_test.shape}"
    # Verify numerical correctness against manual scatter_add
    with torch.no_grad():
        expected = torch.zeros(10, 128).scatter_add_(
            0, index_test.unsqueeze(1).expand(-1, 128), x_test.detach()
        )
    assert torch.allclose(out_test.detach(), expected, atol=1e-5), "SafeSumAggregation incorrect!"
    # Gradient flow
    loss_test = out_test.sum()
    loss_test.backward()
    assert x_test.grad is not None, "No gradient for input x through SafeSumAggregation"
    logger.info("  SafeSumAggregation: shape OK, numerical OK, gradient OK")

    # Test 2: CAMA
    logger.info("  Test 2: CAMAAggregation...")
    cama_test = CAMAAggregation(channels=64)
    x_test2 = torch.randn(100, 64, requires_grad=True)
    out_test = cama_test(x_test2, index=index_test, dim_size=10)
    assert out_test.shape == (10, 64), f"Expected (10,64), got {out_test.shape}"
    loss_test = out_test.sum()
    loss_test.backward()
    for name, p in cama_test.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
    cama_test.start_recording()
    cama_test.zero_grad()
    _ = cama_test(x_test2.detach(), index=index_test, dim_size=10)
    gs = cama_test.stop_recording()
    assert gs is not None, "Gate recording failed"
    assert abs(gs['mean'] - 0.5) < 0.05, f"Initial gate should be ~0.5, got {gs['mean']}"
    logger.info(f"  CAMA gate init mean={gs['mean']:.4f} (expected ~0.5)")

    # Test 3: SAGEConv fuse check - THE CRITICAL VERIFICATION
    logger.info("  Test 3: SAGEConv fuse check...")
    sage_sum = SAGEConv((128, 128), 128, aggr=SafeSumAggregation())
    sage_cama = SAGEConv((128, 128), 128, aggr=CAMAAggregation(128))
    sage_mean = SAGEConv((128, 128), 128, aggr="mean")

    assert not sage_sum.fuse, "SafeSumAggregation MUST NOT trigger fused path!"
    assert not sage_cama.fuse, "CAMA MUST NOT trigger fused path!"
    logger.info(f"  SAGEConv fuse: sum={sage_sum.fuse}, cama={sage_cama.fuse}, "
                f"mean={sage_mean.fuse}")
    logger.info("All unit tests passed!")

    del sum_agg, cama_test, sage_sum, sage_cama, sage_mean
    del x_test, x_test2, index_test, out_test
    gc.collect()

    # ------------------------------------------------------------------
    # PHASE 1: PART A - Amazon item-ltv (regression)
    # ------------------------------------------------------------------
    all_results = _load_partial(partial_path)
    all_task_results = {}  # task_key -> {method -> [scores]}

    logger.info("\n" + "=" * 60)
    logger.info("PART A: Amazon item-ltv (regression, MAE)")
    logger.info("=" * 60)

    task_budget_a = min(remaining_minutes() - 25, 50)  # Reserve 25 min for Part B
    logger.info(f"Part A budget: {task_budget_a:.0f} min")

    part_a_scores = run_task_experiments(
        PART_A_CONFIG, all_results, partial_path,
        remaining_minutes, task_budget_a,
    )
    task_key_a = f"{PART_A_CONFIG['dataset']}__{PART_A_CONFIG['task']}"
    all_task_results[task_key_a] = part_a_scores

    # ------------------------------------------------------------------
    # PHASE 2: PART B - Trial study-outcome (classification)
    # ------------------------------------------------------------------
    mins_left = remaining_minutes()
    logger.info(f"\n{'='*60}")
    logger.info(f"PART B: Trial study-outcome ({mins_left:.0f} min remaining)")
    logger.info("=" * 60)

    if mins_left < 15:
        logger.warning(f"Only {mins_left:.0f} min left, skipping Part B")
        DEVIATIONS.append(f"Skipped Part B (rel-trial): only {mins_left:.0f}min remaining")
    else:
        task_budget_b = mins_left - 8  # reserve 8 min for analysis+output
        logger.info(f"Part B budget: {task_budget_b:.0f} min")

        part_b_scores = run_task_experiments(
            PART_B_CONFIG, all_results, partial_path,
            remaining_minutes, task_budget_b,
        )
        task_key_b = f"{PART_B_CONFIG['dataset']}__{PART_B_CONFIG['task']}"
        all_task_results[task_key_b] = part_b_scores

    # ------------------------------------------------------------------
    # PHASE 3: STATISTICAL ANALYSIS
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Statistical Analysis")
    logger.info("=" * 60)

    statistical_analysis = {}
    task_configs_map = {
        f"{PART_A_CONFIG['dataset']}__{PART_A_CONFIG['task']}": PART_A_CONFIG,
        f"{PART_B_CONFIG['dataset']}__{PART_B_CONFIG['task']}": PART_B_CONFIG,
    }

    for task_key, method_scores in all_task_results.items():
        tcfg = task_configs_map.get(task_key, PART_A_CONFIG)
        higher_is_better = tcfg['higher_is_better']

        sum_scores = np.array(method_scores.get('sum', []))
        mean_scores = np.array(method_scores.get('mean', []))
        cama_scores = np.array(method_scores.get('cama', []))

        comparisons = {}
        if len(cama_scores) >= 2 and len(sum_scores) >= 2:
            comparisons['cama_vs_sum'] = compute_pairwise_stats(
                sum_scores, cama_scores, 'sum', 'cama',
                higher_is_better=higher_is_better,
            )
        if len(cama_scores) >= 2 and len(mean_scores) >= 2:
            comparisons['cama_vs_mean'] = compute_pairwise_stats(
                mean_scores, cama_scores, 'mean', 'cama',
                higher_is_better=higher_is_better,
            )
        if len(sum_scores) >= 2 and len(mean_scores) >= 2:
            comparisons['sum_vs_mean'] = compute_pairwise_stats(
                mean_scores, sum_scores, 'mean', 'sum',
                higher_is_better=higher_is_better,
            )

        per_method = {}
        for m in METHOD_NAMES:
            scores = method_scores.get(m, [])
            if scores:
                per_method[m] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                    'median': float(np.median(scores)),
                    'n_seeds': len(scores),
                    'per_seed_values': [float(s) for s in scores],
                }

        statistical_analysis[task_key] = {
            'comparisons': comparisons,
            'per_method': per_method,
            'metric': tcfg['metric'],
            'higher_is_better': higher_is_better,
        }

        logger.info(f"\n--- Stats for {task_key} ---")
        for comp_name, comp in comparisons.items():
            logger.info(
                f"  {comp_name}: d={comp['cohens_d']:.3f} "
                f"[{comp['d_ci_95'][0]:.3f}, {comp['d_ci_95'][1]:.3f}], "
                f"p_t={comp.get('p_value_ttest', 'N/A')}, "
                f"p_w={comp.get('p_value_wilcoxon', 'N/A')}"
            )

    # ------------------------------------------------------------------
    # PHASE 4: CROSS-ITERATION VALIDATION (Amazon only)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Cross-Iteration Validation")
    logger.info("=" * 60)

    cross_iter_validation = {}
    amazon_key = f"{PART_A_CONFIG['dataset']}__{PART_A_CONFIG['task']}"
    if amazon_key in all_task_results:
        amazon_scores = all_task_results[amazon_key]

        # Compare fresh mean MAEs with iter-5 reference
        fresh_mean_maes = amazon_scores.get('mean', [])
        if len(fresh_mean_maes) >= 3:
            iter5_ref = ITER5_MEAN_MAES[:len(fresh_mean_maes)]
            deviations = [abs(f - r) for f, r in zip(fresh_mean_maes, iter5_ref)]
            max_dev = max(deviations) if deviations else 0
            mean_dev = np.mean(deviations) if deviations else 0
            cross_iter_validation['mean_vs_iter5'] = {
                'fresh_values': [float(v) for v in fresh_mean_maes],
                'iter5_reference': iter5_ref,
                'per_seed_deviation': [float(d) for d in deviations],
                'max_deviation': float(max_dev),
                'mean_deviation': float(mean_dev),
                'warning': max_dev > 2.0,
            }
            if max_dev > 2.0:
                logger.warning(
                    f"Mean MAE deviation from iter-5: max={max_dev:.2f} > 2.0 threshold"
                )
            else:
                logger.info(f"Mean MAE cross-validation OK: max deviation={max_dev:.2f}")

        # Compare fresh CAMA MAEs with iter-5 reference
        fresh_cama_maes = amazon_scores.get('cama', [])
        if len(fresh_cama_maes) >= 3:
            iter5_ref = ITER5_CAMA_MAES[:len(fresh_cama_maes)]
            deviations = [abs(f - r) for f, r in zip(fresh_cama_maes, iter5_ref)]
            max_dev = max(deviations) if deviations else 0
            mean_dev = np.mean(deviations) if deviations else 0
            cross_iter_validation['cama_vs_iter5'] = {
                'fresh_values': [float(v) for v in fresh_cama_maes],
                'iter5_reference': iter5_ref,
                'per_seed_deviation': [float(d) for d in deviations],
                'max_deviation': float(max_dev),
                'mean_deviation': float(mean_dev),
                'warning': max_dev > 2.0,
            }

    # ------------------------------------------------------------------
    # PHASE 5: GATE ANALYSIS (CAMA runs across both tasks)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 5: Gate Analysis")
    logger.info("=" * 60)

    gate_analysis = {}
    for run_key, result in all_results.items():
        if not isinstance(result, dict):
            continue
        if result.get('method') != 'cama':
            continue
        task_k = result.get('task_key', '')
        seed = result.get('seed', 0)
        final_gates = result.get('final_gate_stats', {})
        gate_evo = result.get('gate_evolution', [])

        if final_gates or gate_evo:
            key = f"{task_k}/cama/seed{seed}"
            gate_analysis[key] = {
                'final_gate_stats': final_gates,
                'gate_evolution': gate_evo,
            }

    # ------------------------------------------------------------------
    # PHASE 6: KEY FINDINGS
    # ------------------------------------------------------------------
    key_findings = []
    for task_key, stats in statistical_analysis.items():
        comps = stats.get('comparisons', {})
        metric = stats.get('metric', 'unknown')
        hib = stats.get('higher_is_better', True)

        cama_vs_sum = comps.get('cama_vs_sum', {})
        if cama_vs_sum:
            d = cama_vs_sum['cohens_d']
            if d > 0.2:
                key_findings.append(
                    f"CAMA outperforms sum on {task_key} ({metric}): d={d:.3f}"
                )
            elif d < -0.2:
                key_findings.append(
                    f"Sum outperforms CAMA on {task_key} ({metric}): d={abs(d):.3f}"
                )
            else:
                key_findings.append(
                    f"CAMA and sum comparable on {task_key} ({metric}): d={d:.3f}"
                )

        cama_vs_mean = comps.get('cama_vs_mean', {})
        if cama_vs_mean:
            d = cama_vs_mean['cohens_d']
            key_findings.append(f"CAMA vs mean on {task_key}: d={d:.3f}")

        sum_vs_mean = comps.get('sum_vs_mean', {})
        if sum_vs_mean:
            d = sum_vs_mean['cohens_d']
            key_findings.append(f"Sum vs mean on {task_key}: d={d:.3f}")

    if not key_findings:
        key_findings.append("Insufficient data for statistical comparison")

    # ------------------------------------------------------------------
    # PHASE 7: OUTPUT FORMATTING (exp_gen_sol_out.json schema)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 7: Output formatting")
    logger.info("=" * 60)

    total_elapsed = time.time() - start_time

    datasets_out = []
    for task_key, method_scores in all_task_results.items():
        tcfg = task_configs_map.get(task_key, PART_A_CONFIG)
        metric = tcfg['metric']
        higher_is_better = tcfg['higher_is_better']

        examples = []
        seeds_with_data = set()
        per_seed_per_method = {}
        for run_key, result in all_results.items():
            if not isinstance(result, dict):
                continue
            r_task = result.get('task_key', '')
            if r_task != task_key:
                continue
            m = result.get('method', '')
            s = result.get('seed', 0)
            val = result.get('best_val_metric')
            if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                per_seed_per_method.setdefault(s, {})[m] = val
                seeds_with_data.add(s)

        for seed in sorted(seeds_with_data):
            seed_data = per_seed_per_method.get(seed, {})
            if not seed_data:
                continue

            if higher_is_better:
                best_val = max(seed_data.values())
                best_method = max(seed_data, key=seed_data.get)
            else:
                best_val = min(seed_data.values())
                best_method = min(seed_data, key=seed_data.get)

            input_dict = {
                'task': task_key,
                'seed': seed,
                'metric': metric,
                'higher_is_better': higher_is_better,
                'hyperparameters': {
                    'channels': tcfg['channels'],
                    'lr': tcfg['lr'],
                    'epochs': tcfg['epochs'],
                    'batch_size': tcfg['batch_size'],
                },
            }

            example = {
                'input': json.dumps(input_dict),
                'output': str(best_val),
                'metadata_seed': seed,
                'metadata_task': task_key,
                'metadata_metric': metric,
                'metadata_best_method': best_method,
            }

            for m in METHOD_NAMES:
                if m in seed_data:
                    example[f'predict_{m}'] = str(seed_data[m])

            examples.append(example)

        if examples:
            datasets_out.append({
                'dataset': task_key,
                'examples': examples,
            })

    if not datasets_out:
        datasets_out.append({
            'dataset': 'no_results',
            'examples': [{
                'input': json.dumps({'error': 'No experiments completed'}),
                'output': '0.0',
            }],
        })

    # Convert all_results to serializable list
    all_run_list = []
    for run_key, result in all_results.items():
        if isinstance(result, dict):
            clean = {}
            for k, v in result.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    clean[k] = None
                else:
                    clean[k] = v
            all_run_list.append(clean)

    output = {
        'metadata': {
            'title': (
                'Amazon Sum-vs-CAMA Fix + Trial Classification: '
                'Custom SumAggregation Bypassing pyg-lib Fused Path'
            ),
            'description': (
                'Two-part GPU experiment: (A) Amazon item-ltv regression comparing '
                'sum vs mean vs CAMA with custom SafeSumAggregation that bypasses '
                'the fused spmm path (fixing iter-6 failure), and (B) rel-trial '
                'study-outcome classification with sum baseline. '
                'SafeSumAggregation is an Aggregation object (not string) so '
                'SAGEConv.fuse=False, avoiding pyg-lib spmm dependency.'
            ),
            'methods_tested': METHOD_NAMES,
            'seeds': SEEDS,
            'part_a_config': {k: v for k, v in PART_A_CONFIG.items()
                              if k != 'exclude_cols'},
            'part_b_config': {k: v for k, v in PART_B_CONFIG.items()
                              if k != 'exclude_cols'},
            'deviations': DEVIATIONS,
            'statistical_analysis': statistical_analysis,
            'cross_iteration_validation': cross_iter_validation,
            'gate_analysis': gate_analysis,
            'key_findings': key_findings,
            'all_run_results': all_run_list,
            'total_elapsed_seconds': total_elapsed,
        },
        'datasets': datasets_out,
    }

    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output saved to {out_path}")
    logger.info(f"Total experiment time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Deviations: {DEVIATIONS}")
    logger.info(f"Key findings: {key_findings}")
    logger.info("DONE!")


if __name__ == "__main__":
    main()
