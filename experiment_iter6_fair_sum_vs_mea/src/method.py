#!/usr/bin/env python3
"""Fair Sum vs Mean vs CAMA Classification Comparison with Full Hyperparameters.

Run 3 methods (mean, sum, CAMA) x 2 classification tasks
(rel-stack/user-engagement, rel-trial/study-outcome) x 5 seeds = 30 GNN
training runs with full RelBench hyperparameters, comparing aggregation
methods on classification performance (average_precision).

This is the make-or-break experiment: if CAMA beats sum on at least one
task, the paper has a value proposition.
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
    # Also disable in the utils module if present
    import torch_geometric.sampler.utils as _sampler_utils
    if hasattr(_sampler_utils, 'pyg_lib'):
        _sampler_utils.pyg_lib = None
    # Disable torch-sparse based sampling too if it causes issues
    if hasattr(_ns_module, 'torch_sparse'):
        _ns_module.torch_sparse = None
        logger.info("Also disabled torch-sparse sampler")
except Exception as e:
    logger.warning(f"Could not disable pyg-lib sampler: {e}")
    # Fallback: patch the disjoint property
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
from torch.nn import BCEWithLogitsLoss
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
HYPERPARAMS = {
    'channels': 128,
    'lr': 0.005,
    'epochs': 10,
    'batch_size': 512,
    'num_neighbors': [128, 128],
    'num_layers': 2,
    'max_steps_per_epoch': 2000,
    'grad_clip': 1.0,
}
TASKS = [
    {
        'dataset': 'rel-stack',
        'task': 'user-engagement',
        'type': 'binary_classification',
        'metric': 'average_precision',
        'higher_is_better': True,
    },
    {
        'dataset': 'rel-trial',
        'task': 'study-outcome',
        'type': 'binary_classification',
        'metric': 'average_precision',
        'higher_is_better': True,
    },
]
METHOD_NAMES = ['mean', 'sum', 'cama']
TIME_BUDGET_SEC = 80 * 60  # 80 minutes total
RELBENCH_CACHE_DIR = (
    "/ai-inventor/aii_pipeline/runs/temp-debug-test_sbr/"
    "3_invention_loop/iter_1/gen_art/data_id3_it1__opus/"
    "temp/relbench_cache"
)
DEVIATIONS: List[str] = []


# ============================================================================
# CAMA MODULE
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
# CUSTOM HeteroGraphSAGE (supports string AND Aggregation object aggr)
# ============================================================================
class CustomHeteroGraphSAGE(nn.Module):
    """HeteroGraphSAGE that deep-copies non-string Aggregation objects
    for each SAGEConv, so each edge type gets its own CAMA instance."""

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
# MODEL CLASS
# ============================================================================
class Model(nn.Module):
    def __init__(self, data, col_stats_dict, num_layers, channels,
                 out_channels, aggr):
        """
        Args:
            aggr: 'mean', 'sum', or a CAMAAggregation instance
        """
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
# GLOVE TEXT EMBEDDING
# ============================================================================
class GloveTextEmbedding:
    """Uses sentence-transformers GloVe model for text embedding."""

    def __init__(self, device=None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device or torch.device("cpu"),
        )

    def __call__(self, sentences):
        return torch.from_numpy(
            self.model.encode(sentences, show_progress_bar=False)
        )


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================
def train_epoch(model, loader, optimizer, entity_table,
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
def evaluate_predictions(model, loader, entity_table):
    """Run inference and return sigmoid probabilities for classification."""
    model.eval()
    pred_list = []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch, entity_table).view(-1)
        probs = torch.sigmoid(logits)
        pred_list.append(probs.detach().cpu())
    if pred_list:
        return torch.cat(pred_list, dim=0).numpy()
    return np.array([])


@torch.no_grad()
def get_entity_embeddings(model, loader, entity_table, max_batches=3):
    """Collect entity embeddings after GNN for effective rank measurement."""
    model.eval()
    emb_list = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(DEVICE)
        seed_time = batch[entity_table].seed_time
        x_dict = model.encoder(batch.tf_dict)
        rel_time_dict = model.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        for nt, rel_time in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel_time
        x_dict = model.gnn(
            x_dict, batch.edge_index_dict,
            batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict,
        )
        emb = x_dict[entity_table][:seed_time.size(0)]
        emb_list.append(emb.detach().cpu().numpy())
    if emb_list:
        return np.concatenate(emb_list, axis=0)
    return np.zeros((0, 0))


# ============================================================================
# EFFECTIVE RANK MEASUREMENT
# ============================================================================
def measure_effective_rank(embeddings: np.ndarray) -> Dict[str, Any]:
    """Compute effective rank from entity embeddings via SVD entropy."""
    if embeddings.size == 0 or embeddings.shape[0] < 2:
        return {'effective_rank': 0, 'normalized_effective_rank': 0,
                'embedding_shape': list(embeddings.shape)}
    try:
        U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
        S = S[S > 1e-10]
        if len(S) == 0:
            return {'effective_rank': 0, 'normalized_effective_rank': 0,
                    'embedding_shape': list(embeddings.shape)}
        p = S / S.sum()
        H = -(p * np.log(p)).sum()
        eff_rank = float(np.exp(H))
        norm_eff_rank = eff_rank / min(embeddings.shape)
        return {
            'effective_rank': eff_rank,
            'normalized_effective_rank': norm_eff_rank,
            'embedding_shape': list(embeddings.shape),
        }
    except Exception as e:
        logger.warning(f"SVD failed: {e}")
        return {'effective_rank': 0, 'normalized_effective_rank': 0,
                'embedding_shape': list(embeddings.shape), 'error': str(e)}


# ============================================================================
# GATE STATISTICS HELPERS
# ============================================================================
def start_gate_recording(model):
    """Enable gate recording on all CAMA modules."""
    if hasattr(model.gnn, 'cama_registry'):
        for cama_mod in model.gnn.cama_registry.values():
            cama_mod.start_recording()


def collect_gate_snapshot(model) -> Dict[str, Dict]:
    """Collect gate statistics from all CAMA modules."""
    gate_snapshot = {}
    if not hasattr(model.gnn, 'cama_registry'):
        return gate_snapshot
    for (layer_idx, edge_type), cama_mod in model.gnn.cama_registry.items():
        gs = cama_mod.stop_recording()
        if gs is not None:
            et_str = format_edge_type(edge_type)
            gate_snapshot[f"L{layer_idx}_{et_str}"] = gs
    return gate_snapshot


def format_edge_type(et) -> str:
    """Convert PyG edge type tuple to readable string."""
    if isinstance(et, tuple):
        return "__".join(str(x) for x in et)
    return str(et)


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================
def cohens_d(group1, group2):
    """Pooled-std Cohen's d. Positive = group2 better (higher)."""
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


def bootstrap_ci(group1, group2, n_boot=10000, alpha=0.05):
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.RandomState(42)
    g1, g2 = np.asarray(group1), np.asarray(group2)
    n1, n2 = len(g1), len(g2)
    boot_ds = []
    for _ in range(n_boot):
        idx1 = rng.choice(n1, n1, replace=True)
        idx2 = rng.choice(n2, n2, replace=True)
        boot_ds.append(cohens_d(g1[idx1], g2[idx2]))
    return (
        float(np.percentile(boot_ds, 100 * alpha / 2)),
        float(np.percentile(boot_ds, 100 * (1 - alpha / 2))),
    )


def paired_ttest(group1, group2):
    """Paired t-test for matched seeds."""
    g1, g2 = np.asarray(group1), np.asarray(group2)
    if len(g1) != len(g2) or len(g1) < 3:
        return None, None
    try:
        t_stat, p_val = sp_stats.ttest_rel(g1, g2)
        return float(t_stat), float(p_val)
    except Exception:
        return None, None


def compute_pairwise_stats(scores_a, scores_b, name_a, name_b, higher_is_better=True):
    """Compute full pairwise comparison statistics.
    Positive d means B is better than A (for higher_is_better=True metrics)."""
    if higher_is_better:
        d = cohens_d(scores_a, scores_b)  # positive = B better
    else:
        d = cohens_d(scores_b, scores_a)  # flip for lower-is-better
    ci_lo, ci_hi = bootstrap_ci(scores_a, scores_b)
    t_stat, p_val = paired_ttest(scores_a, scores_b)
    return {
        'comparison': f"{name_a}_vs_{name_b}",
        'cohens_d': d,
        'd_ci_95': [ci_lo, ci_hi],
        't_statistic': t_stat,
        'p_value': p_val,
        f'{name_a}_mean': float(np.mean(scores_a)),
        f'{name_a}_std': float(np.std(scores_a, ddof=1)) if len(scores_a) > 1 else 0.0,
        f'{name_b}_mean': float(np.mean(scores_b)),
        f'{name_b}_std': float(np.std(scores_b, ddof=1)) if len(scores_b) > 1 else 0.0,
        'n_seeds': min(len(scores_a), len(scores_b)),
        'interpretation': (
            f"{name_b} {'better' if d > 0 else 'worse'} than {name_a} "
            f"by d={abs(d):.3f}"
        ),
    }


# ============================================================================
# MEMORY LOGGING
# ============================================================================
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
# DATA LOADING
# ============================================================================
def load_task_data(dataset_name: str, task_name: str):
    """Load RelBench dataset and build materialized hetero graph."""
    from torch_frame import stype as st

    logger.info(f"Loading {dataset_name}/{task_name}...")
    t0 = time.time()

    os.environ["RELBENCH_CACHE_DIR"] = RELBENCH_CACHE_DIR
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
        # Cache for future use
        try:
            stypes_cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_out = {}
            for table, col_to_stype in col_to_stype_dict.items():
                cache_out[table] = {col: str(st_val) for col, st_val in col_to_stype.items()}
            stypes_cache_path.write_text(json.dumps(cache_out, indent=2))
            logger.info(f"Cached stypes for {dataset_name}")
        except Exception:
            logger.warning("Failed to cache stypes")

    # Handle text columns per dataset
    has_text = False
    for table, cols in col_to_stype_dict.items():
        for col, st_val in list(cols.items()):
            if str(st_val) == "text_embedded" or st_val == st.text_embedded:
                has_text = True
                break

    # For rel-stack: remove potentially huge text columns to avoid OOM
    if dataset_name == "rel-stack":
        for table_name in list(col_to_stype_dict.keys()):
            for col_name in list(col_to_stype_dict[table_name].keys()):
                stype_val = col_to_stype_dict[table_name][col_name]
                if str(stype_val) == "text_embedded" or stype_val == st.text_embedded:
                    # Check if table is large
                    table_obj = db.table_dict.get(table_name)
                    if table_obj is not None and len(table_obj.df) > 500_000:
                        removed = col_to_stype_dict[table_name].pop(col_name)
                        logger.warning(
                            f"REMOVED {table_name}.{col_name} ({removed}) - "
                            f"table has {len(table_obj.df)} rows, OOM risk"
                        )
                        DEVIATIONS.append(
                            f"Removed {table_name}.{col_name} text column "
                            f"({len(table_obj.df)} rows) to prevent OOM"
                        )
                        has_text = any(
                            str(v) == "text_embedded" or v == st.text_embedded
                            for t_cols in col_to_stype_dict.values()
                            for v in t_cols.values()
                        )

    # Log remaining features
    for table, stypes in col_to_stype_dict.items():
        logger.debug(f"  {table}: {dict(stypes)}")

    log_memory(f"before {dataset_name} graph build")

    # Text embedder (only if needed)
    text_cfg = None
    if has_text:
        text_embedder = GloveTextEmbedding(device=torch.device("cpu"))
        text_cfg = TextEmbedderConfig(text_embedder=text_embedder, batch_size=256)

    mat_cache = str(SCRIPT_DIR / "mat_cache" / f"{dataset_name}_notext")

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

    # Free temporary objects
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
    """Save partial results to disk for crash recovery."""
    try:
        serializable = {}
        for key, val in results.items():
            skey = f"{key[0]}__{key[1]}__{key[2]}"
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
            if len(parts) != 3:
                continue
            method = parts[0]
            task_key = parts[1]
            seed = int(parts[2])
            if val.get('best_val_metric') is not None:
                results[(method, task_key, seed)] = val
        logger.info(f"Loaded {len(results)} completed runs from partial results")
    except Exception:
        logger.exception("Failed to load partial results, starting fresh")
    return results


# ============================================================================
# SINGLE EXPERIMENT RUN
# ============================================================================
def run_single_experiment(
    task, data, col_stats_dict, entity_table,
    loader_dict, method_name, seed, hp
) -> Dict[str, Any]:
    """Run one full training loop for a method/seed combination."""
    seed_everything(seed)

    # Create aggregation
    if method_name == 'cama':
        aggr = CAMAAggregation(channels=hp['channels'])
    else:
        aggr = method_name  # 'mean' or 'sum' string

    model = Model(
        data=data, col_stats_dict=col_stats_dict,
        num_layers=hp['num_layers'], channels=hp['channels'],
        out_channels=1, aggr=aggr,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model params: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
    best_val_metric = -math.inf
    best_epoch = 0
    best_state = None
    epoch_records = []
    gate_evolution = []

    for epoch in range(1, hp['epochs'] + 1):
        t0 = time.time()

        # Train
        train_loss = train_epoch(
            model, loader_dict["train"], optimizer, entity_table,
            hp['max_steps_per_epoch'], hp['grad_clip'],
        )

        # Validate
        val_pred = evaluate_predictions(model, loader_dict["val"], entity_table)
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        val_ap = float(val_metrics.get('average_precision', 0.0))
        epoch_time = time.time() - t0

        # Track best (higher AP is better)
        if val_ap >= best_val_metric:
            best_val_metric = val_ap
            best_epoch = epoch
            if best_state is not None:
                del best_state
            best_state = copy.deepcopy(model.state_dict())

        epoch_records.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_ap': val_ap,
            'elapsed_s': float(epoch_time),
        })

        # Gate recording for CAMA
        if method_name == 'cama' and hasattr(model.gnn, 'cama_registry'):
            try:
                start_gate_recording(model)
                _ = evaluate_predictions(model, loader_dict["val"], entity_table)
                gate_snapshot = collect_gate_snapshot(model)
                gate_evolution.append({'epoch': epoch, 'gates': gate_snapshot})
            except Exception:
                logger.warning(f"Gate recording failed at epoch {epoch}")

        if HAS_GPU:
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
        else:
            peak_vram = 0.0

        logger.info(
            f"  E{epoch}: loss={train_loss:.4f}, val_ap={val_ap:.4f}, "
            f"best={best_val_metric:.4f}@E{best_epoch}, time={epoch_time:.1f}s"
        )

    # Load best model for test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation (may fail if labels unavailable)
    test_metric = None
    try:
        test_pred = evaluate_predictions(model, loader_dict["test"], entity_table)
        test_metrics = task.evaluate(test_pred, task.get_table("test"))
        test_metric = float(test_metrics.get('average_precision', 0.0))
        logger.info(f"  Test AP: {test_metric:.4f}")
    except Exception as e:
        logger.info(f"  Test labels unavailable: {e}")

    # Final gate stats for CAMA
    final_gate_stats = {}
    if method_name == 'cama' and hasattr(model.gnn, 'cama_registry'):
        try:
            start_gate_recording(model)
            _ = evaluate_predictions(model, loader_dict["val"], entity_table)
            final_gate_stats = collect_gate_snapshot(model)
        except Exception:
            logger.warning("Final gate extraction failed")

    # Effective rank measurement
    eff_rank = {}
    try:
        emb = get_entity_embeddings(model, loader_dict["val"], entity_table, max_batches=3)
        eff_rank = measure_effective_rank(emb)
        del emb
    except Exception:
        logger.warning("Effective rank measurement failed")

    result = {
        'method': method_name,
        'seed': seed,
        'best_val_metric': float(best_val_metric),
        'best_epoch': best_epoch,
        'test_metric': test_metric,
        'epoch_records': epoch_records,
        'gate_evolution': gate_evolution,
        'final_gate_stats': final_gate_stats,
        'effective_rank': eff_rank,
        'peak_vram_gb': float(peak_vram) if HAS_GPU else 0.0,
        'param_count': param_count,
    }

    # Cleanup
    del model, optimizer, best_state
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return result


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
    logger.info("Fair Sum vs Mean vs CAMA Classification Comparison")
    logger.info(f"Methods: {METHOD_NAMES}")
    logger.info(f"Tasks: {[(t['dataset'], t['task']) for t in TASKS]}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Hyperparams: {HYPERPARAMS}")
    logger.info(f"Time budget: {TIME_BUDGET_SEC/60:.0f} minutes")
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

    # Gate values near 0.5 at init
    cama_test.start_recording()
    cama_test.zero_grad()
    _ = cama_test(x_test, index=index_test, dim_size=10)
    gs = cama_test.stop_recording()
    assert gs is not None, "Gate recording failed"
    assert abs(gs['mean'] - 0.5) < 0.05, f"Initial gate should be ~0.5, got {gs['mean']}"
    logger.info(f"  CAMA gate init mean={gs['mean']:.4f} (expected ~0.5)")
    logger.info("CAMA unit tests passed!")

    del cama_test, x_test, index_test, out_test
    gc.collect()

    # ------------------------------------------------------------------
    # PHASE 2: MAIN EXPERIMENT LOOP
    # ------------------------------------------------------------------
    all_results = _load_partial(partial_path)
    hp = dict(HYPERPARAMS)  # mutable copy
    seeds_to_use = list(SEEDS)
    task_results = {}  # task_key -> {method -> [val_metrics per seed]}

    for task_cfg in TASKS:
        task_key = f"{task_cfg['dataset']}__{task_cfg['task']}"
        dataset_name = task_cfg['dataset']
        task_name = task_cfg['task']

        mins_left = remaining_minutes()
        if mins_left < 15:
            logger.warning(f"Only {mins_left:.0f} min left, skipping {task_key}")
            DEVIATIONS.append(f"Skipped {task_key}: {mins_left:.0f}min remaining")
            continue

        logger.info("=" * 60)
        logger.info(f"TASK: {task_key} ({mins_left:.0f} min remaining)")
        logger.info("=" * 60)

        # Load data
        try:
            task_obj, data, col_stats_dict = load_task_data(dataset_name, task_name)
        except Exception:
            logger.exception(f"Failed to load {task_key}")
            DEVIATIONS.append(f"Failed to load {task_key}")
            continue

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
                continue

        # Timing calibration: 1 epoch of mean/seed=42
        logger.info("Timing calibration: 1 epoch mean/seed=42...")
        calib_t0 = time.time()
        try:
            seed_everything(42)
            calib_model = Model(
                data=data, col_stats_dict=col_stats_dict,
                num_layers=hp['num_layers'], channels=hp['channels'],
                out_channels=1, aggr='mean',
            ).to(DEVICE)
            calib_opt = torch.optim.Adam(calib_model.parameters(), lr=hp['lr'])
            calib_loss = train_epoch(
                calib_model, loader_dict["train"], calib_opt, entity_table,
                hp['max_steps_per_epoch'], hp['grad_clip'],
            )
            calib_time = time.time() - calib_t0

            # Quick val check
            val_pred = evaluate_predictions(calib_model, loader_dict["val"], entity_table)
            val_metrics = task_obj.evaluate(val_pred, task_obj.get_table("val"))
            calib_val_ap = float(val_metrics.get('average_precision', 0.0))
            logger.info(f"  Calibration: 1 epoch in {calib_time:.1f}s, "
                        f"val_ap={calib_val_ap:.4f}, loss={calib_loss:.4f}")

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
            calib_time = 120  # estimate
        except Exception:
            logger.exception("Calibration failed")
            calib_time = 120

        # Time budget planning for this task
        eval_overhead = calib_time * 0.5
        est_run_time = hp['epochs'] * calib_time + eval_overhead
        est_task_time = len(METHOD_NAMES) * len(seeds_to_use) * est_run_time
        mins_left = remaining_minutes()
        task_budget_min = min(mins_left - 10, 45)  # max 45 min per task

        logger.info(f"  Est per-run: {est_run_time/60:.1f} min")
        logger.info(f"  Est total for task: {est_task_time/60:.1f} min")
        logger.info(f"  Task budget: {task_budget_min:.0f} min")

        # Adaptive reduction — apply ALL reductions needed to fit budget.
        # Priority: reduce max_steps -> neighbors -> seeds -> epochs
        # Each step recalculates estimated time based on measured calibration.
        task_hp = dict(hp)
        task_seeds = list(seeds_to_use)
        calib_epoch_time = calib_time  # measured seconds per epoch

        # Step 1: Reduce max_steps_per_epoch
        if est_task_time / 60 > task_budget_min:
            # Proportional reduction: new time = old time * (new_steps / old_steps)
            old_steps = task_hp['max_steps_per_epoch']
            task_hp['max_steps_per_epoch'] = 500
            ratio = task_hp['max_steps_per_epoch'] / max(old_steps, 1)
            calib_epoch_time *= ratio
            est_run_time = task_hp['epochs'] * calib_epoch_time + calib_epoch_time * 0.5
            est_task_time = len(METHOD_NAMES) * len(task_seeds) * est_run_time
            DEVIATIONS.append(
                f"Reduced max_steps to {task_hp['max_steps_per_epoch']} for {task_key}"
            )
            logger.info(f"  Reduced max_steps to {task_hp['max_steps_per_epoch']}")

        # Step 2: Reduce num_neighbors
        if est_task_time / 60 > task_budget_min:
            task_hp['num_neighbors'] = [64, 64]
            calib_epoch_time *= 0.6
            est_run_time = task_hp['epochs'] * calib_epoch_time + calib_epoch_time * 0.5
            est_task_time = len(METHOD_NAMES) * len(task_seeds) * est_run_time
            DEVIATIONS.append(
                f"Reduced num_neighbors to [64,64] for {task_key}"
            )
            logger.info("  Reduced num_neighbors to [64,64]")
            # Rebuild loaders with reduced neighbors
            loader_dict, entity_table = build_loaders(task_obj, data, task_hp)

        # Step 3: Reduce seeds
        if est_task_time / 60 > task_budget_min:
            task_seeds = SEEDS[:3]
            est_task_time = len(METHOD_NAMES) * len(task_seeds) * est_run_time
            DEVIATIONS.append(f"Reduced seeds to {len(task_seeds)} for {task_key}")
            logger.info(f"  Reduced seeds to {len(task_seeds)}")

        # Step 4: Reduce epochs (last resort)
        if est_task_time / 60 > task_budget_min:
            task_hp['epochs'] = max(5, task_hp['epochs'] - 3)
            est_run_time = task_hp['epochs'] * calib_epoch_time + calib_epoch_time * 0.5
            est_task_time = len(METHOD_NAMES) * len(task_seeds) * est_run_time
            DEVIATIONS.append(f"Reduced epochs to {task_hp['epochs']} for {task_key}")
            logger.info(f"  Reduced epochs to {task_hp['epochs']}")

        # Log final estimated per-run time
        logger.info(f"  Estimated per-run time after reductions: {est_run_time/60:.1f} min")
        logger.info(f"  Estimated total task time: {est_task_time/60:.1f} min")

        logger.info(
            f"  Final plan for {task_key}: {task_hp['epochs']} epochs, "
            f"{len(task_seeds)} seeds, max_steps={task_hp['max_steps_per_epoch']}, "
            f"neighbors={task_hp['num_neighbors']}"
        )

        # Run experiments for this task
        # CRITICAL: Breadth-first loop order (seed outer, method inner)
        # ensures we get at least 1 seed of each method before 2nd seed
        task_results[task_key] = {m: [] for m in METHOD_NAMES}
        time_exhausted = False

        for seed in task_seeds:
            if time_exhausted:
                break
            for method_name in METHOD_NAMES:
                run_key = (method_name, task_key, seed)

                # Skip completed
                if run_key in all_results:
                    res = all_results[run_key]
                    val_m = res.get('best_val_metric')
                    if val_m is not None and not np.isnan(val_m):
                        task_results[task_key][method_name].append(val_m)
                    logger.info(f"Skipping {run_key} (already completed, AP={val_m})")
                    continue

                mins_left = remaining_minutes()
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
                        loader_dict, method_name, seed, task_hp,
                    )
                    result['task_key'] = task_key
                    all_results[run_key] = result

                    val_m = result['best_val_metric']
                    if not np.isnan(val_m):
                        task_results[task_key][method_name].append(val_m)

                    run_time = time.time() - run_t0
                    logger.info(
                        f"  Completed in {run_time:.1f}s, "
                        f"best_val_ap={val_m:.4f}@E{result['best_epoch']}"
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
            scores = task_results[task_key].get(m, [])
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

    # ------------------------------------------------------------------
    # PHASE 3: STATISTICAL ANALYSIS
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 3: Statistical Analysis")
    logger.info("=" * 60)

    statistical_analysis = {}
    for task_key, method_scores in task_results.items():
        mean_scores = np.array(method_scores.get('mean', []))
        sum_scores = np.array(method_scores.get('sum', []))
        cama_scores = np.array(method_scores.get('cama', []))

        comparisons = {}
        if len(cama_scores) >= 2 and len(mean_scores) >= 2:
            comparisons['cama_vs_mean'] = compute_pairwise_stats(
                mean_scores, cama_scores, 'mean', 'cama', higher_is_better=True
            )
        if len(cama_scores) >= 2 and len(sum_scores) >= 2:
            comparisons['cama_vs_sum'] = compute_pairwise_stats(
                sum_scores, cama_scores, 'sum', 'cama', higher_is_better=True
            )
        if len(mean_scores) >= 2 and len(sum_scores) >= 2:
            comparisons['mean_vs_sum'] = compute_pairwise_stats(
                mean_scores, sum_scores, 'mean', 'sum', higher_is_better=True
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
        }

        logger.info(f"\n--- Stats for {task_key} ---")
        for comp_name, comp in comparisons.items():
            logger.info(
                f"  {comp_name}: d={comp['cohens_d']:.3f} "
                f"[{comp['d_ci_95'][0]:.3f}, {comp['d_ci_95'][1]:.3f}], "
                f"p={comp.get('p_value', 'N/A')}"
            )

    # ------------------------------------------------------------------
    # PHASE 4: EFFECTIVE RANK SUMMARY
    # ------------------------------------------------------------------
    effective_rank_summary = {}
    for run_key, result in all_results.items():
        if isinstance(result, dict) and result.get('effective_rank'):
            method = result.get('method', run_key[0])
            task_k = result.get('task_key', run_key[1] if len(run_key) > 1 else '')
            seed = result.get('seed', run_key[2] if len(run_key) > 2 else 0)
            key = f"{task_k}/{method}/seed{seed}"
            effective_rank_summary[key] = result['effective_rank']

    # ------------------------------------------------------------------
    # PHASE 5: KEY FINDINGS
    # ------------------------------------------------------------------
    key_findings = []
    conclusion_parts = []

    for task_key, stats in statistical_analysis.items():
        comps = stats.get('comparisons', {})

        # CAMA vs sum - the key comparison
        cama_vs_sum = comps.get('cama_vs_sum', {})
        if cama_vs_sum:
            d = cama_vs_sum['cohens_d']
            if d > 0.2:
                key_findings.append(
                    f"CAMA outperforms sum on {task_key}: d={d:.3f} "
                    f"(medium+ effect size)"
                )
                conclusion_parts.append(f"CAMA shows promise on {task_key}")
            elif d < -0.2:
                key_findings.append(
                    f"Sum outperforms CAMA on {task_key}: d={abs(d):.3f}"
                )
                conclusion_parts.append(f"Sum beats CAMA on {task_key}")
            else:
                key_findings.append(
                    f"CAMA and sum comparable on {task_key}: d={d:.3f}"
                )
                conclusion_parts.append(f"CAMA matches sum on {task_key}")

        # CAMA vs mean
        cama_vs_mean = comps.get('cama_vs_mean', {})
        if cama_vs_mean:
            d = cama_vs_mean['cohens_d']
            key_findings.append(
                f"CAMA vs mean on {task_key}: d={d:.3f}"
            )

        # Mean vs sum
        mean_vs_sum = comps.get('mean_vs_sum', {})
        if mean_vs_sum:
            d = mean_vs_sum['cohens_d']
            key_findings.append(
                f"Mean vs sum on {task_key}: d={d:.3f}"
            )

    if not key_findings:
        key_findings.append("Insufficient data for statistical comparison")

    conclusion = "; ".join(conclusion_parts) if conclusion_parts else (
        "Experiment did not produce enough results for a conclusion"
    )

    # ------------------------------------------------------------------
    # PHASE 6: OUTPUT FORMATTING
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 6: Output formatting")
    logger.info("=" * 60)

    total_elapsed = time.time() - start_time

    # Build per-seed examples for each task
    datasets_out = []
    for task_cfg in TASKS:
        task_key = f"{task_cfg['dataset']}__{task_cfg['task']}"
        if task_key not in task_results:
            continue

        examples = []
        method_scores = task_results[task_key]

        # Collect per-seed results across methods
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
            if val is not None and not np.isnan(val):
                per_seed_per_method.setdefault(s, {})[m] = val
                seeds_with_data.add(s)

        for seed in sorted(seeds_with_data):
            seed_data = per_seed_per_method.get(seed, {})
            best_val = max(seed_data.values()) if seed_data else 0.0
            best_method = max(seed_data, key=seed_data.get) if seed_data else 'unknown'

            input_dict = {
                'task': task_key,
                'seed': seed,
                'metric': 'average_precision',
                'hyperparameters': HYPERPARAMS,
            }

            example = {
                'input': json.dumps(input_dict),
                'output': str(best_val),
                'metadata_seed': seed,
                'metadata_task': task_key,
                'metadata_best_method': best_method,
            }

            # Add predictions per method
            for m in METHOD_NAMES:
                if m in seed_data:
                    example[f'predict_{m}'] = str(seed_data[m])

            examples.append(example)

        if examples:
            datasets_out.append({
                'dataset': task_key,
                'examples': examples,
            })

    # Ensure at least one dataset with examples
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
                if k in ('epoch_records', 'gate_evolution', 'final_gate_stats',
                          'effective_rank'):
                    clean[k] = v
                elif isinstance(v, float) and np.isnan(v):
                    clean[k] = None
                else:
                    clean[k] = v
            all_run_list.append(clean)

    output = {
        'metadata': {
            'title': 'Sum vs Mean vs CAMA on Classification Tasks with Full Hyperparameters',
            'description': (
                'Fair comparison of 3 aggregation methods (mean, sum, CAMA) on '
                '2 RelBench classification tasks with full hyperparameters. '
                'CAMA (Cardinality-Aware Moment Aggregation) enriches mean with '
                'variance information gated by log-cardinality.'
            ),
            'methods_tested': METHOD_NAMES,
            'hyperparameters': HYPERPARAMS,
            'hyperparameter_deviations': DEVIATIONS,
            'seeds': SEEDS,
            'evaluation_metric': 'average_precision (val-based, test labels may be unavailable)',
            'statistical_analysis': statistical_analysis,
            'effective_rank_analysis': effective_rank_summary,
            'key_findings': key_findings,
            'conclusion': conclusion,
            'all_run_results': all_run_list,
            'total_elapsed_seconds': total_elapsed,
        },
        'datasets': datasets_out,
    }

    # Write output
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output saved to {out_path}")
    logger.info(f"Total experiment time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Deviations: {DEVIATIONS}")
    logger.info("DONE!")


if __name__ == "__main__":
    main()
