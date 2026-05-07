#!/usr/bin/env python3
"""RAMA Evaluation on New RelBench Tasks (rel-trial / rel-ratebeer fallback).

Extends RAMA evaluation to 2 new entity tasks:
  PRIMARY: rel-trial study-outcome (classification) + study-adverse (regression)
  BACKUP:  rel-ratebeer user-churn (classification) + user-count (regression)

2 tasks x 2 configs (baseline_mean, rama) x 5 seeds = 20 training runs.
Includes post-votes diagnosis as secondary objective.
Statistical analysis via paired Cohen's d.
"""

import copy
import gc
import itertools
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy import stats as scipy_stats
from scipy.stats import norm

# ===================================================================
# SETUP
# ===================================================================
WS = Path(__file__).resolve().parent
LOG_DIR = WS / "logs"
RESULTS_DIR = WS / "results"
CACHE_DIR = WS / "cache"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Dependency workspaces (read-only)
DEP_BASE = Path(
    "/ai-inventor/aii_pipeline/runs/leskovec-predictive-residual-message-passing-v2_sti"
    "/3_invention_loop/iter_1/gen_art"
)
DEP_DATA3 = DEP_BASE / "data_id3_it1__opus"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ===================================================================
# HARDWARE DETECTION (cgroup-aware)
# ===================================================================
def _detect_cpus() -> int:
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
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or 42.0

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
            f"GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM), device={DEVICE}")

# ===================================================================
# RESOURCE LIMITS
# ===================================================================
import psutil

_avail = psutil.virtual_memory().available
RAM_BUDGET = int(min(38 * 1e9, _avail * 0.85))
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f}GB (available: {_avail / 1e9:.1f}GB)")

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.9)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))
    logger.info(f"VRAM budget: {VRAM_BUDGET / 1e9:.1f}GB / {_total / 1e9:.1f}GB")

# ===================================================================
# PyG IMPORTS
# ===================================================================
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn import HeteroConv
from torch_geometric.loader import NeighborLoader

# ===================================================================
# RELBENCH IMPORTS
# ===================================================================
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.utils import get_stype_proposal

# ===================================================================
# HYPERPARAMETERS (per-dataset)
# ===================================================================
SEEDS = [42, 123, 456, 789, 1024]
CONFIGS = ["baseline_mean", "rama"]

# These are set during dataset discovery
CHANNELS = 128
NUM_LAYERS = 2
BATCH_SIZE = 512
NUM_NEIGHBORS_PER_LAYER = 128
LR = 0.005
EPOCHS = 10

# Task configuration (populated during Phase 1)
TASK_CONFIGS = []
DATASET_NAME = ""
BASE_HPARAMS = {}


# ===================================================================
# PART 1: AGGREGATION MODULES
# ===================================================================
class RAMAAggregation(Aggregation):
    """Rank-Aware Moment Aggregation.

    Enriches mean pooling with gated variance injection,
    controlled by effective spectral rank and FK cardinality.
    """
    def __init__(self, channels: int, gate_hidden: int = 16, eps: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.gate_hidden = gate_hidden
        self.eps = eps
        self.W_sigma = nn.Linear(channels, channels, bias=False)
        self.W_g = nn.Linear(2, gate_hidden)
        self.gate_out = nn.Linear(gate_hidden, channels)
        # For gate analysis logging
        self._last_gate_mean = None
        self._last_rho_mean = None
        self._last_N_mean = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.W_sigma.weight)   # Start as pure mean
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        # Step 1: Mean
        mu = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        # Step 2: Variance via E[X^2] - E[X]^2
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce='mean')
        var = (mean_x2 - mu * mu).clamp(min=0)
        # Step 3: Cardinality
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        N = self.reduce(ones, index, ptr, dim_size, dim, reduce='sum')
        # Step 4: Effective rank proxy
        var_sum = var.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        p = var / var_sum
        p = p.clamp(min=self.eps)
        H = -(p * p.log()).sum(dim=-1, keepdim=True)
        log_d = math.log(self.channels) if self.channels > 1 else 1.0
        rho = H / log_d  # [dim_size, 1], in [0, 1]
        # Step 5: Gate
        gate_input = torch.cat([torch.log1p(N), rho], dim=-1)  # [dim_size, 2]
        g = torch.sigmoid(self.gate_out(F.relu(self.W_g(gate_input))))  # [dim_size, d]
        # Log for analysis (detached)
        with torch.no_grad():
            self._last_gate_mean = g.mean().item()
            self._last_rho_mean = rho.mean().item()
            self._last_N_mean = N.mean().item()
        # Step 6: Output
        return mu + g * self.W_sigma(var)


class StandardMeanAggregation(Aggregation):
    """Baseline: Standard mean aggregation wrapped in Aggregation subclass."""
    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        return self.reduce(x, index, ptr, dim_size, dim, reduce="mean")


def make_aggregation(config_name: str, channels: int):
    """Factory function to create aggregation objects."""
    if config_name == "baseline_mean":
        return StandardMeanAggregation()
    elif config_name == "rama":
        return RAMAAggregation(channels=channels, gate_hidden=16)
    else:
        raise ValueError(f"Unknown config: {config_name}")


# ===================================================================
# PART 1b: UNIT TESTS
# ===================================================================
def run_unit_tests():
    """Phase 0: Verify RAMA module before any training."""
    logger.info("=" * 60)
    logger.info("PHASE 0: RAMA Unit Tests")
    logger.info("=" * 60)
    n_edges, n_groups, d = 100, 10, 64

    # Test 1: Shape correctness for both configs
    for config_name in CONFIGS:
        aggr = make_aggregation(config_name, d)
        x = torch.randn(n_edges, d)
        index = torch.randint(0, n_groups, (n_edges,))
        out = aggr(x, index=index, dim_size=n_groups)
        assert out.shape == (n_groups, d), \
            f"{config_name}: expected ({n_groups}, {d}), got {out.shape}"
        assert not torch.isnan(out).any(), f"{config_name}: NaN in output"
        assert not torch.isinf(out).any(), f"{config_name}: Inf in output"
        logger.info(f"  {config_name}: shape OK {out.shape}, no NaN/Inf")

    # Test 2: Zero-init means output approx mean aggregation
    rama = RAMAAggregation(d)
    x = torch.randn(n_edges, d)
    index = torch.randint(0, n_groups, (n_edges,))
    rama_out = rama(x, index=index, dim_size=n_groups)
    mean_aggr = StandardMeanAggregation()
    mean_out = mean_aggr(x, index=index, dim_size=n_groups)
    diff = (rama_out - mean_out).abs().max().item()
    assert diff < 1e-5, f"RAMA zero-init: diff={diff} (expected <1e-5)"
    logger.info(f"  RAMA zero-init test: max diff={diff:.2e} OK")

    # Test 3: Gradient flow
    rama = RAMAAggregation(d)
    x_test = torch.randn(n_edges, d, requires_grad=True)
    out = rama(x_test, index=index, dim_size=n_groups)
    loss = out.sum()
    loss.backward()
    assert x_test.grad is not None, "No gradient on input"
    assert rama.W_g.weight.grad is not None, "No gradient to W_g"
    assert rama.W_sigma.weight.grad is not None, "No gradient to W_sigma"
    logger.info(f"  Gradient flow test: OK")

    # Test 4: SAGEConv integration
    sage = SAGEConv((d, d), d, aggr=RAMAAggregation(d))
    x_src = torch.randn(50, d)
    x_dst = torch.randn(10, d)
    edge_index = torch.stack([
        torch.randint(0, 50, (80,)),
        torch.randint(0, 10, (80,)),
    ])
    out = sage((x_src, x_dst), edge_index)
    assert out.shape == (10, d), f"SAGEConv+RAMA: expected (10, {d}), got {out.shape}"
    logger.info(f"  SAGEConv+RAMA integration: OK {out.shape}")

    # Test 5: Edge cases - single element groups
    x_single = torch.randn(3, d)
    index_single = torch.tensor([0, 1, 2])
    rama_single = RAMAAggregation(d)
    out_single = rama_single(x_single, index=index_single, dim_size=3)
    assert out_single.shape == (3, d)
    assert not torch.isnan(out_single).any(), "NaN with single-element groups"
    logger.info(f"  Single-element groups: OK")

    logger.info("All unit tests PASSED")


# ===================================================================
# PART 2: MODEL ARCHITECTURE
# ===================================================================
class HeteroGraphSAGECustom(nn.Module):
    """HeteroGraphSAGE with custom aggregation.

    Adapted from relbench.modeling.nn.HeteroGraphSAGE to support
    custom Aggregation objects. Each edge type per layer gets its own
    deepcopy of the aggregation module.
    """
    def __init__(self, node_types, edge_types, channels, aggr_factory, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                edge_aggr = aggr_factory(channels)
                conv_dict[edge_type] = SAGEConv(
                    (channels, channels), channels, aggr=edge_aggr
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

            norm_dict = {}
            for node_type in node_types:
                norm_dict[node_type] = nn.LayerNorm(channels)
            self.norms.append(nn.ModuleDict(norm_dict))

    def forward(self, x_dict, edge_index_dict):
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                key: F.relu(norm_dict[key](x))
                for key, x in x_dict.items()
            }
        return x_dict


class EntityModel(nn.Module):
    """Entity prediction model for RelBench.

    Architecture:
    1. Per-table feature encoder (TensorFrame -> embedding)
    2. HeteroGraphSAGE with custom aggregation
    3. MLP prediction head
    """
    def __init__(self, data, col_stats_dict, num_layers, channels,
                 out_channels, aggr_factory, norm="layer_norm"):
        super().__init__()
        self.channels = channels
        self.node_types = data.node_types
        self.edge_types = data.edge_types

        # Per-node-type encoders
        self.encoders = nn.ModuleDict()
        for node_type in data.node_types:
            if hasattr(data[node_type], "tf") and data[node_type].tf is not None:
                tf = data[node_type].tf
                try:
                    from torch_frame import stype
                    from torch_frame.nn import (
                        StypeWiseFeatureEncoder,
                        LinearEncoder,
                        EmbeddingEncoder,
                        TimestampEncoder,
                    )
                    encoder = StypeWiseFeatureEncoder(
                        out_channels=channels,
                        col_stats=col_stats_dict[node_type],
                        col_names_dict=tf.col_names_dict,
                        stype_encoder_dict={
                            stype.numerical: LinearEncoder(),
                            stype.categorical: EmbeddingEncoder(),
                            stype.timestamp: TimestampEncoder(),
                        },
                    )
                    self.encoders[node_type] = encoder
                except (ImportError, Exception) as e:
                    logger.warning(f"torch_frame encoder failed for {node_type}: {e}")
                    num_nodes = data[node_type].num_nodes
                    self.encoders[node_type] = nn.Embedding(num_nodes, channels)
            else:
                num_nodes = data[node_type].num_nodes
                self.encoders[node_type] = nn.Embedding(num_nodes, channels)

        # Temporal encoder
        self.time_encoder = nn.Linear(1, channels)

        # GNN
        self.gnn = HeteroGraphSAGECustom(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr_factory=aggr_factory,
            num_layers=num_layers,
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(channels // 2, out_channels),
        )

    def forward(self, batch, entity_table):
        x_dict = {}
        for node_type in self.node_types:
            if node_type not in batch.node_types:
                continue
            encoder = self.encoders[node_type]
            if hasattr(batch[node_type], "tf") and batch[node_type].tf is not None:
                try:
                    tf_data = batch[node_type].tf
                    for st_key in tf_data.feat_dict:
                        tf_data.feat_dict[st_key] = torch.nan_to_num(
                            tf_data.feat_dict[st_key], nan=0.0
                        )
                    x = encoder(tf_data)
                    if isinstance(x, tuple):
                        x = x[0]
                    if x.dim() == 3:
                        x = x.mean(dim=1)
                    x = torch.nan_to_num(x, nan=0.0)
                except Exception:
                    n = batch[node_type].num_nodes
                    x = torch.zeros(n, self.channels, device=DEVICE)
            elif isinstance(encoder, nn.Embedding):
                n_id = batch[node_type].n_id
                x = encoder(n_id.clamp(0, encoder.num_embeddings - 1))
            else:
                n = batch[node_type].num_nodes
                x = torch.zeros(n, self.channels, device=DEVICE)

            # Add temporal encoding
            if hasattr(batch[node_type], "time") and batch[node_type].time is not None:
                t = batch[node_type].time.float().unsqueeze(-1)
                if t.shape[0] == x.shape[0]:
                    x = x + self.time_encoder(t)

            x_dict[node_type] = x

        # Run GNN
        edge_index_dict = {}
        for edge_type in self.edge_types:
            if edge_type in batch.edge_types:
                edge_index_dict[edge_type] = batch[edge_type].edge_index

        if edge_index_dict:
            x_dict = self.gnn(x_dict, edge_index_dict)

        x_entity = x_dict.get(entity_table)
        if x_entity is None:
            raise ValueError(f"Entity table '{entity_table}' not in batch")

        batch_size = batch[entity_table].batch_size
        x_entity = x_entity[:batch_size]
        return self.head(x_entity)


# ===================================================================
# PART 3: DATASET DISCOVERY & LOADING
# ===================================================================
def discover_dataset_and_tasks():
    """Try rel-trial first, fall back to rel-ratebeer, then rel-f1."""
    global DATASET_NAME, TASK_CONFIGS, BASE_HPARAMS
    global CHANNELS, NUM_LAYERS, BATCH_SIZE, NUM_NEIGHBORS_PER_LAYER, LR, EPOCHS

    os.environ["RELBENCH_CACHE_DIR"] = str(CACHE_DIR / "relbench_data")

    # --- Try rel-trial (primary) ---
    logger.info("Trying rel-trial dataset...")
    try:
        trial_dataset = get_dataset("rel-trial", download=True)
        logger.info(f"rel-trial download OK")

        # Try loading tasks
        task_a = get_task("rel-trial", "study-outcome", download=True)
        task_b = get_task("rel-trial", "study-adverse", download=True)

        DATASET_NAME = "rel-trial"
        TASK_CONFIGS = [
            {
                "task_name": "study-outcome",
                "task_type": "classification",
                "primary_metric": "average_precision",
                "direction": "higher_better",
                "task_obj": task_a,
            },
            {
                "task_name": "study-adverse",
                "task_type": "regression",
                "primary_metric": "mae",
                "direction": "lower_better",
                "task_obj": task_b,
            },
        ]
        # rel-trial special hyperparameters (RelBench paper Table 9)
        BASE_HPARAMS = {
            "lr": 0.0001,
            "epochs": 20,
            "batch_size": 512,
            "channels": 128,
            "num_layers": 2,
            "num_neighbors": 64,
        }
        CHANNELS = 128
        NUM_LAYERS = 2
        BATCH_SIZE = 512
        NUM_NEIGHBORS_PER_LAYER = 64
        LR = 0.0001
        EPOCHS = 20
        logger.info(f"Using rel-trial with tasks: study-outcome, study-adverse")
        return trial_dataset

    except Exception as e:
        logger.warning(f"rel-trial FAILED: {e}")

    # --- Try rel-ratebeer (backup 1) ---
    logger.info("Trying rel-ratebeer dataset...")
    try:
        ratebeer_dataset = get_dataset("rel-ratebeer", download=True)
        logger.info(f"rel-ratebeer download OK")

        task_a = get_task("rel-ratebeer", "user-churn", download=True)
        task_b = get_task("rel-ratebeer", "user-count", download=True)

        DATASET_NAME = "rel-ratebeer"
        TASK_CONFIGS = [
            {
                "task_name": "user-churn",
                "task_type": "classification",
                "primary_metric": "average_precision",
                "direction": "higher_better",
                "task_obj": task_a,
            },
            {
                "task_name": "user-count",
                "task_type": "regression",
                "primary_metric": "mae",
                "direction": "lower_better",
                "task_obj": task_b,
            },
        ]
        BASE_HPARAMS = {
            "lr": 0.005,
            "epochs": 10,
            "batch_size": 512,
            "channels": 128,
            "num_layers": 2,
            "num_neighbors": 128,
        }
        CHANNELS = 128
        NUM_LAYERS = 2
        BATCH_SIZE = 512
        NUM_NEIGHBORS_PER_LAYER = 128
        LR = 0.005
        EPOCHS = 10
        logger.info(f"Using rel-ratebeer with tasks: user-churn, user-count")
        return ratebeer_dataset

    except Exception as e:
        logger.warning(f"rel-ratebeer FAILED: {e}")

    # --- Fallback: rel-f1 (always available) ---
    logger.info("Falling back to rel-f1 dataset...")
    try:
        f1_dataset = get_dataset("rel-f1", download=True)
        task_a = get_task("rel-f1", "driver-dnf", download=True)
        task_b = get_task("rel-f1", "driver-position", download=True)

        DATASET_NAME = "rel-f1"
        TASK_CONFIGS = [
            {
                "task_name": "driver-dnf",
                "task_type": "classification",
                "primary_metric": "average_precision",
                "direction": "higher_better",
                "task_obj": task_a,
            },
            {
                "task_name": "driver-position",
                "task_type": "regression",
                "primary_metric": "mae",
                "direction": "lower_better",
                "task_obj": task_b,
            },
        ]
        BASE_HPARAMS = {
            "lr": 0.005,
            "epochs": 10,
            "batch_size": 512,
            "channels": 128,
            "num_layers": 2,
            "num_neighbors": 128,
        }
        CHANNELS = 128
        NUM_LAYERS = 2
        BATCH_SIZE = 512
        NUM_NEIGHBORS_PER_LAYER = 128
        LR = 0.005
        EPOCHS = 10
        logger.info(f"Using rel-f1 with tasks: driver-dnf, driver-position")
        return f1_dataset

    except Exception as e:
        logger.exception(f"rel-f1 also FAILED: {e}")
        raise RuntimeError("No dataset downloadable. Abort.")


def load_graph(dataset):
    """Build heterogeneous graph from dataset."""
    logger.info(f"Building heterogeneous graph for {DATASET_NAME}...")

    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)

    # Remove text columns for speed
    try:
        from torch_frame import stype as tf_stype
        for table_name in list(col_to_stype_dict.keys()):
            cols_to_remove = []
            for col, st in col_to_stype_dict[table_name].items():
                if st == tf_stype.text_embedded or \
                   (isinstance(st, str) and "text" in str(st).lower()):
                    cols_to_remove.append(col)
            for col in cols_to_remove:
                del col_to_stype_dict[table_name][col]
                logger.info(f"  Removed text column {table_name}.{col}")
    except Exception:
        pass

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,
        cache_dir=str(CACHE_DIR / DATASET_NAME / "materialized"),
    )

    logger.info(f"Graph built: node_types={data.node_types}")
    for nt in data.node_types:
        logger.info(f"  Node '{nt}': {data[nt].num_nodes} nodes")
    for et in data.edge_types:
        logger.info(f"  Edge {et}: {data[et].num_edges} edges")

    return data, col_stats_dict


def print_task_splits():
    """Print split sizes and target distributions for all tasks."""
    for tc in TASK_CONFIGS:
        task = tc["task_obj"]
        task_name = tc["task_name"]
        logger.info(f"Task: {DATASET_NAME}/{task_name} ({tc['task_type']})")
        for split_name in ["train", "val", "test"]:
            try:
                table = task.get_table(split_name)
                df = table.df
                n = len(df)
                target_col = task.target_col
                if target_col in df.columns:
                    target_vals = df[target_col].dropna()
                    logger.info(f"  {split_name}: n={n}, target mean={target_vals.mean():.4f}, "
                                f"std={target_vals.std():.4f}, min={target_vals.min():.4f}, "
                                f"max={target_vals.max():.4f}")
                else:
                    logger.info(f"  {split_name}: n={n} (no target column '{target_col}')")
            except Exception as e:
                logger.warning(f"  {split_name}: failed to load - {e}")


# ===================================================================
# PART 4: TRAINING
# ===================================================================
def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one(config_name: str, seed: int, task_config: dict,
              data, col_stats_dict, epochs: int, lr: float,
              batch_size: int, num_neighbors: list) -> dict:
    """Train one model and return results dict."""
    t0 = time.time()
    task = task_config["task_obj"]
    task_name = task_config["task_name"]
    task_type = task_config["task_type"]
    primary_metric = task_config["primary_metric"]
    direction = task_config["direction"]
    entity_table = task.entity_table

    seed_everything(seed)
    logger.info(f"Training {DATASET_NAME}/{task_name}/{config_name} seed={seed}...")

    # Create aggregation factory
    def aggr_factory(ch):
        return make_aggregation(config_name, ch)

    # Determine out_channels
    if task_type == "classification":
        out_channels = 1  # binary classification with BCEWithLogitsLoss
    else:
        out_channels = 1  # regression

    # Build model
    model = EntityModel(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=NUM_LAYERS,
        channels=CHANNELS,
        out_channels=out_channels,
        aggr_factory=aggr_factory,
        norm="layer_norm",
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model params: {num_params:,}")

    # Create data loaders
    train_table = task.get_table("train")
    train_input = get_node_train_table_input(table=train_table, task=task)
    val_table = task.get_table("val")
    val_input = get_node_train_table_input(table=val_table, task=task)

    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=train_input.nodes,
        input_time=train_input.time,
        transform=train_input.transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    train_target = train_input.target.float()

    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=val_input.nodes,
        input_time=val_input.time,
        transform=val_input.transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_target = val_input.target.float()

    # Test loader
    try:
        test_table = task.get_table("test")
        test_input = get_node_train_table_input(table=test_table, task=task)
        test_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            time_attr="time",
            input_nodes=test_input.nodes,
            input_time=test_input.time,
            transform=test_input.transform,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_target = test_input.target.float()
        has_test = True
    except Exception as e:
        logger.warning(f"  No test split available: {e}")
        test_loader = val_loader
        test_target = val_target
        has_test = False

    # Loss function
    if task_type == "classification":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.L1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Tracking
    if direction == "higher_better":
        best_val_metric = float('-inf')
    else:
        best_val_metric = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    train_losses = []
    gate_history = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            bs = batch[entity_table].batch_size
            input_id = batch[entity_table].input_id[:bs].cpu()
            target = train_target[input_id].to(DEVICE)

            pred = model(batch, entity_table).squeeze(-1)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # --- Validate ---
        model.eval()
        val_preds = []
        val_targets_list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                bs = batch[entity_table].batch_size
                input_id = batch[entity_table].input_id[:bs].cpu()
                target = val_target[input_id]
                pred = model(batch, entity_table).squeeze(-1)
                val_preds.append(pred.cpu())
                val_targets_list.append(target)

        val_pred_cat = torch.cat(val_preds)
        val_target_cat = torch.cat(val_targets_list)

        # Compute validation metrics
        val_metrics = compute_metrics(val_pred_cat.numpy(), val_target_cat.numpy(),
                                      task_type)

        current_metric = val_metrics.get(primary_metric, float('nan'))
        is_better = (
            current_metric > best_val_metric if direction == "higher_better"
            else current_metric < best_val_metric
        )
        if is_better and not math.isnan(current_metric):
            best_val_metric = current_metric
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        # Collect gate stats for RAMA
        if config_name == "rama":
            gate_stats = _collect_rama_gate_stats(model)
            gate_history.append(gate_stats)

        if epoch % 5 == 0 or epoch <= 3:
            gate_info = ""
            if config_name == "rama" and gate_history:
                gs = gate_history[-1]
                gate_info = f" gate={gs.get('mean_gate', 0):.4f} rho={gs.get('mean_rho', 0):.4f}"
            logger.info(f"  ep{epoch}/{epochs}: loss={avg_loss:.4f} "
                        f"val_{primary_metric}={current_metric:.4f} "
                        f"best={best_val_metric:.4f}{gate_info}")

    # --- Test with best model ---
    model.load_state_dict(best_model_state)
    model.eval()

    test_preds = []
    test_targets_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            bs = batch[entity_table].batch_size
            input_id = batch[entity_table].input_id[:bs].cpu()
            target = test_target[input_id]
            pred = model(batch, entity_table).squeeze(-1)
            test_preds.append(pred.cpu())
            test_targets_list.append(target)

    test_pred_cat = torch.cat(test_preds)
    test_target_cat = torch.cat(test_targets_list)
    test_metrics = compute_metrics(test_pred_cat.numpy(), test_target_cat.numpy(),
                                   task_type)

    elapsed = time.time() - t0
    logger.info(f"  DONE {task_name}/{config_name}/seed{seed}: "
                f"test_{primary_metric}={test_metrics.get(primary_metric, float('nan')):.4f} "
                f"(best_epoch={best_epoch}, {elapsed:.0f}s)")

    # Collect final RAMA diagnostics
    diagnostics = {}
    if config_name == "rama":
        diagnostics = _collect_rama_gate_stats(model)

    result = {
        "task_name": task_name,
        "config": config_name,
        "seed": seed,
        "dataset": DATASET_NAME,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "direction": direction,
        "best_val_metric": float(best_val_metric),
        "best_epoch": best_epoch,
        "val_metrics": {k: float(v) for k, v in compute_metrics(
            val_pred_cat.numpy(), val_target_cat.numpy(), task_type
        ).items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "num_params": num_params,
        "runtime_sec": elapsed,
        "train_losses": train_losses,
        "gate_history": gate_history if config_name == "rama" else None,
        "diagnostics": diagnostics,
        "has_test_split": has_test,
    }

    # Save per-run result
    result_path = RESULTS_DIR / f"{task_name}_{config_name}_seed{seed}.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))

    # Cleanup
    del model, optimizer
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

    return result


def compute_metrics(pred: np.ndarray, target: np.ndarray, task_type: str) -> dict:
    """Compute evaluation metrics for predictions."""
    metrics = {}
    if task_type == "regression":
        metrics["mae"] = float(np.mean(np.abs(pred - target)))
        metrics["rmse"] = float(np.sqrt(np.mean((pred - target) ** 2)))
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        metrics["r2"] = float(1 - ss_res / max(ss_tot, 1e-12))
    elif task_type == "classification":
        from sklearn.metrics import roc_auc_score, average_precision_score
        # Apply sigmoid for probability
        prob = 1 / (1 + np.exp(-np.clip(pred, -50, 50)))
        try:
            metrics["auroc"] = float(roc_auc_score(target, prob))
        except ValueError:
            metrics["auroc"] = 0.5
        try:
            metrics["average_precision"] = float(average_precision_score(target, prob))
        except ValueError:
            metrics["average_precision"] = 0.0
    return metrics


def _collect_rama_gate_stats(model: EntityModel) -> dict:
    """Collect gate and rank diagnostics from RAMA modules in the model."""
    gate_vals = []
    rho_vals = []
    n_vals = []
    for module in model.modules():
        if isinstance(module, RAMAAggregation):
            if module._last_gate_mean is not None:
                gate_vals.append(module._last_gate_mean)
            if module._last_rho_mean is not None:
                rho_vals.append(module._last_rho_mean)
            if module._last_N_mean is not None:
                n_vals.append(module._last_N_mean)
    return {
        "mean_gate": float(np.mean(gate_vals)) if gate_vals else None,
        "mean_rho": float(np.mean(rho_vals)) if rho_vals else None,
        "mean_N": float(np.mean(n_vals)) if n_vals else None,
    }


# ===================================================================
# PART 5: POST-VOTES DIAGNOSIS
# ===================================================================
def run_post_votes_diagnosis() -> dict:
    """Diagnose the null result on rel-stack/post-votes."""
    logger.info("=" * 60)
    logger.info("POST-VOTES DIAGNOSIS")
    logger.info("=" * 60)

    diagnosis = {}
    try:
        stack_dataset = get_dataset("rel-stack", download=True)
        pv_task = get_task("rel-stack", "post-votes", download=True)

        train_df = pv_task.get_table("train").df
        val_df = pv_task.get_table("val").df
        test_df = pv_task.get_table("test").df

        target_col = pv_task.target_col

        diagnosis = {
            "train_target_mean": float(train_df[target_col].mean()),
            "train_target_std": float(train_df[target_col].std()),
            "train_target_median": float(train_df[target_col].median()),
            "val_target_mean": float(val_df[target_col].mean()),
            "val_target_std": float(val_df[target_col].std()),
            "val_target_median": float(val_df[target_col].median()),
            "test_target_mean": float(test_df[target_col].mean()),
            "test_target_std": float(test_df[target_col].std()),
            # Distribution shift
            "train_val_mean_shift": abs(float(train_df[target_col].mean() - val_df[target_col].mean())),
            "train_val_std_ratio": float(val_df[target_col].std() / max(train_df[target_col].std(), 1e-12)),
            # Constant baseline MAE
            "constant_baseline_mae_on_val": float((val_df[target_col] - train_df[target_col].mean()).abs().mean()),
            "constant_baseline_mae_on_test": float((test_df[target_col] - train_df[target_col].mean()).abs().mean()),
            # Zero-variance check
            "train_target_nunique": int(train_df[target_col].nunique()),
            "val_target_nunique": int(val_df[target_col].nunique()),
            # Fraction near zero
            "train_frac_zero": float((train_df[target_col] == 0).mean()),
            "val_frac_zero": float((val_df[target_col] == 0).mean()),
            # Skewness
            "train_skewness": float(train_df[target_col].skew()),
            "val_skewness": float(val_df[target_col].skew()),
        }
        logger.info(f"Post-votes diagnosis:")
        for k, v in diagnosis.items():
            logger.info(f"  {k}: {v}")

    except Exception as e:
        logger.warning(f"Post-votes diagnosis failed: {e}")
        diagnosis = {"error": str(e)}

    return diagnosis


# ===================================================================
# PART 6: STATISTICAL ANALYSIS
# ===================================================================
def cohens_d_paired(group1_values: list, group2_values: list,
                    higher_better: bool = True) -> dict:
    """Compute Cohen's d with 95% CI for paired samples.

    Positive d always means RAMA is better.
    For higher_better metrics: d = (group2 - group1) / sd_diff
    For lower_better metrics: d = (group1 - group2) / sd_diff
    """
    a1 = np.array(group1_values)
    a2 = np.array(group2_values)
    n = len(a1)

    if n < 2:
        return {"d": 0.0, "ci_low": float("-inf"), "ci_high": float("inf"),
                "p_value": 1.0, "n_seeds": n}

    if higher_better:
        diffs = a2 - a1  # positive = RAMA better
    else:
        diffs = a1 - a2  # positive = RAMA better (lower metric is better)

    sd_diff = float(diffs.std(ddof=1))
    if sd_diff < 1e-12:
        d = 0.0
    else:
        d = float(diffs.mean() / sd_diff)

    se_d = math.sqrt((1 / n) + (d ** 2 / (2 * n)))
    ci_low = d - 1.96 * se_d
    ci_high = d + 1.96 * se_d

    # Paired t-test
    if sd_diff > 1e-12:
        t_stat, p_val = scipy_stats.ttest_rel(a2, a1)
        p_val = float(p_val)
    else:
        p_val = 1.0

    return {
        "d": round(d, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "p_value": round(p_val, 6),
        "n_seeds": n,
    }


def build_statistical_analysis(all_results: list) -> dict:
    """Build complete statistical analysis from all results."""
    # Per-task effect sizes
    task_effects = {}
    for tc in TASK_CONFIGS:
        task_name = tc["task_name"]
        metric = tc["primary_metric"]
        higher_better = tc["direction"] == "higher_better"

        baseline_vals = []
        rama_vals = []
        for r in all_results:
            if r["task_name"] == task_name:
                val = r["test_metrics"].get(metric)
                if val is not None and not math.isnan(val):
                    if r["config"] == "baseline_mean":
                        baseline_vals.append(val)
                    elif r["config"] == "rama":
                        rama_vals.append(val)

        if len(baseline_vals) >= 2 and len(rama_vals) >= 2:
            effect = cohens_d_paired(baseline_vals, rama_vals, higher_better)
        else:
            effect = {"d": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0,
                      "n_seeds": min(len(baseline_vals), len(rama_vals))}

        task_effects[task_name] = {
            "baseline_mean": round(float(np.mean(baseline_vals)), 6) if baseline_vals else None,
            "baseline_std": round(float(np.std(baseline_vals, ddof=1)), 6) if len(baseline_vals) > 1 else 0.0,
            "rama_mean": round(float(np.mean(rama_vals)), 6) if rama_vals else None,
            "rama_std": round(float(np.std(rama_vals, ddof=1)), 6) if len(rama_vals) > 1 else 0.0,
            "cohens_d": effect["d"],
            "ci_95": [effect["ci_low"], effect["ci_high"]],
            "p_value": effect["p_value"],
            "task_type": tc["task_type"],
            "metric": metric,
            "direction": tc["direction"],
            "per_seed_baseline": [round(v, 6) for v in baseline_vals],
            "per_seed_rama": [round(v, 6) for v in rama_vals],
        }

    # Gate behavior analysis
    gate_analysis = {}
    for tc in TASK_CONFIGS:
        task_name = tc["task_name"]
        final_gates = []
        final_rhos = []
        for r in all_results:
            if r["task_name"] == task_name and r["config"] == "rama":
                gh = r.get("gate_history")
                if gh and len(gh) > 0:
                    last = gh[-1]
                    if last.get("mean_gate") is not None:
                        final_gates.append(last["mean_gate"])
                    if last.get("mean_rho") is not None:
                        final_rhos.append(last["mean_rho"])
        gate_analysis[task_name] = {
            "mean_gate_final_epoch": round(float(np.mean(final_gates)), 4) if final_gates else None,
            "mean_rho_final_epoch": round(float(np.mean(final_rhos)), 4) if final_rhos else None,
        }

    # Per-config summary
    per_config = {}
    for config_name in CONFIGS:
        for tc in TASK_CONFIGS:
            task_name = tc["task_name"]
            metric = tc["primary_metric"]
            key = f"{task_name}/{config_name}"
            vals = [r["test_metrics"].get(metric) for r in all_results
                    if r["task_name"] == task_name and r["config"] == config_name
                    and r["test_metrics"].get(metric) is not None]
            if vals:
                per_config[key] = {
                    "mean": round(float(np.mean(vals)), 6),
                    "std": round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else 0.0,
                    "n_seeds": len(vals),
                    "per_seed": [round(v, 6) for v in vals],
                }

    # Ranking per task
    rankings = {}
    for tc in TASK_CONFIGS:
        task_name = tc["task_name"]
        metric = tc["primary_metric"]
        direction = tc["direction"]
        task_ranking = []
        for config_name in CONFIGS:
            key = f"{task_name}/{config_name}"
            if key in per_config:
                task_ranking.append({
                    "config": config_name,
                    "mean": per_config[key]["mean"],
                    "std": per_config[key]["std"],
                })
        reverse = (direction == "higher_better")
        task_ranking.sort(key=lambda x: x["mean"], reverse=reverse)
        for i, entry in enumerate(task_ranking):
            entry["rank"] = i + 1
        rankings[task_name] = task_ranking

    return {
        "per_task_results": task_effects,
        "gate_analysis": gate_analysis,
        "per_config_results": per_config,
        "rankings": rankings,
    }


# ===================================================================
# PART 7: OUTPUT GENERATION
# ===================================================================
def load_dependency_examples() -> list:
    """Load examples from ALL dependency datasets."""
    all_datasets = []
    dep3_path = DEP_DATA3 / "full_data_out.json"
    if dep3_path.exists():
        logger.info(f"Loading dependency data from {dep3_path.name}...")
        try:
            dep3 = json.loads(dep3_path.read_text())
            for ds_entry in dep3.get("datasets", []):
                all_datasets.append(ds_entry)
            logger.info(f"  Loaded {len(dep3.get('datasets', []))} datasets from data_id3")
        except Exception:
            logger.exception("Failed to load data_id3")
    else:
        logger.warning(f"Dependency file not found: {dep3_path}")
    return all_datasets


def generate_output(all_results: list, stats: dict,
                    post_votes_diagnosis: dict) -> dict:
    """Generate method_out.json in exp_gen_sol_out format."""
    logger.info("Generating method_out.json...")

    dep_datasets = load_dependency_examples()

    # Build output datasets with predictions
    output_datasets = []
    tested_task_prefixes = [
        f"{DATASET_NAME}/{tc['task_name']}" for tc in TASK_CONFIGS
    ]

    for ds_entry in dep_datasets:
        ds_name = ds_entry["dataset"]
        examples = ds_entry.get("examples", [])

        clean_examples = []
        for ex in examples:
            clean_ex = {
                "input": ex.get("input", ""),
                "output": ex.get("output", ""),
            }
            for k, v in ex.items():
                if k.startswith("metadata_"):
                    clean_ex[k] = v
            # Add predict_* fields
            for config_name in CONFIGS:
                key_match = None
                for tc in TASK_CONFIGS:
                    prefix = f"{DATASET_NAME}/{tc['task_name']}"
                    if ds_name.startswith(prefix) or ds_name == prefix:
                        key_match = f"{tc['task_name']}/{config_name}"
                        break

                per_config = stats.get("per_config_results", {})
                if key_match and key_match in per_config:
                    info = per_config[key_match]
                    clean_ex[f"predict_{config_name}"] = (
                        f"mean={info['mean']:.6f}, std={info['std']:.6f}, n={info['n_seeds']}"
                    )
                else:
                    clean_ex[f"predict_{config_name}"] = "not_evaluated_on_this_task"

            clean_examples.append(clean_ex)
        output_datasets.append({"dataset": ds_name, "examples": clean_examples})

    # If no dep datasets loaded, create a synthetic one with our results
    if not output_datasets:
        for tc in TASK_CONFIGS:
            task_name = tc["task_name"]
            metric = tc["primary_metric"]
            examples = []
            for config_name in CONFIGS:
                per_seed_vals = stats["per_task_results"].get(task_name, {}).get(
                    f"per_seed_{'baseline' if config_name == 'baseline_mean' else 'rama'}", []
                )
                for i, val in enumerate(per_seed_vals):
                    examples.append({
                        "input": json.dumps({
                            "dataset": DATASET_NAME,
                            "task": task_name,
                            "config": config_name,
                            "seed": SEEDS[i] if i < len(SEEDS) else i,
                        }),
                        "output": str(val),
                        "metadata_task_type": tc["task_type"],
                        "metadata_metric": metric,
                        "predict_baseline_mean": str(
                            stats["per_task_results"].get(task_name, {}).get("baseline_mean", "N/A")
                        ),
                        "predict_rama": str(
                            stats["per_task_results"].get(task_name, {}).get("rama_mean", "N/A")
                        ),
                    })
            if examples:
                output_datasets.append({
                    "dataset": f"{DATASET_NAME}/{task_name}",
                    "examples": examples,
                })

    # Build metadata
    metadata = {
        "experiment": f"RAMA Evaluation on New RelBench Tasks ({DATASET_NAME})",
        "description": (
            f"Evaluation of RAMA (Rank-Aware Moment Aggregation) on 2 new entity tasks "
            f"from {DATASET_NAME}: {', '.join(tc['task_name'] for tc in TASK_CONFIGS)}. "
            f"Compares baseline_mean vs RAMA across {len(SEEDS)} seeds per config. "
            f"Includes post-votes diagnosis as secondary objective."
        ),
        "methods": CONFIGS,
        "hyperparameters": BASE_HPARAMS,
        "dataset_info": {
            "dataset": DATASET_NAME,
            "tasks": [tc["task_name"] for tc in TASK_CONFIGS],
            "task_types": [tc["task_type"] for tc in TASK_CONFIGS],
            "primary_metrics": [tc["primary_metric"] for tc in TASK_CONFIGS],
        },
        "seeds": SEEDS,
        "statistical_analysis": stats,
        "post_votes_diagnosis": post_votes_diagnosis,
        "summary": {
            "num_new_tasks": len(TASK_CONFIGS),
            "new_effect_sizes_for_meta_analysis": [
                {
                    "task": tn,
                    "d": te["cohens_d"],
                    "ci": te["ci_95"],
                    "p": te["p_value"],
                }
                for tn, te in stats["per_task_results"].items()
            ],
            "any_catastrophic_failure": any(
                abs(te["cohens_d"]) > 5 for te in stats["per_task_results"].values()
            ),
        },
    }

    output = {
        "metadata": metadata,
        "datasets": output_datasets,
    }
    return output


# ===================================================================
# MAIN
# ===================================================================
@logger.catch
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("RAMA Evaluation on New RelBench Tasks")
    logger.info("=" * 60)

    # ---- Phase 0: Unit tests ----
    run_unit_tests()

    # ---- Phase 1: Dataset discovery ----
    logger.info("=" * 60)
    logger.info("PHASE 1: Dataset Discovery & Validation")
    logger.info("=" * 60)

    dataset = discover_dataset_and_tasks()
    print_task_splits()

    # Build graph
    data, col_stats_dict = load_graph(dataset)

    # ---- Phase 2: Mini validation runs ----
    logger.info("=" * 60)
    logger.info("PHASE 2: Mini Validation Runs (smoke test)")
    logger.info("=" * 60)

    num_neighbors = [NUM_NEIGHBORS_PER_LAYER] * NUM_LAYERS
    smoke_results = []
    mini_epochs = 3

    for tc in TASK_CONFIGS:
        for config_name in CONFIGS:
            try:
                r = train_one(
                    config_name=config_name,
                    seed=42,
                    task_config=tc,
                    data=data,
                    col_stats_dict=col_stats_dict,
                    epochs=mini_epochs,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    num_neighbors=num_neighbors,
                )
                smoke_results.append(r)

                metric_val = r["test_metrics"].get(tc["primary_metric"])
                logger.info(f"  Smoke {tc['task_name']}/{config_name}: "
                            f"test_{tc['primary_metric']}={metric_val:.4f}, "
                            f"time={r['runtime_sec']:.1f}s")

                # Sanity checks
                if metric_val is None or math.isnan(metric_val):
                    logger.error(f"  NaN metric for {tc['task_name']}/{config_name}")
                if tc["task_type"] == "classification" and metric_val < 0.3:
                    logger.warning(f"  Low AUROC/AP for classification: {metric_val:.4f}")

            except Exception:
                logger.exception(f"Smoke test failed: {tc['task_name']}/{config_name}")

    if not smoke_results:
        logger.error("All smoke tests failed. Aborting.")
        return

    # Estimate runtime for full experiment
    avg_smoke_time = np.mean([r["runtime_sec"] for r in smoke_results])
    est_per_run = avg_smoke_time * (EPOCHS / mini_epochs)
    n_total_runs = len(TASK_CONFIGS) * len(CONFIGS) * len(SEEDS)
    est_total_min = est_per_run * n_total_runs / 60
    logger.info(f"Runtime estimate: {est_per_run:.0f}s/run x {n_total_runs} runs = "
                f"{est_total_min:.0f}min total")

    # Adjust parameters if too slow
    epochs_to_use = EPOCHS
    seeds_to_use = list(SEEDS)
    elapsed_so_far = (time.time() - start_time) / 60
    remaining_minutes = 340 - elapsed_so_far  # Conservative

    if est_total_min > remaining_minutes * 0.7:
        # Reduce epochs
        epochs_to_use = max(EPOCHS // 2, 5)
        est_total_min = est_per_run * (epochs_to_use / EPOCHS) * len(TASK_CONFIGS) * len(CONFIGS) * len(seeds_to_use) / 60
        logger.info(f"Reduced epochs to {epochs_to_use}, est={est_total_min:.0f}min")

    if est_total_min > remaining_minutes * 0.7:
        # Reduce seeds
        seeds_to_use = SEEDS[:3]
        est_total_min = est_per_run * (epochs_to_use / EPOCHS) * len(TASK_CONFIGS) * len(CONFIGS) * len(seeds_to_use) / 60
        logger.info(f"Reduced seeds to {len(seeds_to_use)}, est={est_total_min:.0f}min")

    # ---- Phase 3: Full experiments ----
    logger.info("=" * 60)
    logger.info(f"PHASE 3: Full Experiments ({len(TASK_CONFIGS)} tasks x "
                f"{len(CONFIGS)} configs x {len(seeds_to_use)} seeds x "
                f"{epochs_to_use} epochs)")
    logger.info("=" * 60)

    all_results = []
    for tc in TASK_CONFIGS:
        task_name = tc["task_name"]
        for config_name in CONFIGS:
            config_start = time.time()
            for seed in seeds_to_use:
                try:
                    r = train_one(
                        config_name=config_name,
                        seed=seed,
                        task_config=tc,
                        data=data,
                        col_stats_dict=col_stats_dict,
                        epochs=epochs_to_use,
                        lr=LR,
                        batch_size=BATCH_SIZE,
                        num_neighbors=num_neighbors,
                    )
                    all_results.append(r)
                except Exception:
                    logger.exception(f"Failed: {task_name}/{config_name}/seed{seed}")
                    continue

            config_elapsed = time.time() - config_start
            config_vals = [
                r["test_metrics"].get(tc["primary_metric"])
                for r in all_results
                if r["task_name"] == task_name and r["config"] == config_name
            ]
            if config_vals:
                logger.info(f"  {task_name}/{config_name}: "
                            f"mean={np.mean(config_vals):.4f} +/- "
                            f"{np.std(config_vals):.4f} ({config_elapsed:.0f}s)")

    if not all_results:
        logger.error("No successful runs. Aborting.")
        return

    # ---- Phase 4: Post-votes diagnosis ----
    logger.info("=" * 60)
    logger.info("PHASE 4: Post-Votes Diagnosis")
    logger.info("=" * 60)

    post_votes_diagnosis = run_post_votes_diagnosis()

    # ---- Phase 5: Statistical analysis ----
    logger.info("=" * 60)
    logger.info("PHASE 5: Statistical Analysis")
    logger.info("=" * 60)

    stats = build_statistical_analysis(all_results)

    # Log key results
    logger.info("Per-task effect sizes (positive d = RAMA better):")
    for task_name, effect in stats["per_task_results"].items():
        logger.info(f"  {task_name}: d={effect['cohens_d']:.4f} "
                    f"CI=[{effect['ci_95'][0]:.4f}, {effect['ci_95'][1]:.4f}] "
                    f"p={effect['p_value']:.4f}")
        logger.info(f"    baseline={effect['baseline_mean']:.6f}+/-{effect['baseline_std']:.6f}, "
                    f"rama={effect['rama_mean']:.6f}+/-{effect['rama_std']:.6f}")

    logger.info("Rankings per task:")
    for task_name, ranking in stats["rankings"].items():
        for entry in ranking:
            logger.info(f"  {task_name} #{entry['rank']}: {entry['config']} = "
                        f"{entry['mean']:.6f} +/- {entry['std']:.6f}")

    if stats["gate_analysis"]:
        logger.info("Gate analysis:")
        for task_name, ga in stats["gate_analysis"].items():
            logger.info(f"  {task_name}: gate={ga['mean_gate_final_epoch']}, "
                        f"rho={ga['mean_rho_final_epoch']}")

    # ---- Phase 6: Generate output ----
    logger.info("=" * 60)
    logger.info("PHASE 6: Generate Output")
    logger.info("=" * 60)

    output = generate_output(all_results, stats, post_votes_diagnosis)

    out_path = WS / "method_out.json"
    out_text = json.dumps(output, indent=2, default=str)
    out_path.write_text(out_text)
    file_size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved method_out.json ({file_size_mb:.1f} MB)")

    total_elapsed = time.time() - start_time
    logger.info(f"Total experiment time: {total_elapsed / 60:.1f} minutes")
    logger.info("EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
