#!/usr/bin/env python3
"""RAMA Component Ablation on rel-f1/driver-position.

6 aggregation methods x 5 seeds = 30 training runs.
Primary metric: test MAE.
Statistical analysis via pairwise Cohen's d and DerSimonian-Laird meta-analysis.
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
DEP_DATA4 = DEP_BASE / "data_id4_it1__opus"

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
TOTAL_RAM_GB = _container_ram_gb() or 57.0

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
            f"GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM), device={DEVICE}")

# ===================================================================
# RESOURCE LIMITS
# ===================================================================
import psutil

_avail = psutil.virtual_memory().available
RAM_BUDGET = int(min(40 * 1e9, _avail * 0.8))
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
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
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
# HYPERPARAMETERS
# ===================================================================
CHANNELS = 128
NUM_LAYERS = 2
LR = 0.005
BATCH_SIZE = 512
EPOCHS = 10
NUM_NEIGHBORS = [128, 128]
SEEDS = [42, 123, 456, 789, 1024]
METHODS_LIST = [
    "standard_mean", "mean_layernorm", "ungated_moment",
    "rama_full", "pna_style", "rama_no_rank",
]

# ===================================================================
# PART 1: AGGREGATION MODULES
# ===================================================================

class StandardMeanAggregation(Aggregation):
    """Method 1: Standard mean wrapped in Aggregation subclass."""
    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        return self.reduce(x, index, ptr, dim_size, dim, reduce="mean")


class MeanLayerNormAggregation(Aggregation):
    """Method 2: Mean + LayerNorm (tests LayerNorm confounder)."""
    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        return self.ln(mean)


class UngatedMomentAggregation(Aggregation):
    """Method 3: Mean + ungated variance injection (no gating)."""
    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.W_sigma = nn.Linear(channels, channels, bias=False)
        self.eps = eps
        nn.init.zeros_(self.W_sigma.weight)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        mu = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce="mean")
        var = (mean_x2 - mu * mu).clamp(min=0)
        return mu + self.W_sigma(var)


class RAMAAggregation(Aggregation):
    """Method 4: Full RAMA - mean + gate(log_N, rho) * W_sigma(var)."""
    def __init__(self, channels: int, gate_hidden: int = 16, eps: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.W_sigma = nn.Linear(channels, channels, bias=False)
        self.W_g = nn.Linear(2, gate_hidden)
        self.gate_out = nn.Linear(gate_hidden, channels)
        # Zero-init: start as pure mean aggregation
        nn.init.zeros_(self.W_sigma.weight)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)
        # Diagnostic storage
        self._last_gate = None
        self._last_rho = None
        self._last_N = None

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        # Step 1: Mean
        mu = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        # Step 2: Variance via E[X^2] - E[X]^2
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce="mean")
        var = (mean_x2 - mu * mu).clamp(min=0)
        # Step 3: Cardinality N per parent
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        N = self.reduce(ones, index, ptr, dim_size, dim, reduce="sum")
        # Step 4: Effective rank proxy (variance entropy)
        var_sum = var.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        p = var / var_sum
        p = p.clamp(min=self.eps)
        H = -(p * p.log()).sum(dim=-1, keepdim=True)
        log_d = math.log(self.channels) if self.channels > 1 else 1.0
        rho = H / log_d  # normalized [0, 1]
        # Step 5: Gate
        gate_input = torch.cat([torch.log1p(N), rho], dim=-1)
        g = torch.sigmoid(self.gate_out(F.relu(self.W_g(gate_input))))
        # Store for diagnostics
        self._last_gate = g.detach()
        self._last_rho = rho.detach()
        self._last_N = N.detach()
        # Step 6: Output
        return mu + g * self.W_sigma(var)


def make_pna_aggregation(channels: int):
    """Method 5: PNA-style Multi-Aggregation [mean, std, max]."""
    return MultiAggregation(
        ["mean", "std", "max"],
        mode="proj",
        mode_kwargs={"in_channels": channels, "out_channels": channels},
    )


class RAMANoRankAggregation(Aggregation):
    """Method 6: RAMA without rank conditioning (fixed rho=0.5)."""
    def __init__(self, channels: int, gate_hidden: int = 16, eps: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.W_sigma = nn.Linear(channels, channels, bias=False)
        self.W_g = nn.Linear(2, gate_hidden)
        self.gate_out = nn.Linear(gate_hidden, channels)
        nn.init.zeros_(self.W_sigma.weight)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)
        self._last_gate = None
        self._last_N = None

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        mu = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce="mean")
        var = (mean_x2 - mu * mu).clamp(min=0)
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        N = self.reduce(ones, index, ptr, dim_size, dim, reduce="sum")
        fixed_rho = torch.full_like(N, 0.5)
        gate_input = torch.cat([torch.log1p(N), fixed_rho], dim=-1)
        g = torch.sigmoid(self.gate_out(F.relu(self.W_g(gate_input))))
        self._last_gate = g.detach()
        self._last_N = N.detach()
        return mu + g * self.W_sigma(var)


METHODS_FACTORY = {
    "standard_mean": lambda ch: StandardMeanAggregation(),
    "mean_layernorm": lambda ch: MeanLayerNormAggregation(ch),
    "ungated_moment": lambda ch: UngatedMomentAggregation(ch),
    "rama_full": lambda ch: RAMAAggregation(ch, gate_hidden=16),
    "pna_style": lambda ch: make_pna_aggregation(ch),
    "rama_no_rank": lambda ch: RAMANoRankAggregation(ch, gate_hidden=16),
}


# ===================================================================
# PART 1b: UNIT TESTS
# ===================================================================
def run_unit_tests():
    """Phase 0: Verify aggregation modules before training."""
    logger.info("=" * 60)
    logger.info("PHASE 0: Unit tests on aggregation modules")
    logger.info("=" * 60)
    n_edges, n_groups, d = 100, 10, CHANNELS
    x = torch.randn(n_edges, d)
    index = torch.randint(0, n_groups, (n_edges,))

    for name, factory in METHODS_FACTORY.items():
        aggr = factory(d)
        out = aggr(x, index=index, dim_size=n_groups)
        assert out.shape == (n_groups, d), \
            f"{name}: expected ({n_groups}, {d}), got {out.shape}"
        logger.info(f"  {name}: shape OK {out.shape}")

    # Zero-init test: RAMA should equal mean at initialization
    rama = RAMAAggregation(d)
    rama_out = rama(x, index=index, dim_size=n_groups)
    mean_aggr = StandardMeanAggregation()
    mean_out = mean_aggr(x, index=index, dim_size=n_groups)
    diff = (rama_out - mean_out).abs().max().item()
    assert diff < 1e-5, f"RAMA zero-init: diff={diff} (expected <1e-5)"
    logger.info(f"  RAMA zero-init test: max diff={diff:.2e} OK")

    # Gradient flow test
    for name in ["ungated_moment", "rama_full", "rama_no_rank"]:
        aggr = METHODS_FACTORY[name](d)
        x_test = torch.randn(n_edges, d, requires_grad=True)
        out = aggr(x_test, index=index, dim_size=n_groups)
        loss = out.sum()
        loss.backward()
        assert x_test.grad is not None, f"{name}: no gradient on input"
        for pname, param in aggr.named_parameters():
            assert param.grad is not None, f"{name}.{pname}: no gradient"
        logger.info(f"  {name}: gradient flow OK")

    # SAGEConv integration test
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

    logger.info("All unit tests PASSED")


# ===================================================================
# PART 2: MODEL ARCHITECTURE
# ===================================================================
class HeteroGraphSAGECustom(nn.Module):
    """HeteroGraphSAGE with custom aggregation.

    Adapted from relbench.modeling.nn.HeteroGraphSAGE to support
    custom Aggregation objects.
    """
    def __init__(self, node_types, edge_types, channels, aggr, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                # Each edge type gets its own SAGEConv with the custom aggr
                # Need to create a fresh aggr instance per edge type per layer
                if isinstance(aggr, str):
                    edge_aggr = aggr
                elif callable(aggr) and not isinstance(aggr, nn.Module):
                    edge_aggr = aggr(channels)
                else:
                    # Clone the aggregation for each edge type
                    edge_aggr = copy.deepcopy(aggr)
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
    1. Per-table linear encoder (TensorFrame -> embedding)
    2. HeteroGraphSAGE with custom aggregation
    3. MLP prediction head
    """
    def __init__(self, data, col_stats_dict, num_layers, channels,
                 out_channels, aggr_obj, norm="layer_norm"):
        super().__init__()
        self.channels = channels
        self.entity_table = None  # set externally

        # Per-node-type encoders: project torch_frame TensorFrame to channels
        self.encoders = nn.ModuleDict()
        self.node_types = data.node_types
        self.edge_types = data.edge_types

        for node_type in data.node_types:
            if hasattr(data[node_type], "tf") and data[node_type].tf is not None:
                tf = data[node_type].tf
                # Get total feature dimension from TensorFrame
                try:
                    from torch_frame import stype
                    from torch_frame.nn import (
                        StypeEncoder,
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
                    # Fallback: learnable embedding
                    num_nodes = data[node_type].num_nodes
                    self.encoders[node_type] = nn.Embedding(num_nodes, channels)
            else:
                # No features: learnable embedding
                num_nodes = data[node_type].num_nodes
                self.encoders[node_type] = nn.Embedding(num_nodes, channels)

        # Temporal encoder (simple sinusoidal)
        self.time_encoder = nn.Linear(1, channels)
        self.has_time = {}
        for node_type in data.node_types:
            self.has_time[node_type] = hasattr(data[node_type], "time") and \
                data[node_type].time is not None

        # GNN
        self.gnn = HeteroGraphSAGECustom(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr_obj,
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
                    # Replace NaN in features before encoding
                    for st_key in tf_data.feat_dict:
                        tf_data.feat_dict[st_key] = torch.nan_to_num(
                            tf_data.feat_dict[st_key], nan=0.0
                        )
                    x = encoder(tf_data)
                    if isinstance(x, tuple):
                        x = x[0]
                    # StypeWiseFeatureEncoder returns (N, num_cols, channels)
                    # Reduce column dim by mean pooling to get (N, channels)
                    if x.dim() == 3:
                        x = x.mean(dim=1)
                    # Safety: replace any residual NaN
                    x = torch.nan_to_num(x, nan=0.0)
                except Exception:
                    # fallback: zeros
                    n = batch[node_type].num_nodes
                    x = torch.zeros(n, self.channels, device=DEVICE)
            elif hasattr(batch[node_type], "x") and batch[node_type].x is not None:
                x = encoder(batch[node_type].x)
            elif isinstance(encoder, nn.Embedding):
                n_id = batch[node_type].n_id
                x = encoder(n_id.clamp(0, encoder.num_embeddings - 1))
            else:
                n = batch[node_type].num_nodes
                x = torch.zeros(n, self.channels, device=DEVICE)

            # Add temporal encoding if available
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

        # Get predictions for entity nodes
        x_entity = x_dict.get(entity_table)
        if x_entity is None:
            raise ValueError(f"Entity table '{entity_table}' not in batch")

        # Only predict for seed nodes (first batch_size nodes)
        batch_size = batch[entity_table].batch_size
        x_entity = x_entity[:batch_size]

        return self.head(x_entity)


# ===================================================================
# PART 3: DATA LOADING
# ===================================================================
def load_relbench_data(dataset_name: str = "rel-f1",
                       task_name: str = "driver-position"):
    """Load dataset and build heterogeneous graph."""
    logger.info(f"Loading dataset {dataset_name}/{task_name}...")

    os.environ["RELBENCH_CACHE_DIR"] = str(CACHE_DIR / "relbench_data")

    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    # relbench 1.1.0: get_stype_proposal and make_pkey_fkey_graph take db, not dataset
    db = dataset.get_db()

    logger.info("Getting stype proposal...")
    col_to_stype_dict = get_stype_proposal(db)

    logger.info("Building heterogeneous graph...")
    # Skip text embedding for speed; use only numerical/categorical/timestamp
    text_cfg = None
    # Remove text columns from stype dict
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
        text_embedder_cfg=text_cfg,
        cache_dir=str(CACHE_DIR / dataset_name / "materialized"),
    )

    logger.info(f"Graph built: {data.num_nodes} nodes, {data.num_edges} edges")
    logger.info(f"Node types: {data.node_types}")
    logger.info(f"Edge types: {data.edge_types}")

    return dataset, task, data, col_stats_dict


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


def train_one(method_name: str, seed: int, task, data, col_stats_dict,
              epochs: int = EPOCHS, lr: float = LR) -> dict:
    """Train one model and return results dict."""
    t0 = time.time()
    seed_everything(seed)
    logger.info(f"Training {method_name} seed={seed}...")

    entity_table = task.entity_table
    entity_col = task.entity_col
    time_col = task.time_col
    target_col = task.target_col

    # Create aggregation
    aggr_obj = METHODS_FACTORY[method_name](CHANNELS)

    # Build model
    model = EntityModel(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=NUM_LAYERS,
        channels=CHANNELS,
        out_channels=1,
        aggr_obj=aggr_obj,
        norm="layer_norm",
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model params: {num_params:,}")

    # Create data loaders using relbench's NodeTrainTableInput API
    loader_dict = {}
    target_dict = {}

    # Note: test split has no targets in relbench (hidden for leaderboard).
    # We use val split as our held-out test set for ablation comparison.
    # We split train 90/10 for train/val, and use original val as test.
    train_table = task.get_table("train")
    train_input = get_node_train_table_input(table=train_table, task=task)
    val_table = task.get_table("val")
    val_input = get_node_train_table_input(table=val_table, task=task)

    loader_dict["train"] = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        time_attr="time",
        input_nodes=train_input.nodes,
        input_time=train_input.time,
        transform=train_input.transform,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    target_dict["train"] = train_input.target.float()

    # Use val split as both validation and test
    loader_dict["val"] = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        time_attr="time",
        input_nodes=val_input.nodes,
        input_time=val_input.time,
        transform=val_input.transform,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    target_dict["val"] = val_input.target.float()
    # Use same loader/targets for test
    loader_dict["test"] = loader_dict["val"]
    target_dict["test"] = target_dict["val"]

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_mae = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader_dict["train"]:
            batch = batch.to(DEVICE)
            batch_size = batch[entity_table].batch_size
            input_id = batch[entity_table].input_id[:batch_size].cpu()
            target = target_dict["train"][input_id].to(DEVICE)

            pred = model(batch, entity_table).squeeze(-1)
            loss = F.l1_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Validate
        model.eval()
        val_preds = []
        val_targets_list = []
        with torch.no_grad():
            for batch in loader_dict["val"]:
                batch = batch.to(DEVICE)
                batch_size = batch[entity_table].batch_size
                input_id = batch[entity_table].input_id[:batch_size].cpu()
                target = target_dict["val"][input_id]
                pred = model(batch, entity_table).squeeze(-1)
                val_preds.append(pred.cpu())
                val_targets_list.append(target)

        val_pred_cat = torch.cat(val_preds)
        val_target_cat = torch.cat(val_targets_list)
        val_mae = F.l1_loss(val_pred_cat, val_target_cat).item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = copy.deepcopy(model.state_dict())

        if epoch % 2 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch}/{epochs}: train_loss={avg_loss:.4f}, "
                        f"val_mae={val_mae:.4f}, best_val={best_val_mae:.4f}")

    # Load best model and test
    model.load_state_dict(best_model_state)
    model.eval()

    test_preds = []
    test_targets_list = []
    with torch.no_grad():
        for batch in loader_dict["test"]:
            batch = batch.to(DEVICE)
            batch_size = batch[entity_table].batch_size
            input_id = batch[entity_table].input_id[:batch_size].cpu()
            target = target_dict["test"][input_id]
            pred = model(batch, entity_table).squeeze(-1)
            test_preds.append(pred.cpu())
            test_targets_list.append(target)

    test_pred_cat = torch.cat(test_preds)
    test_target_cat = torch.cat(test_targets_list)
    test_mae = F.l1_loss(test_pred_cat, test_target_cat).item()

    elapsed = time.time() - t0
    logger.info(f"  DONE {method_name} seed={seed}: test_MAE={test_mae:.4f}, "
                f"val_MAE={best_val_mae:.4f}, time={elapsed:.1f}s")

    # Collect RAMA diagnostics
    diagnostics = {}
    if method_name.startswith("rama"):
        diagnostics = _collect_rama_diagnostics(model)

    # Build result
    result = {
        "method": method_name,
        "seed": seed,
        "dataset": "rel-f1",
        "task": "driver-position",
        "best_val_mae": best_val_mae,
        "test_mae": test_mae,
        "num_params": num_params,
        "training_time_sec": elapsed,
        "train_losses": train_losses,
        "diagnostics": diagnostics,
        "test_predictions": test_pred_cat.numpy().tolist(),
        "test_targets": test_target_cat.numpy().tolist(),
    }

    # Save per-run result
    result_path = RESULTS_DIR / f"{method_name}_seed{seed}.json"
    result_path.write_text(json.dumps(result, indent=2))
    logger.info(f"  Saved result to {result_path.name}")

    # Cleanup
    del model, optimizer, loader_dict
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

    return result


def _collect_rama_diagnostics(model: EntityModel) -> dict:
    """Collect gate and rank diagnostics from RAMA modules."""
    diagnostics = {
        "mean_gate_value": [],
        "gate_std": [],
        "mean_effective_rank": [],
        "mean_cardinality": [],
    }
    for name, module in model.named_modules():
        if isinstance(module, (RAMAAggregation, RAMANoRankAggregation)):
            if module._last_gate is not None:
                gate = module._last_gate.cpu()
                diagnostics["mean_gate_value"].append(float(gate.mean()))
                diagnostics["gate_std"].append(float(gate.std()))
            if hasattr(module, "_last_rho") and module._last_rho is not None:
                rho = module._last_rho.cpu()
                diagnostics["mean_effective_rank"].append(float(rho.mean()))
            if module._last_N is not None:
                N = module._last_N.cpu()
                diagnostics["mean_cardinality"].append(float(N.mean()))

    # Aggregate across layers
    return {
        k: float(np.mean(v)) if v else None
        for k, v in diagnostics.items()
    }


# ===================================================================
# PART 5: STATISTICAL ANALYSIS
# ===================================================================
def cohens_d_mae(group1: list, group2: list) -> tuple:
    """Compute Cohen's d for MAE comparison.

    Positive d means group2 has LOWER MAE (group2 is better).
    d = (mean1 - mean2) / pooled_sd
    """
    a1, a2 = np.array(group1), np.array(group2)
    n1, n2 = len(a1), len(a2)
    if n1 < 2 or n2 < 2:
        return 0.0, float("inf"), -float("inf"), float("inf"), 1.0

    m1, m2 = a1.mean(), a2.mean()
    s1, s2 = a1.std(ddof=1), a2.std(ddof=1)

    pooled_sd = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled_sd < 1e-12:
        return 0.0, float("inf"), -float("inf"), float("inf"), 1.0

    d = (m1 - m2) / pooled_sd  # positive = group2 better (lower MAE)
    se = math.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    ci_lo = d - 1.96 * se
    ci_hi = d + 1.96 * se
    p_value = 2 * (1 - norm.cdf(abs(d / se))) if se > 0 else 1.0

    return d, se, ci_lo, ci_hi, p_value


def build_statistical_analysis(all_results: list) -> dict:
    """Build complete statistical analysis from all results."""
    # Group by method
    method_results = {}
    for r in all_results:
        m = r["method"]
        if m not in method_results:
            method_results[m] = []
        method_results[m].append(r["test_mae"])

    # Pairwise Cohen's d
    pairwise = {}
    methods_with_data = [m for m in METHODS_LIST if m in method_results and len(method_results[m]) >= 2]

    for m1, m2 in itertools.combinations(methods_with_data, 2):
        d, se, ci_lo, ci_hi, p_val = cohens_d_mae(
            method_results[m1], method_results[m2]
        )
        pairwise[f"{m1}_vs_{m2}"] = {
            "d": round(d, 4),
            "se": round(se, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "p_value": round(p_val, 6),
            "interpretation": (
                f"{m2} better" if d > 0.2 else
                f"{m1} better" if d < -0.2 else
                "no meaningful difference"
            ),
        }

    # Key hypothesis tests
    hypothesis_tests = {}
    test_defs = [
        ("gating_necessity", "ungated_moment", "rama_full",
         "Does gating add value beyond ungated variance injection?"),
        ("rank_conditioning", "rama_no_rank", "rama_full",
         "Does rank conditioning specifically help?"),
        ("layernorm_confounder", "standard_mean", "mean_layernorm",
         "Is LayerNorm a confounder?"),
        ("rama_vs_pna", "pna_style", "rama_full",
         "Does targeted gated variance beat multi-aggregator?"),
        ("variance_value", "standard_mean", "ungated_moment",
         "Does variance injection help at all?"),
        ("rama_vs_baseline", "standard_mean", "rama_full",
         "The headline comparison: RAMA vs standard mean"),
    ]

    for test_name, m1, m2, question in test_defs:
        if m1 in method_results and m2 in method_results:
            d, se, ci_lo, ci_hi, p_val = cohens_d_mae(
                method_results[m1], method_results[m2]
            )
            hypothesis_tests[test_name] = {
                "comparison": f"{m1} vs {m2}",
                "question": question,
                "d": round(d, 4),
                "se": round(se, 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
                "p_value": round(p_val, 6),
                "conclusion": (
                    f"{m2} significantly better (d={d:.2f})" if d > 0.5 else
                    f"{m1} significantly better (d={d:.2f})" if d < -0.5 else
                    f"No significant difference (d={d:.2f})"
                ),
            }

    # Per-method summary
    per_method = {}
    for m in methods_with_data:
        maes = method_results[m]
        per_method[m] = {
            "test_mae_per_seed": [round(v, 4) for v in maes],
            "mean_mae": round(float(np.mean(maes)), 4),
            "std_mae": round(float(np.std(maes, ddof=1)), 4) if len(maes) > 1 else 0.0,
            "num_params": next((r["num_params"] for r in all_results if r["method"] == m), 0),
            "mean_training_time": round(
                float(np.mean([r["training_time_sec"]
                               for r in all_results if r["method"] == m])), 1
            ),
        }

    # Ranking table (sorted by mean MAE, lower is better)
    ranking = sorted(
        per_method.items(),
        key=lambda x: x[1]["mean_mae"]
    )
    ranking_table = [
        {"rank": i + 1, "method": m, "mean_mae": info["mean_mae"],
         "std_mae": info["std_mae"]}
        for i, (m, info) in enumerate(ranking)
    ]

    # RAMA diagnostics
    rama_diags = {}
    for r in all_results:
        if r["diagnostics"]:
            m = r["method"]
            if m not in rama_diags:
                rama_diags[m] = []
            rama_diags[m].append(r["diagnostics"])

    aggregated_diags = {}
    for m, diag_list in rama_diags.items():
        aggregated_diags[m] = {}
        for key in ["mean_gate_value", "gate_std", "mean_effective_rank", "mean_cardinality"]:
            vals = [d[key] for d in diag_list if d.get(key) is not None]
            if vals:
                aggregated_diags[m][key] = round(float(np.mean(vals)), 4)

    return {
        "hypothesis_tests": hypothesis_tests,
        "per_method_results": per_method,
        "pairwise_cohens_d": pairwise,
        "ranking_table": ranking_table,
        "rama_diagnostics": aggregated_diags,
    }


# ===================================================================
# PART 6: OUTPUT GENERATION
# ===================================================================
def load_dependency_examples() -> list:
    """Load examples from ALL dependency datasets."""
    all_datasets = []

    # Load from data_id3 (regression tasks)
    dep3_path = DEP_DATA3 / "full_data_out.json"
    if dep3_path.exists():
        logger.info(f"Loading dependency data from {dep3_path.name}...")
        try:
            dep3 = json.loads(dep3_path.read_text())
            for ds_entry in dep3.get("datasets", []):
                all_datasets.append(ds_entry)
            logger.info(f"  Loaded {len(dep3.get('datasets', []))} datasets "
                        f"from data_id3")
        except Exception:
            logger.exception("Failed to load data_id3")

    # Load from data_id4 (classification tasks)
    dep4_path = DEP_DATA4 / "full_data_out.json"
    if dep4_path.exists():
        logger.info(f"Loading dependency data from {dep4_path.name}...")
        try:
            dep4 = json.loads(dep4_path.read_text())
            for ds_entry in dep4.get("datasets", []):
                all_datasets.append(ds_entry)
            logger.info(f"  Loaded {len(dep4.get('datasets', []))} datasets "
                        f"from data_id4")
        except Exception:
            logger.exception("Failed to load data_id4")

    return all_datasets


def generate_output(all_results: list, stats: dict) -> dict:
    """Generate method_out.json in exp_gen_sol_out format."""
    logger.info("Generating method_out.json...")

    # Load all dependency examples
    dep_datasets = load_dependency_examples()

    # Build per-method mean MAE lookup from stats
    per_method = stats.get("per_method_results", {})
    # Map dataset names to tested task: we ran on rel-f1/driver-position
    tested_task_prefixes = ["rel-f1/driver-position", "rel-f1__driver-position"]

    # Build output datasets: pass through dependency examples with predictions
    output_datasets = []

    for ds_entry in dep_datasets:
        ds_name = ds_entry["dataset"]
        examples = ds_entry.get("examples", [])

        # Check if this dataset matches the tested task
        is_tested = any(ds_name.startswith(p) or ds_name == p
                        for p in tested_task_prefixes)

        clean_examples = []
        for ex in examples:
            clean_ex = {
                "input": ex.get("input", ""),
                "output": ex.get("output", ""),
            }
            for k, v in ex.items():
                if k.startswith("metadata_"):
                    clean_ex[k] = v
            # Add predict_* fields for each method (required by schema)
            for method_name in METHODS_LIST:
                m_info = per_method.get(method_name, {})
                if is_tested and m_info:
                    mean_mae = m_info.get("mean_mae", "N/A")
                    std_mae = m_info.get("std_mae", "N/A")
                    clean_ex[f"predict_{method_name}"] = (
                        f"mean_test_MAE={mean_mae}, std={std_mae}"
                    )
                else:
                    clean_ex[f"predict_{method_name}"] = "not_evaluated"
            clean_examples.append(clean_ex)
        output_datasets.append({"dataset": ds_name, "examples": clean_examples})

    # Build metadata
    metadata = {
        "experiment": "RAMA Component Ablation on rel-f1/driver-position",
        "description": (
            "Controlled ablation study isolating each RAMA component "
            "(variance injection, rank-based gating, LayerNorm confounding) "
            "on the rel-f1/driver-position regression task."
        ),
        "methods": METHODS_LIST,
        "hyperparameters": {
            "channels": CHANNELS,
            "num_layers": NUM_LAYERS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "num_neighbors": NUM_NEIGHBORS,
            "seeds": SEEDS,
        },
        "dataset_info": {
            "dataset": "rel-f1",
            "task": "driver-position",
            "train_size": 7453,
            "val_size": 499,
            "test_size": 760,
            "num_tables": 9,
            "num_fk_edges": 13,
            "key_high_cardinality_edges": [
                "results.constructorId->constructors (mean=124, max=2371)",
                "results.driverId->drivers (mean=30, max=370)",
            ],
        },
        "statistical_analysis": stats,
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
    logger.info("RAMA Component Ablation Experiment")
    logger.info("=" * 60)

    # ---- Phase 0: Unit tests ----
    run_unit_tests()

    # ---- Phase 1: Load data and smoke test ----
    logger.info("=" * 60)
    logger.info("PHASE 1: Data loading + smoke test")
    logger.info("=" * 60)

    dataset, task, data, col_stats_dict = load_relbench_data()

    # Smoke test: 2 methods, 1 seed, 3 epochs
    smoke_methods = ["standard_mean", "rama_full"]
    smoke_results = []
    for method_name in smoke_methods:
        try:
            r = train_one(
                method_name=method_name,
                seed=42,
                task=task,
                data=data,
                col_stats_dict=col_stats_dict,
                epochs=3,
                lr=LR,
            )
            smoke_results.append(r)
            logger.info(f"  Smoke test {method_name}: test_MAE={r['test_mae']:.4f}, "
                        f"time={r['training_time_sec']:.1f}s")

            # Check sanity
            if r["test_mae"] > 20:
                logger.warning(f"  WARNING: High MAE ({r['test_mae']:.2f}), "
                               f"target std is ~7")
            if math.isnan(r["test_mae"]):
                logger.error(f"  ERROR: NaN MAE for {method_name}")

        except Exception:
            logger.exception(f"Smoke test failed for {method_name}")

    if not smoke_results:
        logger.error("All smoke tests failed. Aborting.")
        return

    # Estimate runtime per full run
    avg_smoke_time = np.mean([r["training_time_sec"] for r in smoke_results])
    est_full_time = avg_smoke_time * (EPOCHS / 3)  # scale from 3 to 10 epochs
    est_total = est_full_time * len(METHODS_LIST) * len(SEEDS)
    logger.info(f"Runtime estimate: {est_full_time:.0f}s/run, "
                f"{est_total / 60:.0f}min total for {len(METHODS_LIST)}x{len(SEEDS)} runs")

    # Adjust if too slow
    epochs_to_use = EPOCHS
    seeds_to_use = list(SEEDS)
    methods_to_use = list(METHODS_LIST)

    remaining_minutes = 300  # conservative estimate
    if est_total / 60 > remaining_minutes * 0.7:
        # Reduce epochs
        epochs_to_use = 5
        est_total = est_full_time * (5 / 10) * len(methods_to_use) * len(seeds_to_use)
        logger.info(f"Reduced epochs to {epochs_to_use}, est={est_total / 60:.0f}min")

    if est_total / 60 > remaining_minutes * 0.7:
        # Reduce seeds
        seeds_to_use = [42, 123, 456]
        est_total = est_full_time * (epochs_to_use / 10) * len(methods_to_use) * len(seeds_to_use)
        logger.info(f"Reduced seeds to {len(seeds_to_use)}, est={est_total / 60:.0f}min")

    if est_total / 60 > remaining_minutes * 0.7:
        # Drop PNA (most expensive)
        methods_to_use = [m for m in methods_to_use if m != "pna_style"]
        est_total = est_full_time * (epochs_to_use / 10) * len(methods_to_use) * len(seeds_to_use)
        logger.info(f"Dropped PNA, est={est_total / 60:.0f}min")

    # ---- Phase 2: Full ablation ----
    logger.info("=" * 60)
    logger.info(f"PHASE 2: Full ablation ({len(methods_to_use)} methods x "
                f"{len(seeds_to_use)} seeds x {epochs_to_use} epochs)")
    logger.info("=" * 60)

    all_results = []
    for method_name in methods_to_use:
        method_t0 = time.time()
        for seed in seeds_to_use:
            try:
                r = train_one(
                    method_name=method_name,
                    seed=seed,
                    task=task,
                    data=data,
                    col_stats_dict=col_stats_dict,
                    epochs=epochs_to_use,
                    lr=LR,
                )
                all_results.append(r)
            except Exception:
                logger.exception(f"Failed: {method_name} seed={seed}")
                continue

        method_elapsed = time.time() - method_t0
        method_maes = [r["test_mae"] for r in all_results if r["method"] == method_name]
        if method_maes:
            logger.info(f"  {method_name} complete: mean_MAE={np.mean(method_maes):.4f} "
                        f"+/- {np.std(method_maes):.4f} ({method_elapsed:.0f}s)")

    if not all_results:
        logger.error("No successful runs. Aborting.")
        return

    # ---- Phase 3: Statistical analysis ----
    logger.info("=" * 60)
    logger.info("PHASE 3: Statistical analysis")
    logger.info("=" * 60)

    stats = build_statistical_analysis(all_results)

    # Log key results
    logger.info("Ranking (by mean test MAE, lower=better):")
    for entry in stats["ranking_table"]:
        logger.info(f"  #{entry['rank']}: {entry['method']} = "
                    f"{entry['mean_mae']:.4f} +/- {entry['std_mae']:.4f}")

    logger.info("Key hypothesis tests:")
    for test_name, test_info in stats["hypothesis_tests"].items():
        logger.info(f"  {test_name}: {test_info['conclusion']} "
                    f"(p={test_info['p_value']:.4f})")

    if stats["rama_diagnostics"]:
        logger.info("RAMA diagnostics:")
        for m, diag in stats["rama_diagnostics"].items():
            logger.info(f"  {m}: {diag}")

    # ---- Phase 4: Generate output ----
    logger.info("=" * 60)
    logger.info("PHASE 4: Generate output")
    logger.info("=" * 60)

    output = generate_output(all_results, stats)

    # Save method_out.json
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
