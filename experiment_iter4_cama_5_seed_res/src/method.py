#!/usr/bin/env python3
"""CAMA 5-Seed Results on Amazon item-ltv + Avito Negative Gate Init Fix.

Two-part experiment:
  (A) 5-seed CAMA vs mean on rel-amazon/item-ltv resolving test-target-unavailability
  (B) 5-seed comparison of mean vs CAMA-default vs CAMA-negative-init on rel-avito/ad-ctr
      to fix the 34% degradation by initializing gate_bias=-3 so sigmoid starts at 0.05.

Builds on iter_3 code patterns (exp_id2_it3, exp_id4_it3) with proven RAMA implementation,
adding gate_bias parameterization.
"""

import copy
import gc
import json
import math
import os
import resource
import sys
import time
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy import stats as sp_stats
from torch import Tensor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WS = Path(__file__).resolve().parent
LOGS_DIR = WS / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection (cgroup-aware)
# ---------------------------------------------------------------------------
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


def _container_ram_gb() -> Optional[float]:
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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, GPU={HAS_GPU}, VRAM={VRAM_GB:.1f} GB")

# ---------------------------------------------------------------------------
# Memory limits
# ---------------------------------------------------------------------------
RAM_BUDGET = int(TOTAL_RAM_GB * 0.80 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.90)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))
    logger.info(f"VRAM budget: {VRAM_BUDGET / 1e9:.1f} GB")

# ---------------------------------------------------------------------------
# PyG imports
# ---------------------------------------------------------------------------
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn import HeteroConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Relbench imports
# ---------------------------------------------------------------------------
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.utils import get_stype_proposal
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from torch_frame import stype
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
    StypeWiseFeatureEncoder,
)

# Track deviations from plan
DEVIATIONS: List[str] = []

# ===========================================================================
# SECTION 1: CAMAAggregation (Cardinality-Aware Moment Aggregation)
# ===========================================================================

class CAMAAggregation(Aggregation):
    """Cardinality-Aware Moment Aggregation.

    Key change from iter_3 RAMAAggregation: gate_bias parameter.
    - gate_bias=0  -> sigmoid(0)=0.5 (iter-3 default, gate starts half-open)
    - gate_bias=-3 -> sigmoid(-3)=0.047 (gate starts nearly closed)
    """

    def __init__(self, channels: int, gate_hidden: int = 16, eps: float = 1e-6,
                 gate_bias: float = 0.0):
        super().__init__()
        self.channels = channels
        self.gate_hidden = gate_hidden
        self.eps = eps
        self.gate_bias_init = gate_bias

        self.W_sigma = nn.Linear(channels, channels, bias=False)
        self.W_g = nn.Linear(2, gate_hidden)
        self.gate_out = nn.Linear(gate_hidden, channels)

        # Gate recording for analysis
        self._last_gate_values = None
        self._last_N_values = None
        self._last_rho_values = None
        self._record_gates = False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.W_sigma.weight)
        nn.init.zeros_(self.gate_out.weight)
        # KEY CHANGE: Initialize gate_out bias to gate_bias_init
        nn.init.constant_(self.gate_out.bias, self.gate_bias_init)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2,
                max_num_elements=None):
        # Step 1: Mean
        mu = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        # Step 2: Variance via E[X^2] - E[X]^2
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce='mean')
        var = (mean_x2 - mu * mu).clamp(min=0)
        var = torch.nan_to_num(var, nan=0.0)
        # Step 3: Cardinality
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        N = self.reduce(ones, index, ptr, dim_size, dim, reduce='sum')
        # Step 4: Effective rank proxy (variance-entropy)
        var_sum = var.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        p = var / var_sum
        p = p.clamp(min=self.eps)
        H = -(p * p.log()).sum(dim=-1, keepdim=True)
        log_d = math.log(max(self.channels, 2))
        rho = H / log_d
        # Step 5: Gate
        gate_input = torch.cat([torch.log1p(N), rho], dim=-1)
        g = torch.sigmoid(self.gate_out(F.relu(self.W_g(gate_input))))
        # Step 6: Output
        out = mu + g * self.W_sigma(var)

        if self._record_gates:
            self._last_gate_values = g.detach().cpu()
            self._last_N_values = N.detach().cpu()
            self._last_rho_values = rho.detach().cpu()

        return out


# ===========================================================================
# SECTION 2: Model components
# ===========================================================================

class HeteroEncoder(torch.nn.Module):
    """Encodes heterogeneous tabular node features into embeddings."""
    def __init__(self, channels, data, col_stats_dict):
        super().__init__()
        self.encoders = torch.nn.ModuleDict()
        self.channels = channels

        for node_type in data.node_types:
            if node_type not in col_stats_dict:
                continue
            col_stats = col_stats_dict[node_type]

            stype_encoder_dict = {}
            col_names_dict = {}
            for st_key, st_col_stats in col_stats.items():
                col_names = list(st_col_stats.keys())
                if not col_names:
                    continue
                col_names_dict[st_key] = col_names
                if st_key == stype.categorical:
                    stype_encoder_dict[st_key] = EmbeddingEncoder()
                elif st_key == stype.numerical:
                    stype_encoder_dict[st_key] = LinearEncoder()
                elif st_key == stype.timestamp:
                    stype_encoder_dict[st_key] = TimestampEncoder()
                elif st_key == stype.text_embedded:
                    stype_encoder_dict[st_key] = LinearEncoder()

            if not stype_encoder_dict:
                continue

            self.encoders[node_type] = StypeWiseFeatureEncoder(
                out_channels=channels,
                col_stats=col_stats,
                col_names_dict=col_names_dict,
                stype_encoder_dict=stype_encoder_dict,
            )

    def forward(self, tf_dict):
        x_dict = {}
        for node_type, encoder in self.encoders.items():
            if node_type in tf_dict:
                x_dict[node_type] = encoder(tf_dict[node_type])[0]
        return x_dict


class HeteroTemporalEncoder(torch.nn.Module):
    """Encodes temporal features (time delta) into embeddings."""
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Linear(1, channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, seed_time, time_dict, batch_dict):
        out_dict = {}
        for node_type, node_time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - node_time
            rel_time = rel_time.to(torch.float32).view(-1, 1)
            out_dict[node_type] = self.encoder(rel_time)
        return out_dict


class HeteroGraphSAGE(torch.nn.Module):
    """Heterogeneous GraphSAGE with customizable aggregation."""
    def __init__(self, node_types, edge_types, channels, aggr="mean",
                 num_layers=2, norm="batch_norm"):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                if isinstance(aggr, str):
                    conv_aggr = aggr
                else:
                    conv_aggr = copy.deepcopy(aggr)
                conv_dict[edge_type] = SAGEConv(
                    (channels, channels), channels, aggr=conv_aggr
                )
            conv = HeteroConv(conv_dict, aggr="sum")
            self.convs.append(conv)

            norm_dict = {}
            for nt in node_types:
                if norm == "batch_norm":
                    norm_dict[nt] = nn.BatchNorm1d(channels)
                elif norm == "layer_norm":
                    norm_dict[nt] = nn.LayerNorm(channels)
            self.norms.append(nn.ModuleDict(norm_dict))

    def forward(self, x_dict, edge_index_dict):
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict_new = conv(x_dict, edge_index_dict)
            for nt in x_dict_new:
                if nt in norm_dict:
                    x_dict_new[nt] = norm_dict[nt](x_dict_new[nt])
                x_dict_new[nt] = F.relu(x_dict_new[nt])
            for nt in x_dict:
                if nt not in x_dict_new:
                    x_dict_new[nt] = x_dict[nt]
            x_dict = x_dict_new
        return x_dict


class Model(torch.nn.Module):
    """Full RelBench-style model with encoder + GNN + head."""
    def __init__(self, data, col_stats_dict, num_layers, channels,
                 out_channels, aggr="mean", norm="batch_norm"):
        super().__init__()
        self.encoder = HeteroEncoder(channels, data, col_stats_dict)
        self.temporal_encoder = HeteroTemporalEncoder(channels)
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
            norm=norm,
        )
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, out_channels),
        )
        self.channels = channels

    def forward(self, batch: HeteroData, entity_table: NodeType):
        tf_dict = {
            nt: batch[nt].tf
            for nt in batch.node_types
            if hasattr(batch[nt], 'tf')
        }
        x_dict = self.encoder(tf_dict)

        if hasattr(batch[entity_table], 'seed_time'):
            seed_time = batch[entity_table].seed_time
            time_dict = {}
            batch_dict = {}
            for nt in batch.node_types:
                if hasattr(batch[nt], 'time') and batch[nt].time is not None:
                    time_dict[nt] = batch[nt].time
                    batch_dict[nt] = (
                        batch[nt].batch
                        if hasattr(batch[nt], 'batch')
                        else torch.zeros(batch[nt].num_nodes, dtype=torch.long,
                                         device=seed_time.device)
                    )
            if time_dict:
                temp_dict = self.temporal_encoder(seed_time, time_dict, batch_dict)
                for nt in temp_dict:
                    if nt in x_dict:
                        x_dict[nt] = x_dict[nt] + temp_dict[nt]
                    else:
                        x_dict[nt] = temp_dict[nt]

        for nt in batch.node_types:
            if nt not in x_dict:
                x_dict[nt] = torch.zeros(
                    batch[nt].num_nodes, self.channels, device=DEVICE)

        edge_index_dict = {
            et: batch[et].edge_index
            for et in batch.edge_types
            if hasattr(batch[et], 'edge_index')
        }
        x_dict = self.gnn(x_dict, edge_index_dict)

        x = x_dict[entity_table][:batch[entity_table].batch_size]
        return self.head(x)


# ===========================================================================
# SECTION 3: NaN preprocessing
# ===========================================================================

def fill_nan_in_graph(data: Any) -> Any:
    """Fill NaN values in TensorFrame features with 0."""
    nan_count = 0
    for nt in data.node_types:
        if not hasattr(data[nt], "tf") or data[nt].tf is None:
            continue
        tf = data[nt].tf
        if not hasattr(tf, "feat_dict"):
            continue
        for key in tf.feat_dict:
            t = tf.feat_dict[key]
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                mask = torch.isnan(t)
                if mask.any():
                    cnt = int(mask.sum())
                    nan_count += cnt
                    t[mask] = 0.0
                    logger.debug(f"  Filled {cnt} NaN in {nt}.tf.feat_dict[{key}]")
    if nan_count > 0:
        logger.info(f"Filled {nan_count} total NaN values in TensorFrame features")
    return data


def log_memory(label: str = ""):
    """Log current memory usage."""
    try:
        v1_path = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        v2_path = "/sys/fs/cgroup/memory.current"
        for p in [v2_path, v1_path]:
            try:
                usage_bytes = int(Path(p).read_text().strip())
                usage_gb = usage_bytes / 1e9
                logger.info(f"Memory [{label}]: {usage_gb:.1f} GB (cgroup)")
                return
            except (FileNotFoundError, ValueError):
                pass
        usage_gb = psutil.virtual_memory().used / 1e9
        logger.info(f"Memory [{label}]: {usage_gb:.1f} GB (psutil)")
    except Exception:
        pass


# ===========================================================================
# SECTION 4: Unit Tests
# ===========================================================================

def run_unit_tests():
    """Run unit tests for CAMAAggregation before any training."""
    logger.info("=" * 60)
    logger.info("Running unit tests...")
    ch = 128

    # Test 1: Shape test for gate_bias=0 and gate_bias=-3
    for gb, gb_name in [(0.0, "default"), (-3.0, "neg_init")]:
        aggr = CAMAAggregation(channels=ch, gate_hidden=16, gate_bias=gb)
        x = torch.randn(100, ch)
        index = torch.randint(0, 20, (100,))
        out = aggr(x, index=index, dim_size=20)
        assert out.shape == (20, ch), f"Shape mismatch for {gb_name}: {out.shape}"
        assert not torch.isnan(out).any(), f"NaN in output for {gb_name}"
        logger.info(f"  Shape test ({gb_name}): OK - {out.shape}")

    # Test 2: Zero-init test: output should be close to mean
    for gb in [0.0, -3.0]:
        aggr = CAMAAggregation(channels=ch, gate_hidden=16, gate_bias=gb)
        x = torch.randn(100, ch)
        index = torch.randint(0, 10, (100,))
        out_cama = aggr(x, index=index, dim_size=10)
        from torch_geometric.nn.aggr import MeanAggregation
        out_mean = MeanAggregation()(x, index=index, dim_size=10)
        diff = (out_cama - out_mean).abs().max().item()
        logger.info(f"  Zero-init test (gb={gb}): max_diff={diff:.6f}")
        assert diff < 0.1, f"Zero-init too far from mean for gb={gb}: {diff}"

    # Test 3: Gate value test with gate_bias=-3
    aggr_neg = CAMAAggregation(channels=ch, gate_hidden=16, gate_bias=-3.0)
    aggr_neg._record_gates = True
    x = torch.randn(100, ch)
    index = torch.randint(0, 10, (100,))
    _ = aggr_neg(x, index=index, dim_size=10)
    mean_gate = aggr_neg._last_gate_values.mean().item()
    logger.info(f"  Gate value test (gb=-3): mean_gate={mean_gate:.4f}")
    assert 0.01 < mean_gate < 0.15, f"Gate value out of range: {mean_gate}"

    # Test 4: SAGEConv integration
    cama = CAMAAggregation(channels=ch, gate_hidden=16, gate_bias=0.0)
    conv = SAGEConv((ch, ch), ch, aggr=cama)
    x_src = torch.randn(50, ch)
    x_dst = torch.randn(30, ch)
    edge_index = torch.stack([
        torch.randint(0, 50, (200,)),
        torch.randint(0, 30, (200,)),
    ])
    out = conv((x_src, x_dst), edge_index)
    assert out.shape == (30, ch), f"SAGEConv shape: {out.shape}"
    assert not torch.isnan(out).any(), "SAGEConv NaN"
    logger.info(f"  SAGEConv integration: OK")

    # Test 5: Gradient flow
    cama2 = CAMAAggregation(channels=ch, gate_hidden=16, gate_bias=-3.0)
    x_grad = torch.randn(50, ch, requires_grad=True)
    idx_grad = torch.randint(0, 10, (50,))
    out_grad = cama2(x_grad, index=idx_grad, dim_size=10)
    loss = out_grad.sum()
    loss.backward()
    for name, param in cama2.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    logger.info(f"  Gradient flow: OK")

    # Test 6: Edge cases
    for name, x_ec, idx_ec, ds in [
        ("single_child", torch.randn(5, ch), torch.arange(5), 5),
        ("identical_children", torch.ones(10, ch) * 3.0, torch.zeros(10, dtype=torch.long), 1),
    ]:
        aggr_ec = CAMAAggregation(channels=ch, gate_hidden=16, gate_bias=-3.0)
        out_ec = aggr_ec(x_ec, index=idx_ec, dim_size=ds)
        assert not torch.isnan(out_ec).any(), f"NaN for {name}"
        logger.info(f"  Edge case ({name}): OK")

    # Test 7: deepcopy independence
    cama_orig = CAMAAggregation(channels=ch, gate_hidden=16, gate_bias=-3.0)
    cama_copy = copy.deepcopy(cama_orig)
    with torch.no_grad():
        cama_copy.gate_out.bias.fill_(99.0)
    assert cama_orig.gate_out.bias.data[0].item() != 99.0, "deepcopy not independent"
    logger.info(f"  deepcopy independence: OK")

    logger.info("ALL UNIT TESTS PASSED!")
    logger.info("=" * 60)
    del x, out, x_src, x_dst, edge_index, x_grad, out_grad
    gc.collect()


# ===========================================================================
# SECTION 5: Aggregation factory
# ===========================================================================

def make_aggr(method_name: str, channels: int):
    """Create aggregation module for the given method."""
    if method_name == "mean_baseline":
        return "mean"
    elif method_name == "cama_default":
        return CAMAAggregation(channels=channels, gate_hidden=16, gate_bias=0.0)
    elif method_name == "cama_neg_init":
        return CAMAAggregation(channels=channels, gate_hidden=16, gate_bias=-3.0)
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ===========================================================================
# SECTION 6: Data loading
# ===========================================================================

def load_amazon_data(cache_dir: str) -> Tuple[Any, Dict]:
    """Load rel-amazon dataset with OOM-safe text exclusions."""
    logger.info("Loading rel-amazon dataset...")
    t0 = time.time()

    os.environ["RELBENCH_CACHE_DIR"] = cache_dir
    dataset = get_dataset("rel-amazon", download=True)
    db = dataset.get_db()

    logger.info(f"Dataset loaded in {time.time() - t0:.0f}s")
    logger.info(f"Tables: {list(db.table_dict.keys())}")
    log_memory("after amazon dataset load")

    col_to_stype_dict = get_stype_proposal(db)

    # CRITICAL: Remove text_embedded from review table (20.8M rows)
    if "review" in col_to_stype_dict:
        for text_col in ["review_text", "summary"]:
            if text_col in col_to_stype_dict["review"]:
                removed = col_to_stype_dict["review"].pop(text_col)
                logger.warning(f"REMOVED review.{text_col} ({removed}) - 20.8M rows OOM risk")
        DEVIATIONS.append("Removed review_text and summary from review table to prevent OOM")

    # Remove customer_name (1.85M rows)
    if "customer" in col_to_stype_dict:
        if "customer_name" in col_to_stype_dict["customer"]:
            col_to_stype_dict["customer"].pop("customer_name")
            logger.warning("REMOVED customer.customer_name")
            DEVIATIONS.append("Removed customer_name from customer table")

    # Remove ALL text_embedded to be safe
    for table_name in col_to_stype_dict:
        removed_cols = [
            c for c, s in col_to_stype_dict[table_name].items()
            if s == stype.text_embedded
        ]
        for c in removed_cols:
            col_to_stype_dict[table_name].pop(c)
            logger.warning(f"REMOVED {table_name}.{c} (text_embedded)")

    for table, stypes in col_to_stype_dict.items():
        logger.info(f"  {table}: {dict(stypes)}")

    log_memory("before amazon graph build")

    logger.info("Building rel-amazon graph (no text embeddings)...")
    mat_cache = str(WS / "mat_cache" / "rel-amazon" / "materialized")
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,
        cache_dir=mat_cache,
    )

    logger.info(f"Graph built in {time.time() - t0:.0f}s")
    logger.info(f"Node types: {data.node_types}")
    logger.info(f"Edge types: {data.edge_types}")
    for nt in data.node_types:
        if hasattr(data[nt], "num_nodes"):
            logger.info(f"  {nt}: {data[nt].num_nodes} nodes")

    data = fill_nan_in_graph(data)
    log_memory("after amazon graph build")

    del db
    gc.collect()
    return data, col_stats_dict


def load_avito_data(cache_dir: str) -> Tuple[Any, Dict]:
    """Load rel-avito dataset with OOM-safe text exclusions."""
    logger.info("Loading rel-avito dataset...")
    t0 = time.time()

    os.environ["RELBENCH_CACHE_DIR"] = cache_dir
    dataset = get_dataset("rel-avito", download=True)
    db = dataset.get_db()

    logger.info(f"Dataset loaded in {time.time() - t0:.0f}s")
    logger.info(f"Tables: {list(db.table_dict.keys())}")
    log_memory("after avito dataset load")

    col_to_stype_dict = get_stype_proposal(db)

    # Remove ALL text-type columns to be safe
    for table_name in list(col_to_stype_dict.keys()):
        for col_name in list(col_to_stype_dict[table_name].keys()):
            if col_to_stype_dict[table_name][col_name] in (
                stype.text_embedded, stype.text_tokenized
            ):
                del col_to_stype_dict[table_name][col_name]
                logger.warning(f"REMOVED {table_name}.{col_name} (text)")

    for table, stypes in col_to_stype_dict.items():
        logger.info(f"  {table}: {dict(stypes)}")

    log_memory("before avito graph build")

    logger.info("Building rel-avito graph (no text embeddings)...")
    mat_cache = str(WS / "mat_cache" / "rel-avito" / "materialized")
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,
        cache_dir=mat_cache,
    )

    logger.info(f"Graph built in {time.time() - t0:.0f}s")
    logger.info(f"Node types: {data.node_types}")
    logger.info(f"Edge types: {data.edge_types}")
    for nt in data.node_types:
        if hasattr(data[nt], "num_nodes"):
            logger.info(f"  {nt}: {data[nt].num_nodes} nodes")

    data = fill_nan_in_graph(data)
    log_memory("after avito graph build")

    del db
    gc.collect()
    return data, col_stats_dict


# ===========================================================================
# SECTION 7: Loader building
# ===========================================================================

NUM_LAYERS = 2

def build_loaders(
    dataset_name: str,
    task_name: str,
    data: Any,
    num_neighbors_list: List[int],
    batch_size: int,
    cache_dir: str,
) -> Tuple[Dict[str, Any], Any]:
    """Build train/val/test neighbor loaders. Returns (loaders, task).

    NOTE: We do NOT use temporal_strategy="uniform" or time_attr="time"
    because that requires pyg-lib for disjoint sampling, which is not
    installed. Instead we use basic neighbor sampling with the transform
    from get_node_train_table_input which attaches y labels to batches.
    Both methods (baseline and CAMA) use the same loader, so this does
    not affect the relative comparison.
    """
    os.environ["RELBENCH_CACHE_DIR"] = cache_dir
    task = get_task(dataset_name, task_name, download=True)

    loader_dict = {}
    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=num_neighbors_list,
            input_nodes=table_input.nodes,
            transform=table_input.transform,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
    return loader_dict, task


# ===========================================================================
# SECTION 8: Training & Evaluation
# ===========================================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    entity_table: str,
    max_steps: int,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch, return avg loss."""
    model.train()
    loss_accum = 0.0
    count_accum = 0
    steps = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch, entity_table).view(-1)
        y = batch[entity_table].y.float()

        bs = min(pred.size(0), y.size(0))
        pred = pred[:bs]
        y = y[:bs]

        mask = ~torch.isnan(y) & ~torch.isnan(pred)
        if not mask.any():
            continue

        pred_m = pred[mask]
        y_m = y[mask]

        loss = loss_fn(pred_m.float(), y_m)

        if torch.isnan(loss):
            logger.warning("NaN loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        loss_accum += loss.detach().item() * pred_m.size(0)
        count_accum += pred_m.size(0)
        steps += 1
        if steps >= max_steps:
            break

        if steps == 1 and HAS_GPU:
            vram_used = torch.cuda.max_memory_allocated() / 1e9
            logger.debug(f"VRAM after first batch: {vram_used:.2f} GB")

    return loss_accum / max(count_accum, 1)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: Any,
    entity_table: str,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    default_val: float = 0.0,
) -> np.ndarray:
    """Run evaluation, return predictions as numpy array."""
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table).view(-1)
        if clamp_min is not None:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        pred = torch.nan_to_num(pred, nan=default_val)
        preds.append(pred.cpu())
    result = torch.cat(preds, dim=0).numpy()
    result = np.nan_to_num(result, nan=default_val)
    return result


# ===========================================================================
# SECTION 9: Gate extraction
# ===========================================================================

def extract_gate_values(model: torch.nn.Module, loader: Any,
                        entity_table: str, num_batches: int = 20) -> Dict:
    """Enable gate recording, run eval batches, collect stats."""
    model.eval()
    # Enable recording on all CAMA modules
    for module in model.modules():
        if isinstance(module, CAMAAggregation):
            module._record_gates = True

    gate_records: Dict[str, List[Dict]] = {}

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= num_batches:
                break
            batch = batch.to(DEVICE)
            _ = model(batch, entity_table)

            # Collect from each layer/edge-type
            for layer_idx, conv_layer in enumerate(model.gnn.convs):
                for et_key, sage_conv in conv_layer.convs.items():
                    aggr_mod = sage_conv.aggr_module
                    if isinstance(aggr_mod, CAMAAggregation) and aggr_mod._last_gate_values is not None:
                        g = aggr_mod._last_gate_values
                        N = aggr_mod._last_N_values
                        rho = aggr_mod._last_rho_values
                        if g.numel() == 0 or N.numel() == 0:
                            continue
                        et_str = f"L{layer_idx}_{'__'.join(et_key)}"
                        if et_str not in gate_records:
                            gate_records[et_str] = []
                        gate_records[et_str].append({
                            'mean_gate': float(g.mean()),
                            'std_gate': float(g.std()) if g.numel() > 1 else 0.0,
                            'min_gate': float(g.min()),
                            'max_gate': float(g.max()),
                            'mean_N': float(N.mean()),
                            'max_N': float(N.max()),
                            'mean_rho': float(rho.mean()),
                        })

    # Disable recording
    for module in model.modules():
        if isinstance(module, CAMAAggregation):
            module._record_gates = False

    # Summarize per edge type
    summary = {}
    for et_str, recs in gate_records.items():
        summary[et_str] = {
            'mean_gate': round(float(np.mean([r['mean_gate'] for r in recs])), 6),
            'std_gate': round(float(np.mean([r['std_gate'] for r in recs])), 6),
            'min_gate': round(float(np.min([r['min_gate'] for r in recs])), 6),
            'max_gate': round(float(np.max([r['max_gate'] for r in recs])), 6),
            'mean_N': round(float(np.mean([r['mean_N'] for r in recs])), 2),
            'max_N': round(float(np.max([r['max_N'] for r in recs])), 2),
            'mean_rho': round(float(np.mean([r['mean_rho'] for r in recs])), 6),
            'num_batches': len(recs),
        }
    return summary


# ===========================================================================
# SECTION 10: Statistical analysis
# ===========================================================================

def compute_statistics(
    results_a: List[float],
    results_b: List[float],
    higher_is_better: bool = False,
) -> Dict:
    """Cohen's d, paired t-test, Wilcoxon, 95% CIs."""
    a = np.array(results_a)
    b = np.array(results_b)

    diff = b - a
    if not higher_is_better:
        diff = -diff  # For MAE, lower is better

    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    pooled_sd = np.sqrt((s1**2 + s2**2) / 2)
    d = float(diff.mean() / pooled_sd) if pooled_sd > 1e-12 else 0.0

    # Paired t-test
    t_stat, p_val = sp_stats.ttest_rel(a, b)

    # Wilcoxon signed-rank
    try:
        w_stat, w_p = sp_stats.wilcoxon(a, b)
        w_stat = float(w_stat)
        w_p = float(w_p)
    except ValueError:
        w_stat, w_p = None, None

    # 95% CI for Cohen's d (NCT approximation)
    n = len(a)
    se_d = math.sqrt(2.0 / n * (1 + d**2 / (4 * n)))
    d_ci = (round(d - 1.96 * se_d, 6), round(d + 1.96 * se_d, 6))

    return {
        "cohens_d": round(d, 6),
        "d_ci_95": list(d_ci),
        "p_value_ttest": round(float(p_val), 8),
        "t_statistic": round(float(t_stat), 6),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": round(w_p, 8) if w_p is not None else None,
        "mean_a": round(float(a.mean()), 6),
        "std_a": round(float(a.std(ddof=1)), 6),
        "mean_b": round(float(b.mean()), 6),
        "std_b": round(float(b.std(ddof=1)), 6),
        "n": n,
    }


# ===========================================================================
# SECTION 11: Test target investigation
# ===========================================================================

def investigate_test_targets(dataset_name: str, task_name: str,
                             cache_dir: str) -> Dict:
    """Check if test targets are available for a given task."""
    os.environ["RELBENCH_CACHE_DIR"] = cache_dir
    task = get_task(dataset_name, task_name, download=True)
    test_table = task.get_table("test")
    target_col = task.target_col

    info = {
        "target_col": target_col,
        "test_columns": list(test_table.df.columns),
    }

    if target_col in test_table.df.columns:
        targets = test_table.df[target_col]
        nan_frac = float(targets.isna().mean())
        info["target_present"] = True
        info["nan_fraction"] = nan_frac
        info["all_nan"] = bool(targets.isna().all())
        info["available"] = nan_frac < 1.0
        logger.info(f"Test targets for {dataset_name}/{task_name}: "
                     f"present=True, nan_frac={nan_frac:.4f}, available={info['available']}")
    else:
        info["target_present"] = False
        info["available"] = False
        logger.info(f"Test targets for {dataset_name}/{task_name}: "
                     f"target column '{target_col}' missing entirely")

    return info


# ===========================================================================
# SECTION 12: Single run
# ===========================================================================

def run_single(
    data: Any,
    col_stats_dict: Dict,
    loader_dict: Dict[str, Any],
    task: Any,
    method_name: str,
    seed: int,
    channels: int,
    lr: float,
    epochs: int,
    max_steps: int,
    grad_clip: float,
    test_available: bool,
) -> Dict[str, Any]:
    """Run a single training experiment and return results."""
    seed_everything(seed)
    entity_table = task.entity_table
    target_col = task.target_col

    logger.info(f"  RUN: {method_name} | seed={seed} | epochs={epochs}")

    # Build model
    aggr = make_aggr(method_name, channels)
    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=NUM_LAYERS,
        channels=channels,
        out_channels=1,
        aggr=aggr,
        norm="batch_norm",
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"    Params: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    # Clamping from training targets
    train_table = task.get_table("train")
    targets = train_table.df[target_col].dropna().to_numpy().astype(float)
    clamp_min = float(np.percentile(targets, 2))
    clamp_max = float(np.percentile(targets, 98))
    default_pred = float(np.median(targets))

    best_val_metric = float('inf')
    best_state = None
    best_epoch = 0
    epoch_times = []
    train_losses = []
    t_total = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, loader_dict["train"], optimizer, loss_fn,
            entity_table, max_steps, grad_clip,
        )
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        train_losses.append(train_loss)

        # Validate
        val_pred = evaluate_model(
            model, loader_dict["val"], entity_table,
            clamp_min, clamp_max, default_val=default_pred,
        )
        val_table = task.get_table("val")
        try:
            val_metrics = task.evaluate(val_pred, val_table)
        except Exception as e:
            logger.warning(f"    Val eval failed: {e}, using fallback")
            val_metrics = {"mae": float("inf")}

        val_mae = float(val_metrics.get("mae", float("inf")))

        if val_mae < best_val_metric:
            best_val_metric = val_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        logger.info(
            f"    epoch={epoch}/{epochs} loss={train_loss:.4f} "
            f"val_mae={val_mae:.4f} best={best_val_metric:.4f} "
            f"time={epoch_time:.1f}s"
        )

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Val evaluation (always report)
    val_pred = evaluate_model(
        model, loader_dict["val"], entity_table,
        clamp_min, clamp_max, default_val=default_pred,
    )
    val_table = task.get_table("val")
    try:
        val_results = task.evaluate(val_pred, val_table)
        val_results = {k: float(v) for k, v in val_results.items()}
    except Exception as e:
        logger.warning(f"    Final val eval failed: {e}")
        val_results = {"mae": float("inf")}

    # Test evaluation (if available)
    test_results = None
    if test_available:
        try:
            test_pred = evaluate_model(
                model, loader_dict["test"], entity_table,
                clamp_min, clamp_max, default_val=default_pred,
            )
            test_table = task.get_table("test")
            test_results = task.evaluate(test_pred, test_table)
            test_results = {k: float(v) for k, v in test_results.items()}
        except Exception as e:
            logger.warning(f"    Test eval failed: {e}")
            test_results = None

    # Gate analysis (CAMA methods only)
    gate_stats = None
    if method_name != "mean_baseline":
        try:
            gate_stats = extract_gate_values(model, loader_dict["val"],
                                              entity_table, num_batches=20)
        except Exception as e:
            logger.warning(f"    Gate extraction failed: {e}")

    wall_clock = time.time() - t_total
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if HAS_GPU else 0
    if HAS_GPU:
        torch.cuda.reset_peak_memory_stats()

    logger.info(
        f"  FINAL [{method_name}] seed={seed}: "
        f"val_mae={val_results.get('mae', 'N/A'):.4f} "
        f"test_mae={test_results.get('mae', 'N/A') if test_results else 'N/A'} "
        f"best_epoch={best_epoch} wall={wall_clock:.0f}s"
    )

    # Cleanup
    del model, optimizer, best_state
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

    return {
        "val_mae": val_results.get("mae"),
        "val_r2": val_results.get("r2"),
        "val_rmse": val_results.get("rmse"),
        "test_mae": test_results.get("mae") if test_results else None,
        "test_r2": test_results.get("r2") if test_results else None,
        "test_rmse": test_results.get("rmse") if test_results else None,
        "best_epoch": best_epoch,
        "wall_clock_s": round(wall_clock, 1),
        "peak_vram_gb": round(peak_vram, 2),
        "gate_stats": gate_stats,
        "epoch_times": [round(t, 1) for t in epoch_times],
        "train_losses": [round(l, 6) for l in train_losses],
    }


# ===========================================================================
# SECTION 13: Output generation (exp_gen_sol_out.json schema)
# ===========================================================================

def build_output(
    results: Dict,
    configs: Dict,
    test_target_status: Dict,
    seeds: List[int],
) -> Dict:
    """Build method_out.json in exp_gen_sol_out schema format."""

    metadata = {
        "method_name": "CAMA 5-Seed Amazon + Avito Negative Init",
        "description": (
            "Two-part experiment: (A) 5-seed CAMA vs mean on rel-amazon/item-ltv, "
            "(B) 5-seed mean vs CAMA-default vs CAMA-negative-init on rel-avito/ad-ctr "
            "to fix 34% degradation by initializing gate_bias=-3."
        ),
        "methods": {
            "mean_baseline": "Standard mean aggregation",
            "cama_default": "CAMA with gate_bias=0 (iter-3 default)",
            "cama_neg_init": "CAMA with gate_bias=-3 (starts nearly closed)",
        },
        "hyperparameters": configs,
        "seeds": seeds,
        "test_target_status": test_target_status,
        "deviations": DEVIATIONS,
    }

    # Build results section
    results_section = {}

    # -- Part A: Amazon --
    if "rel-amazon" in results:
        amazon_res = results["rel-amazon"]
        part_a = {"task": "rel-amazon/item-ltv"}

        # Determine evaluation split
        amazon_test_avail = test_target_status.get("rel-amazon/item-ltv", {}).get("available", False)
        part_a["evaluation_split"] = "test" if amazon_test_avail else "val"
        metric_key = "test_mae" if amazon_test_avail else "val_mae"
        r2_key = "test_r2" if amazon_test_avail else "val_r2"

        part_a["per_method"] = {}
        for method_name in ["mean_baseline", "cama_default"]:
            if method_name not in amazon_res:
                continue
            method_data = amazon_res[method_name]
            maes = [method_data[s][metric_key] for s in seeds if s in method_data
                    and method_data[s][metric_key] is not None]
            r2s = [method_data[s][r2_key] for s in seeds if s in method_data
                   and method_data[s][r2_key] is not None]
            val_maes = [method_data[s]["val_mae"] for s in seeds if s in method_data
                        and method_data[s]["val_mae"] is not None]

            part_a["per_method"][method_name] = {
                "per_seed": {str(s): method_data[s] for s in seeds if s in method_data},
                "mean_mae": round(float(np.mean(maes)), 6) if maes else None,
                "std_mae": round(float(np.std(maes, ddof=1)), 6) if len(maes) > 1 else None,
                "mean_r2": round(float(np.mean(r2s)), 6) if r2s else None,
                "mean_val_mae": round(float(np.mean(val_maes)), 6) if val_maes else None,
            }

        # Statistical comparison
        mean_maes = [amazon_res["mean_baseline"][s][metric_key] for s in seeds
                     if s in amazon_res.get("mean_baseline", {})
                     and amazon_res["mean_baseline"][s][metric_key] is not None]
        cama_maes = [amazon_res["cama_default"][s][metric_key] for s in seeds
                     if s in amazon_res.get("cama_default", {})
                     and amazon_res["cama_default"][s][metric_key] is not None]

        if len(mean_maes) >= 2 and len(cama_maes) >= 2:
            part_a["statistical_comparison"] = {
                "cama_default_vs_mean": compute_statistics(mean_maes, cama_maes,
                                                           higher_is_better=False),
            }

        results_section["part_a_amazon"] = part_a

    # -- Part B: Avito --
    if "rel-avito" in results:
        avito_res = results["rel-avito"]
        part_b = {"task": "rel-avito/ad-ctr"}

        avito_test_avail = test_target_status.get("rel-avito/ad-ctr", {}).get("available", False)
        part_b["evaluation_split"] = "test" if avito_test_avail else "val"
        metric_key = "test_mae" if avito_test_avail else "val_mae"
        r2_key = "test_r2" if avito_test_avail else "val_r2"

        part_b["per_method"] = {}
        for method_name in ["mean_baseline", "cama_default", "cama_neg_init"]:
            if method_name not in avito_res:
                continue
            method_data = avito_res[method_name]
            maes = [method_data[s][metric_key] for s in seeds if s in method_data
                    and method_data[s][metric_key] is not None]
            r2s = [method_data[s][r2_key] for s in seeds if s in method_data
                   and method_data[s][r2_key] is not None]
            val_maes = [method_data[s]["val_mae"] for s in seeds if s in method_data
                        and method_data[s]["val_mae"] is not None]

            part_b["per_method"][method_name] = {
                "per_seed": {str(s): method_data[s] for s in seeds if s in method_data},
                "mean_mae": round(float(np.mean(maes)), 6) if maes else None,
                "std_mae": round(float(np.std(maes, ddof=1)), 6) if len(maes) > 1 else None,
                "mean_r2": round(float(np.mean(r2s)), 6) if r2s else None,
                "mean_val_mae": round(float(np.mean(val_maes)), 6) if val_maes else None,
            }

        # Statistical comparisons
        def _get_maes(method):
            return [avito_res[method][s][metric_key] for s in seeds
                    if s in avito_res.get(method, {})
                    and avito_res[method][s][metric_key] is not None]

        mean_maes = _get_maes("mean_baseline")
        cama_def_maes = _get_maes("cama_default")
        cama_neg_maes = _get_maes("cama_neg_init")

        part_b["statistical_comparison"] = {}
        if len(mean_maes) >= 2 and len(cama_def_maes) >= 2:
            part_b["statistical_comparison"]["cama_default_vs_mean"] = \
                compute_statistics(mean_maes, cama_def_maes, higher_is_better=False)
        if len(mean_maes) >= 2 and len(cama_neg_maes) >= 2:
            part_b["statistical_comparison"]["cama_neg_init_vs_mean"] = \
                compute_statistics(mean_maes, cama_neg_maes, higher_is_better=False)
        if len(cama_def_maes) >= 2 and len(cama_neg_maes) >= 2:
            part_b["statistical_comparison"]["cama_neg_init_vs_cama_default"] = \
                compute_statistics(cama_def_maes, cama_neg_maes, higher_is_better=False)

        # Safety check
        if mean_maes and cama_def_maes and cama_neg_maes:
            mean_mae_avg = float(np.mean(mean_maes))
            cama_def_avg = float(np.mean(cama_def_maes))
            cama_neg_avg = float(np.mean(cama_neg_maes))

            cama_def_pct = abs(cama_def_avg - mean_mae_avg) / max(mean_mae_avg, 1e-12) * 100
            cama_neg_pct = abs(cama_neg_avg - mean_mae_avg) / max(mean_mae_avg, 1e-12) * 100

            part_b["safety_check"] = {
                "cama_default_mae_pct_diff": round(cama_def_pct, 4),
                "cama_neg_init_mae_pct_diff": round(cama_neg_pct, 4),
                "neg_init_passes_safety": cama_neg_pct < 5.0,
            }

        # Gate analysis
        gate_analysis = {}
        for method_name in ["cama_default", "cama_neg_init"]:
            if method_name not in avito_res:
                continue
            all_gates = {}
            for s in seeds:
                if s not in avito_res[method_name]:
                    continue
                gs = avito_res[method_name][s].get("gate_stats")
                if gs:
                    for et, vals in gs.items():
                        if et not in all_gates:
                            all_gates[et] = []
                        all_gates[et].append(vals)
            merged = {}
            for et, val_list in all_gates.items():
                merged[et] = {
                    "mean_gate": round(float(np.mean([v['mean_gate'] for v in val_list])), 6),
                    "mean_N": round(float(np.mean([v['mean_N'] for v in val_list])), 2),
                    "max_N": round(float(np.max([v['max_N'] for v in val_list])), 2),
                    "mean_rho": round(float(np.mean([v['mean_rho'] for v in val_list])), 6),
                }
            gate_analysis[method_name] = merged

        part_b["gate_analysis"] = gate_analysis
        results_section["part_b_avito"] = part_b

    metadata["results"] = results_section

    # Build examples in exp_gen_sol_out schema format
    all_examples = []

    # Amazon examples
    if "rel-amazon" in results:
        amazon_examples = []
        amazon_test_avail = test_target_status.get("rel-amazon/item-ltv", {}).get("available", False)
        mk = "test_mae" if amazon_test_avail else "val_mae"

        for seed in seeds:
            baseline_r = results["rel-amazon"].get("mean_baseline", {}).get(seed, {})
            cama_r = results["rel-amazon"].get("cama_default", {}).get(seed, {})

            input_text = (
                f"=== CAMA Amazon Experiment ===\n"
                f"Dataset: rel-amazon\nTask: item-ltv (regression)\n"
                f"Seed: {seed}\n"
                f"Config: channels={configs.get('part_a', {}).get('channels', 128)}, "
                f"epochs={configs.get('part_a', {}).get('epochs', 10)}\n"
                f"Methods: mean_baseline vs cama_default (gate_bias=0)\n"
                f"Evaluation: {mk}"
            )
            output_text = json.dumps({
                "baseline_mae": baseline_r.get(mk),
                "cama_mae": cama_r.get(mk),
                "baseline_val_mae": baseline_r.get("val_mae"),
                "cama_val_mae": cama_r.get("val_mae"),
            })

            amazon_examples.append({
                "input": input_text,
                "output": output_text,
                "predict_baseline": json.dumps({
                    "mae": baseline_r.get(mk), "r2": baseline_r.get(mk.replace("mae", "r2"))
                }),
                "predict_cama_default": json.dumps({
                    "mae": cama_r.get(mk), "r2": cama_r.get(mk.replace("mae", "r2"))
                }),
                "metadata_seed": seed,
                "metadata_dataset": "rel-amazon",
                "metadata_task": "item-ltv",
                "metadata_task_type": "regression",
            })
        all_examples.append({"dataset": "rel-amazon__item-ltv", "examples": amazon_examples})

    # Avito examples
    if "rel-avito" in results:
        avito_examples = []
        avito_test_avail = test_target_status.get("rel-avito/ad-ctr", {}).get("available", False)
        mk = "test_mae" if avito_test_avail else "val_mae"

        for seed in seeds:
            baseline_r = results["rel-avito"].get("mean_baseline", {}).get(seed, {})
            cama_def_r = results["rel-avito"].get("cama_default", {}).get(seed, {})
            cama_neg_r = results["rel-avito"].get("cama_neg_init", {}).get(seed, {})

            input_text = (
                f"=== CAMA Avito Experiment ===\n"
                f"Dataset: rel-avito\nTask: ad-ctr (regression)\n"
                f"Seed: {seed}\n"
                f"Config: channels={configs.get('part_b', {}).get('channels', 64)}, "
                f"epochs={configs.get('part_b', {}).get('epochs', 10)}\n"
                f"Methods: mean_baseline vs cama_default (gb=0) vs cama_neg_init (gb=-3)\n"
                f"Evaluation: {mk}"
            )
            output_text = json.dumps({
                "baseline_mae": baseline_r.get(mk),
                "cama_default_mae": cama_def_r.get(mk),
                "cama_neg_init_mae": cama_neg_r.get(mk),
            })

            avito_examples.append({
                "input": input_text,
                "output": output_text,
                "predict_baseline": json.dumps({
                    "mae": baseline_r.get(mk), "r2": baseline_r.get(mk.replace("mae", "r2"))
                }),
                "predict_cama_default": json.dumps({
                    "mae": cama_def_r.get(mk), "r2": cama_def_r.get(mk.replace("mae", "r2"))
                }),
                "predict_cama_neg_init": json.dumps({
                    "mae": cama_neg_r.get(mk), "r2": cama_neg_r.get(mk.replace("mae", "r2"))
                }),
                "metadata_seed": seed,
                "metadata_dataset": "rel-avito",
                "metadata_task": "ad-ctr",
                "metadata_task_type": "regression",
            })
        all_examples.append({"dataset": "rel-avito__ad-ctr", "examples": avito_examples})

    return {
        "metadata": metadata,
        "datasets": all_examples,
    }


# ===========================================================================
# SECTION 14: Main
# ===========================================================================

@logger.catch
def main():
    parser = ArgumentParser()
    parser.add_argument("--phase", choices=["mini", "smoke", "full"], default="mini")
    args = parser.parse_args()

    logger.info(f"=== CAMA Amazon + Avito Experiment (phase={args.phase}) ===")

    # ── Cache directory ──────────────────────────────────────────────────
    # Try known cache locations
    cache_candidates = [
        str(WS / "relbench_cache"),
        "/ai-inventor/aii_pipeline/runs/leskovec-predictive-residual-message-passing-v2_sti/"
        "3_invention_loop/iter_1/gen_art/data_id3_it1__opus/relbench_cache",
        str(Path.home() / ".cache" / "relbench"),
    ]
    CACHE_DIR = str(WS / "relbench_cache")
    for c in cache_candidates:
        if Path(c).exists():
            CACHE_DIR = c
            break
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["RELBENCH_CACHE_DIR"] = CACHE_DIR
    logger.info(f"Cache dir: {CACHE_DIR}")

    # ── Phase configs ────────────────────────────────────────────────────
    if args.phase == "mini":
        SEEDS = [42]
        part_a_cfg = {
            "channels": 128, "lr": 0.005, "epochs": 2, "batch_size": 512,
            "num_neighbors": [128, 128], "max_steps_per_epoch": 50,
            "grad_clip": 1.0, "methods": ["mean_baseline", "cama_default"],
        }
        part_b_cfg = {
            "channels": 64, "lr": 0.005, "epochs": 2, "batch_size": 256,
            "num_neighbors": [64, 64], "max_steps_per_epoch": 50,
            "grad_clip": 1.0,
            "methods": ["mean_baseline", "cama_default", "cama_neg_init"],
        }
    elif args.phase == "smoke":
        SEEDS = [42]
        part_a_cfg = {
            "channels": 128, "lr": 0.005, "epochs": 2, "batch_size": 512,
            "num_neighbors": [128, 128], "max_steps_per_epoch": 200,
            "grad_clip": 1.0, "methods": ["mean_baseline", "cama_default"],
        }
        part_b_cfg = {
            "channels": 64, "lr": 0.005, "epochs": 2, "batch_size": 256,
            "num_neighbors": [64, 64], "max_steps_per_epoch": 200,
            "grad_clip": 1.0,
            "methods": ["mean_baseline", "cama_default", "cama_neg_init"],
        }
    else:  # full
        # Using 5 epochs instead of 10 due to time constraints (plan fallback mode 7).
        # 5 seeds × (2+3) methods × 5 epochs × 2000 steps ≈ 3h total.
        # DEVIATION: reduced epochs from 10 to 5 for time budget.
        SEEDS = [42, 123, 456, 789, 1024]
        part_a_cfg = {
            "channels": 128, "lr": 0.005, "epochs": 5, "batch_size": 512,
            "num_neighbors": [128, 128], "max_steps_per_epoch": 2000,
            "grad_clip": 1.0, "methods": ["mean_baseline", "cama_default"],
        }
        part_b_cfg = {
            "channels": 64, "lr": 0.005, "epochs": 5, "batch_size": 256,
            "num_neighbors": [64, 64], "max_steps_per_epoch": 2000,
            "grad_clip": 1.0,
            "methods": ["mean_baseline", "cama_default", "cama_neg_init"],
        }
        DEVIATIONS.append("Reduced epochs from 10 to 5 (plan fallback mode 7: time constraints)")

    configs = {"part_a": part_a_cfg, "part_b": part_b_cfg, "seeds": SEEDS}
    logger.info(f"Seeds: {SEEDS}")

    # ── Unit tests ───────────────────────────────────────────────────────
    run_unit_tests()

    # ── Test target investigation ────────────────────────────────────────
    test_target_status = {}
    for ds_name, task_name in [("rel-amazon", "item-ltv"), ("rel-avito", "ad-ctr")]:
        key = f"{ds_name}/{task_name}"
        try:
            test_target_status[key] = investigate_test_targets(
                ds_name, task_name, CACHE_DIR)
        except Exception as e:
            logger.warning(f"Could not investigate test targets for {key}: {e}")
            test_target_status[key] = {"available": False, "error": str(e)}

    all_results: Dict[str, Dict[str, Dict[int, Dict]]] = {}

    # ═══════════════════════════════════════════════════════════════════════
    # PART A: Amazon item-ltv
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PART A: rel-amazon/item-ltv")
    logger.info("=" * 60)

    try:
        amazon_data, amazon_col_stats = load_amazon_data(CACHE_DIR)
        amazon_loaders, amazon_task = build_loaders(
            "rel-amazon", "item-ltv", amazon_data,
            part_a_cfg["num_neighbors"], part_a_cfg["batch_size"], CACHE_DIR,
        )

        amazon_test_avail = test_target_status.get(
            "rel-amazon/item-ltv", {}).get("available", False)

        all_results["rel-amazon"] = {}
        for method in part_a_cfg["methods"]:
            all_results["rel-amazon"][method] = {}
            for seed in SEEDS:
                try:
                    result = run_single(
                        data=amazon_data,
                        col_stats_dict=amazon_col_stats,
                        loader_dict=amazon_loaders,
                        task=amazon_task,
                        method_name=method,
                        seed=seed,
                        channels=part_a_cfg["channels"],
                        lr=part_a_cfg["lr"],
                        epochs=part_a_cfg["epochs"],
                        max_steps=part_a_cfg["max_steps_per_epoch"],
                        grad_clip=part_a_cfg["grad_clip"],
                        test_available=amazon_test_avail,
                    )
                    all_results["rel-amazon"][method][seed] = result
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM on Amazon {method} seed={seed}, reducing batch size")
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Retry with reduced config
                    try:
                        reduced_loaders, _ = build_loaders(
                            "rel-amazon", "item-ltv", amazon_data,
                            [64, 64], 256, CACHE_DIR,
                        )
                        DEVIATIONS.append(
                            f"Reduced Amazon batch_size=256, neighbors=[64,64] for {method} seed={seed}"
                        )
                        result = run_single(
                            data=amazon_data,
                            col_stats_dict=amazon_col_stats,
                            loader_dict=reduced_loaders,
                            task=amazon_task,
                            method_name=method,
                            seed=seed,
                            channels=part_a_cfg["channels"],
                            lr=part_a_cfg["lr"],
                            epochs=part_a_cfg["epochs"],
                            max_steps=part_a_cfg["max_steps_per_epoch"],
                            grad_clip=part_a_cfg["grad_clip"],
                            test_available=amazon_test_avail,
                        )
                        all_results["rel-amazon"][method][seed] = result
                    except Exception as e2:
                        logger.exception(f"Retry also failed for Amazon {method} seed={seed}")
                        all_results["rel-amazon"][method][seed] = {
                            "val_mae": None, "val_r2": None, "val_rmse": None,
                            "test_mae": None, "test_r2": None, "test_rmse": None,
                            "best_epoch": 0, "wall_clock_s": 0, "peak_vram_gb": 0,
                            "gate_stats": None, "epoch_times": [], "train_losses": [],
                            "error": str(e2),
                        }
                except Exception as e:
                    logger.exception(f"Failed Amazon {method} seed={seed}")
                    all_results["rel-amazon"][method][seed] = {
                        "val_mae": None, "val_r2": None, "val_rmse": None,
                        "test_mae": None, "test_r2": None, "test_rmse": None,
                        "best_epoch": 0, "wall_clock_s": 0, "peak_vram_gb": 0,
                        "gate_stats": None, "epoch_times": [], "train_losses": [],
                        "error": str(e),
                    }

        # Free Amazon data
        del amazon_data, amazon_col_stats, amazon_loaders
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()
        log_memory("after Amazon cleanup")

    except Exception as e:
        logger.exception(f"PART A failed entirely: {e}")
        DEVIATIONS.append(f"Part A (Amazon) failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # PART B: Avito ad-ctr
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PART B: rel-avito/ad-ctr")
    logger.info("=" * 60)

    try:
        avito_data, avito_col_stats = load_avito_data(CACHE_DIR)
        avito_loaders, avito_task = build_loaders(
            "rel-avito", "ad-ctr", avito_data,
            part_b_cfg["num_neighbors"], part_b_cfg["batch_size"], CACHE_DIR,
        )

        avito_test_avail = test_target_status.get(
            "rel-avito/ad-ctr", {}).get("available", False)

        all_results["rel-avito"] = {}
        for method in part_b_cfg["methods"]:
            all_results["rel-avito"][method] = {}
            for seed in SEEDS:
                try:
                    result = run_single(
                        data=avito_data,
                        col_stats_dict=avito_col_stats,
                        loader_dict=avito_loaders,
                        task=avito_task,
                        method_name=method,
                        seed=seed,
                        channels=part_b_cfg["channels"],
                        lr=part_b_cfg["lr"],
                        epochs=part_b_cfg["epochs"],
                        max_steps=part_b_cfg["max_steps_per_epoch"],
                        grad_clip=part_b_cfg["grad_clip"],
                        test_available=avito_test_avail,
                    )
                    all_results["rel-avito"][method][seed] = result
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM on Avito {method} seed={seed}, reducing batch size")
                    torch.cuda.empty_cache()
                    gc.collect()
                    try:
                        reduced_loaders, _ = build_loaders(
                            "rel-avito", "ad-ctr", avito_data,
                            [32, 32], 128, CACHE_DIR,
                        )
                        DEVIATIONS.append(
                            f"Reduced Avito batch_size=128, neighbors=[32,32] for {method} seed={seed}"
                        )
                        result = run_single(
                            data=avito_data,
                            col_stats_dict=avito_col_stats,
                            loader_dict=reduced_loaders,
                            task=avito_task,
                            method_name=method,
                            seed=seed,
                            channels=part_b_cfg["channels"],
                            lr=part_b_cfg["lr"],
                            epochs=part_b_cfg["epochs"],
                            max_steps=part_b_cfg["max_steps_per_epoch"],
                            grad_clip=part_b_cfg["grad_clip"],
                            test_available=avito_test_avail,
                        )
                        all_results["rel-avito"][method][seed] = result
                    except Exception as e2:
                        logger.exception(f"Retry also failed for Avito {method} seed={seed}")
                        all_results["rel-avito"][method][seed] = {
                            "val_mae": None, "val_r2": None, "val_rmse": None,
                            "test_mae": None, "test_r2": None, "test_rmse": None,
                            "best_epoch": 0, "wall_clock_s": 0, "peak_vram_gb": 0,
                            "gate_stats": None, "epoch_times": [], "train_losses": [],
                            "error": str(e2),
                        }
                except Exception as e:
                    logger.exception(f"Failed Avito {method} seed={seed}")
                    all_results["rel-avito"][method][seed] = {
                        "val_mae": None, "val_r2": None, "val_rmse": None,
                        "test_mae": None, "test_r2": None, "test_rmse": None,
                        "best_epoch": 0, "wall_clock_s": 0, "peak_vram_gb": 0,
                        "gate_stats": None, "epoch_times": [], "train_losses": [],
                        "error": str(e),
                    }

        # Free Avito data
        del avito_data, avito_col_stats, avito_loaders
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()

    except Exception as e:
        logger.exception(f"PART B failed entirely: {e}")
        DEVIATIONS.append(f"Part B (Avito) failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # Build and save output
    # ═══════════════════════════════════════════════════════════════════════
    output = build_output(all_results, configs, test_target_status, SEEDS)

    out_path = WS / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    meta = output.get("metadata", {})
    res = meta.get("results", {})

    if "part_a_amazon" in res:
        pa = res["part_a_amazon"]
        for m in ["mean_baseline", "cama_default"]:
            if m in pa.get("per_method", {}):
                pm = pa["per_method"][m]
                logger.info(f"  Amazon {m}: MAE={pm.get('mean_mae', 'N/A')} "
                            f"+/- {pm.get('std_mae', 'N/A')}")
        if "statistical_comparison" in pa:
            sc = pa["statistical_comparison"].get("cama_default_vs_mean", {})
            logger.info(f"  Amazon Cohen's d: {sc.get('cohens_d', 'N/A')}")

    if "part_b_avito" in res:
        pb = res["part_b_avito"]
        for m in ["mean_baseline", "cama_default", "cama_neg_init"]:
            if m in pb.get("per_method", {}):
                pm = pb["per_method"][m]
                logger.info(f"  Avito {m}: MAE={pm.get('mean_mae', 'N/A')} "
                            f"+/- {pm.get('std_mae', 'N/A')}")
        if "safety_check" in pb:
            sc = pb["safety_check"]
            logger.info(f"  Avito safety: cama_default_pct_diff={sc.get('cama_default_mae_pct_diff', 'N/A')}%")
            logger.info(f"  Avito safety: cama_neg_init_pct_diff={sc.get('cama_neg_init_mae_pct_diff', 'N/A')}%")
            logger.info(f"  Avito safety: neg_init_passes={sc.get('neg_init_passes_safety', 'N/A')}")

    if DEVIATIONS:
        logger.info(f"  Deviations: {DEVIATIONS}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
