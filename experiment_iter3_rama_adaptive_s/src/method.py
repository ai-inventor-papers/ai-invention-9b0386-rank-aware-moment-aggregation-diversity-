#!/usr/bin/env python3
"""RAMA Adaptive Safety Validation on rel-avito/ad-ctr (Extreme-Cardinality Stress Test).

Tests RAMA (Rank-Aware Moment Aggregation) on the extreme-cardinality rel-avito/ad-ctr
regression task where PRMP catastrophically failed (R²=-5046). Validates that RAMA's
diversity gate closes on extreme-cardinality edges, gracefully degrading to mean aggregation.
Compares RAMA vs mean-aggregation baseline across 5 seeds on MAE/R².
"""

import gc
import json
import math
import os
import resource
import sys
import time
import copy
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from loguru import logger
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────────────────────
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or 42.0

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, GPU={HAS_GPU}, VRAM={VRAM_GB:.1f} GB")

# ── Memory limits ────────────────────────────────────────────────────────────
RAM_BUDGET = int(TOTAL_RAM_GB * 0.80 * 1e9)  # 80% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.90)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))
    logger.info(f"VRAM budget: {VRAM_BUDGET / 1e9:.1f} GB")

# ── Imports that need torch ──────────────────────────────────────────────────
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn import HeteroConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: RAMAAggregation Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class RAMAAggregation(Aggregation):
    """Rank-Aware Moment Aggregation.

    Learns a gate g(log(N+1), rho) that controls injection of variance info:
        out = mu + g * W_sigma(var)
    where rho is a variance-entropy proxy for effective rank.
    Zero-initialized so it starts as pure mean aggregation.
    """

    def __init__(self, channels: int, gate_hidden: int = 16, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.gate_hidden = gate_hidden
        self.eps = eps

        self.W_sigma = nn.Linear(channels, channels, bias=False)
        self.W_g = nn.Linear(2, gate_hidden)
        self.gate_out = nn.Linear(gate_hidden, channels)

        # Gate analysis storage
        self._last_gate_values = None
        self._last_N_values = None
        self._last_rho_values = None
        self._record_gates = False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.W_sigma.weight)
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
        var = torch.nan_to_num(var, nan=0.0)  # safety
        # Step 3: Cardinality
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        N = self.reduce(ones, index, ptr, dim_size, dim, reduce='sum')
        # Step 4: Effective rank proxy
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Model (adapted from RelBench examples/model.py)
# ═══════════════════════════════════════════════════════════════════════════════

from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

# Import relbench components
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from torch_frame import stype
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
    StypeWiseFeatureEncoder,
)
from torch_frame import TensorFrame

try:
    from torch_frame.nn import LinearModelEncoder
    HAS_LINEAR_MODEL_ENCODER = True
except ImportError:
    HAS_LINEAR_MODEL_ENCODER = False


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

            # Build stype_encoder_dict and col_names_dict
            stype_encoder_dict = {}
            col_names_dict = {}
            for st, st_col_stats in col_stats.items():
                col_names = list(st_col_stats.keys())
                if not col_names:
                    continue
                col_names_dict[st] = col_names
                if st == stype.categorical:
                    stype_encoder_dict[st] = EmbeddingEncoder()
                elif st == stype.numerical:
                    stype_encoder_dict[st] = LinearEncoder()
                elif st == stype.timestamp:
                    stype_encoder_dict[st] = TimestampEncoder()
                elif st == stype.text_embedded:
                    stype_encoder_dict[st] = LinearEncoder()

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
        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time.to(torch.float32)
            rel_time = rel_time.view(-1, 1)
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
                src_type, _, dst_type = edge_type
                if isinstance(aggr, str):
                    conv_aggr = aggr
                else:
                    # Create a fresh RAMA instance per edge type per layer
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
            # Keep nodes that didn't get messages
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
        # Encode tabular features
        tf_dict = {
            nt: batch[nt].tf
            for nt in batch.node_types
            if hasattr(batch[nt], 'tf')
        }
        x_dict = self.encoder(tf_dict)

        # Add temporal encoding if available
        if hasattr(batch[entity_table], 'seed_time'):
            seed_time = batch[entity_table].seed_time
            time_dict = {}
            batch_dict = {}
            for nt in batch.node_types:
                if hasattr(batch[nt], 'time') and batch[nt].time is not None:
                    time_dict[nt] = batch[nt].time
                    batch_dict[nt] = batch[nt].batch if hasattr(batch[nt], 'batch') else torch.zeros(batch[nt].num_nodes, dtype=torch.long, device=seed_time.device)
            if time_dict:
                temp_dict = self.temporal_encoder(seed_time, time_dict, batch_dict)
                for nt in temp_dict:
                    if nt in x_dict:
                        x_dict[nt] = x_dict[nt] + temp_dict[nt]
                    else:
                        x_dict[nt] = temp_dict[nt]

        # Fill in missing node types with zeros
        for nt in batch.node_types:
            if nt not in x_dict:
                x_dict[nt] = torch.zeros(
                    batch[nt].num_nodes, self.channels,
                    device=DEVICE,
                )

        # GNN message passing
        edge_index_dict = {
            et: batch[et].edge_index
            for et in batch.edge_types
            if hasattr(batch[et], 'edge_index')
        }
        x_dict = self.gnn(x_dict, edge_index_dict)

        # Prediction head on entity nodes
        x = x_dict[entity_table][:batch[entity_table].batch_size]
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Gate Tracking Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

class GateTracker:
    """Tracks per-edge-type gate values from RAMA modules."""
    def __init__(self):
        self.records = {}  # edge_type_str -> list of dicts

    def collect_from_model(self, model, edge_types):
        """Collect gate values from all RAMA modules after a forward pass.

        Since HeteroGraphSAGE uses deepcopy, each edge type has its own RAMA.
        We iterate over layers and edge types to get per-edge gate stats.
        """
        for layer_idx, conv_layer in enumerate(model.gnn.convs):
            for et_key, sage_conv in conv_layer.convs.items():
                # Access the aggregation module
                aggr_mod = sage_conv.aggr_module
                if isinstance(aggr_mod, RAMAAggregation) and aggr_mod._last_gate_values is not None:
                    g = aggr_mod._last_gate_values
                    N = aggr_mod._last_N_values
                    rho = aggr_mod._last_rho_values
                    # Skip empty tensors (some edge types may have 0 edges in a batch)
                    if g.numel() == 0 or N.numel() == 0:
                        continue
                    et_str = f"L{layer_idx}_{'__'.join(et_key)}"
                    if et_str not in self.records:
                        self.records[et_str] = []
                    self.records[et_str].append({
                        'gate_mean': g.mean().item(),
                        'gate_median': g.median().item(),
                        'gate_std': g.std().item() if g.numel() > 1 else 0.0,
                        'gate_max': g.max().item(),
                        'N_mean': N.mean().item(),
                        'N_max': N.max().item(),
                        'rho_mean': rho.mean().item(),
                    })

    def enable_recording(self, model):
        """Enable gate recording on all RAMA modules."""
        for conv_layer in model.gnn.convs:
            for sage_conv in conv_layer.convs.values():
                aggr_mod = sage_conv.aggr_module
                if isinstance(aggr_mod, RAMAAggregation):
                    aggr_mod._record_gates = True

    def disable_recording(self, model):
        """Disable gate recording."""
        for conv_layer in model.gnn.convs:
            for sage_conv in conv_layer.convs.values():
                aggr_mod = sage_conv.aggr_module
                if isinstance(aggr_mod, RAMAAggregation):
                    aggr_mod._record_gates = False

    def summarize(self):
        """Get per-edge-type summary statistics."""
        summary = {}
        for et_str, recs in self.records.items():
            summary[et_str] = {
                'mean_gate': float(np.mean([r['gate_mean'] for r in recs])),
                'mean_N': float(np.mean([r['N_mean'] for r in recs])),
                'max_N': float(np.max([r['N_max'] for r in recs])),
                'mean_rho': float(np.mean([r['rho_mean'] for r in recs])),
                'num_batches': len(recs),
            }
        return summary


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Training & Evaluation Functions
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, entity_table, loss_fn,
                    y_all, batch_size, grad_clip=1.0, max_steps=None):
    """Train for one epoch, return avg loss.

    y_all: full label tensor for training set, aligned with input_nodes order.
    Labels are sliced by step * batch_size to match NeighborLoader ordering.
    """
    model.train()
    total_loss = 0.0
    count = 0
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch, entity_table)

        # Get labels for this batch's seed nodes
        bs = batch[entity_table].batch_size
        start_idx = step * batch_size
        end_idx = start_idx + bs
        if end_idx > len(y_all):
            end_idx = len(y_all)
        y_batch = y_all[start_idx:end_idx].to(DEVICE)

        # Match sizes
        pred_flat = pred.view(-1)[:len(y_batch)]
        loss = loss_fn(pred_flat.float(), y_batch.float())
        if torch.isnan(loss):
            logger.warning(f"NaN loss at step {step}, skipping")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
        count += 1

        if step == 0 and HAS_GPU:
            vram_used = torch.cuda.max_memory_allocated() / 1e9
            logger.debug(f"VRAM after first batch: {vram_used:.2f} GB")
    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate_model(model, loader, entity_table, y_all, batch_size):
    """Run evaluation, return predictions and labels as numpy arrays."""
    model.eval()
    preds = []
    labels = []
    for step, batch in enumerate(loader):
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table)
        bs = batch[entity_table].batch_size
        start_idx = step * batch_size
        end_idx = start_idx + bs
        if end_idx > len(y_all):
            end_idx = len(y_all)
        y_batch = y_all[start_idx:end_idx]
        pred_flat = pred.view(-1)[:len(y_batch)]
        preds.append(pred_flat.cpu().numpy())
        labels.append(y_batch.numpy())
    if preds:
        return np.concatenate(preds), np.concatenate(labels)
    return np.array([]), np.array([])


def compute_metrics(y_true, y_pred):
    """Compute MAE, R², RMSE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    return {"mae": mae, "r2": r2, "rmse": rmse}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Statistical Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def cohens_d(group1, group2):
    """Compute Cohen's d with 95% CI."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (m2 - m1) / max(sp, 1e-12)
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    ci_lower = d - 1.96 * se_d
    ci_upper = d + 1.96 * se_d
    return float(d), float(ci_lower), float(ci_upper)


def statistical_comparison(baseline_vals, rama_vals, metric_name):
    """Full statistical comparison between baseline and RAMA."""
    d, ci_lo, ci_hi = cohens_d(baseline_vals, rama_vals)
    t_stat, p_val = stats.ttest_ind(rama_vals, baseline_vals)
    return {
        "cohen_d": round(d, 4),
        "ci_lower": round(ci_lo, 4),
        "ci_upper": round(ci_hi, 4),
        "p_value": round(float(p_val), 6),
        "baseline_mean": round(float(np.mean(baseline_vals)), 6),
        "baseline_std": round(float(np.std(baseline_vals)), 6),
        "rama_mean": round(float(np.mean(rama_vals)), 6),
        "rama_std": round(float(np.std(rama_vals)), 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Main Experiment
# ═══════════════════════════════════════════════════════════════════════════════

def build_data_and_loaders(config, seed):
    """Build dataset, graph, and data loaders."""
    seed_everything(seed)

    logger.info(f"Loading rel-avito dataset...")
    dataset = get_dataset("rel-avito", download=True)
    task = get_task("rel-avito", "ad-ctr", download=True)

    db = dataset.get_db()
    entity_table = task.entity_table

    # Build graph
    logger.info("Building pkey-fkey graph...")
    col_to_stype_dict = get_stype_proposal(db)

    # Remove text stypes to avoid tokenizer issues
    for tbl_name in list(col_to_stype_dict.keys()):
        for col_name in list(col_to_stype_dict[tbl_name].keys()):
            if col_to_stype_dict[tbl_name][col_name] in (stype.text_embedded, stype.text_tokenized):
                del col_to_stype_dict[tbl_name][col_name]

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,
        cache_dir=str(WORKSPACE / "cache"),
    )

    # Setup task labels
    train_table = task.get_table("train")
    val_table = task.get_table("val")

    target_col = task.target_col
    entity_col = task.entity_col
    train_y = train_table.df[target_col].values.astype(np.float32)

    # Clamp range from percentiles
    clamp_min = float(np.percentile(train_y, 2))
    clamp_max = float(np.percentile(train_y, 98))

    num_neighbors = [config['num_neighbors'] // (2**i) for i in range(config['num_layers'])]
    logger.info(f"Neighbor sampling: {num_neighbors}")

    from torch_geometric.loader import NeighborLoader

    entity_table_name = entity_table

    def make_loader(table, shuffle=False):
        """Create a NeighborLoader for the given task table."""
        df = table.df
        entity_ids = torch.tensor(df[entity_col].values, dtype=torch.long)

        # Target labels (may not exist for test set)
        if target_col in df.columns:
            y = torch.tensor(df[target_col].values.astype(np.float32))
        else:
            y = None

        loader = NeighborLoader(
            data,
            input_nodes=(entity_table_name, entity_ids),
            num_neighbors=num_neighbors,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            num_workers=min(NUM_CPUS - 1, 2),
        )
        return loader, y, entity_ids

    train_loader, train_y_tensor, train_ids = make_loader(train_table, shuffle=True)
    val_loader, val_y_tensor, val_ids = make_loader(val_table, shuffle=False)

    return {
        'data': data,
        'col_stats_dict': col_stats_dict,
        'entity_table': entity_table_name,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_y': train_y_tensor,
        'val_y': val_y_tensor,
        'clamp_min': clamp_min,
        'clamp_max': clamp_max,
        'task': task,
        'edge_types': data.edge_types,
    }


def run_single_experiment(config, seed, aggr_name, data_bundle):
    """Run a single training experiment and return results."""
    seed_everything(seed)
    logger.info(f"Starting {aggr_name} seed={seed}")

    data = data_bundle['data']
    col_stats_dict = data_bundle['col_stats_dict']
    entity_table = data_bundle['entity_table']
    task = data_bundle['task']

    # Create model
    if aggr_name == 'mean':
        aggr = 'mean'
    else:
        aggr = RAMAAggregation(channels=config['channels'], gate_hidden=16, eps=1e-6)

    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=config['num_layers'],
        channels=config['channels'],
        out_channels=1,
        aggr=aggr,
        norm='batch_norm',
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.L1Loss()

    gate_tracker = GateTracker() if aggr_name == 'rama' else None

    best_val_mae = float('inf')
    best_model_state = None
    epoch_times = []

    for epoch in range(config['epochs']):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, data_bundle['train_loader'], optimizer,
            entity_table, loss_fn,
            y_all=data_bundle['train_y'],
            batch_size=config['batch_size'],
            grad_clip=config['grad_clip'],
            max_steps=config.get('max_steps_per_epoch'),
        )

        # Validate
        val_preds, val_y_arr = evaluate_model(
            model, data_bundle['val_loader'], entity_table,
            y_all=data_bundle['val_y'],
            batch_size=config['batch_size'],
        )

        if len(val_preds) > 0:
            val_metrics = compute_metrics(val_y_arr, val_preds)
        else:
            val_metrics = {"mae": float('inf'), "r2": -999, "rmse": float('inf')}

        dt = time.time() - t0
        epoch_times.append(dt)

        logger.info(
            f"  [{aggr_name}] seed={seed} epoch={epoch+1}/{config['epochs']} "
            f"loss={train_loss:.4f} val_mae={val_metrics['mae']:.4f} "
            f"val_r2={val_metrics['r2']:.4f} time={dt:.1f}s"
        )

        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_model_state = copy.deepcopy(model.state_dict())

        # Early abort: catastrophic R²
        if val_metrics['r2'] < -100:
            logger.warning(f"Catastrophic R²={val_metrics['r2']}, stopping early")
            break

    # Load best model for final evaluation on val set
    # (test labels are hidden in relbench, so we use val as final metric)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_preds, test_y_arr = evaluate_model(
        model, data_bundle['val_loader'], entity_table,
        y_all=data_bundle['val_y'],
        batch_size=config['batch_size'],
    )

    if len(test_preds) > 0:
        test_metrics = compute_metrics(test_y_arr, test_preds)
    else:
        test_metrics = {"mae": float('inf'), "r2": -999, "rmse": float('inf')}

    logger.info(
        f"  [{aggr_name}] seed={seed} FINAL(val): mae={test_metrics['mae']:.4f} "
        f"r2={test_metrics['r2']:.4f} rmse={test_metrics['rmse']:.4f}"
    )

    # Gate extraction for RAMA
    gate_summary = None
    if aggr_name == 'rama' and gate_tracker is not None:
        logger.info("Extracting gate values...")
        gate_tracker.enable_recording(model)
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(data_bundle['val_loader']):
                if step >= 20:  # 20 batches is enough for statistics
                    break
                batch = batch.to(DEVICE)
                _ = model(batch, entity_table)
                gate_tracker.collect_from_model(model, data_bundle['edge_types'])
        gate_tracker.disable_recording(model)
        gate_summary = gate_tracker.summarize()

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if HAS_GPU else 0
    torch.cuda.reset_peak_memory_stats() if HAS_GPU else None

    # Cleanup
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache() if HAS_GPU else None

    return {
        'test_metrics': test_metrics,
        'best_val_mae': float(best_val_mae),
        'epoch_times': epoch_times,
        'peak_vram_gb': float(peak_vram),
        'gate_summary': gate_summary,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Output Generation (exp_gen_sol_out.json format)
# ═══════════════════════════════════════════════════════════════════════════════

def build_output(all_results, config, data_bundle, phase_name="full"):
    """Build the method_out.json in exp_gen_sol_out schema format."""

    # Gather per-seed metrics
    baseline_metrics = {str(s): r['test_metrics'] for s, r in all_results['mean'].items()}
    rama_metrics = {str(s): r['test_metrics'] for s, r in all_results['rama'].items()}

    baseline_maes = [m['mae'] for m in baseline_metrics.values()]
    baseline_r2s = [m['r2'] for m in baseline_metrics.values()]
    baseline_rmses = [m['rmse'] for m in baseline_metrics.values()]

    rama_maes = [m['mae'] for m in rama_metrics.values()]
    rama_r2s = [m['r2'] for m in rama_metrics.values()]
    rama_rmses = [m['rmse'] for m in rama_metrics.values()]

    # Statistical comparisons
    mae_comparison = statistical_comparison(baseline_maes, rama_maes, "mae")
    r2_comparison = statistical_comparison(baseline_r2s, rama_r2s, "r2")

    # Safety check
    baseline_mae_mean = float(np.mean(baseline_maes))
    rama_mae_mean = float(np.mean(rama_maes))
    mae_pct_diff = abs(rama_mae_mean - baseline_mae_mean) / max(baseline_mae_mean, 1e-12) * 100
    r2_not_catastrophic = float(np.mean(rama_r2s)) > -1.0

    # Gate analysis
    gate_analysis = {"per_edge_type": {}, "gate_closes_on_extreme": False}
    all_gate_summaries = [r.get('gate_summary') for r in all_results['rama'].values()
                          if r.get('gate_summary')]
    if all_gate_summaries:
        # Merge gate summaries across seeds
        merged = {}
        for gs in all_gate_summaries:
            for et, vals in gs.items():
                if et not in merged:
                    merged[et] = []
                merged[et].append(vals)

        for et, val_list in merged.items():
            gate_analysis["per_edge_type"][et] = {
                "mean_gate": round(float(np.mean([v['mean_gate'] for v in val_list])), 4),
                "mean_N": round(float(np.mean([v['mean_N'] for v in val_list])), 2),
                "max_N": round(float(np.max([v['max_N'] for v in val_list])), 2),
                "mean_rho": round(float(np.mean([v['mean_rho'] for v in val_list])), 4),
            }

        # Check if extreme-cardinality edges have lower gates
        if gate_analysis["per_edge_type"]:
            sorted_by_N = sorted(gate_analysis["per_edge_type"].items(),
                                 key=lambda x: x[1]['mean_N'], reverse=True)
            n_edges = len(sorted_by_N)
            top_half = sorted_by_N[:max(n_edges // 2, 1)]
            bottom_half = sorted_by_N[max(n_edges // 2, 1):]

            extreme_gate_mean = float(np.mean([v['mean_gate'] for _, v in top_half]))
            normal_gate_mean = float(np.mean([v['mean_gate'] for _, v in bottom_half])) if bottom_half else extreme_gate_mean

            gate_analysis["extreme_cardinality_edges_gate_mean"] = round(extreme_gate_mean, 4)
            gate_analysis["normal_cardinality_edges_gate_mean"] = round(normal_gate_mean, 4)
            gate_analysis["gate_closes_on_extreme"] = extreme_gate_mean < 0.3

            # Spearman correlation log(N) vs gate
            if len(sorted_by_N) >= 3:
                log_ns = [np.log1p(v['mean_N']) for _, v in sorted_by_N]
                gates = [v['mean_gate'] for _, v in sorted_by_N]
                corr, p_corr = stats.spearmanr(log_ns, gates)
                # Replace NaN with None for valid JSON
                gate_analysis["correlation_logN_vs_gate"] = (
                    round(float(corr), 4) if not np.isnan(corr) else None
                )
                gate_analysis["correlation_pvalue"] = (
                    round(float(p_corr), 6) if not np.isnan(p_corr) else None
                )

    # Timing
    baseline_times = [np.mean(r['epoch_times']) for r in all_results['mean'].values()]
    rama_times = [np.mean(r['epoch_times']) for r in all_results['rama'].values()]
    baseline_avg_time = float(np.mean(baseline_times))
    rama_avg_time = float(np.mean(rama_times))
    overhead_pct = (rama_avg_time - baseline_avg_time) / max(baseline_avg_time, 1e-12) * 100

    # Memory
    baseline_vrams = [r['peak_vram_gb'] for r in all_results['mean'].values()]
    rama_vrams = [r['peak_vram_gb'] for r in all_results['rama'].values()]

    # Build metadata
    metadata = {
        "method_name": "RAMA (Rank-Aware Moment Aggregation)",
        "experiment": "rama_avito_stress_test",
        "dataset": "rel-avito",
        "task": "ad-ctr",
        "task_type": "regression",
        "phase": phase_name,
        "config": {k: v for k, v in config.items() if k != 'seeds'},
        "seeds": config['seeds'],
        "results": {
            "baseline_mean": {
                "per_seed": baseline_metrics,
                "mean": {"mae": round(float(np.mean(baseline_maes)), 6),
                         "r2": round(float(np.mean(baseline_r2s)), 6),
                         "rmse": round(float(np.mean(baseline_rmses)), 6)},
                "std": {"mae": round(float(np.std(baseline_maes)), 6),
                        "r2": round(float(np.std(baseline_r2s)), 6),
                        "rmse": round(float(np.std(baseline_rmses)), 6)},
            },
            "rama": {
                "per_seed": rama_metrics,
                "mean": {"mae": round(float(np.mean(rama_maes)), 6),
                         "r2": round(float(np.mean(rama_r2s)), 6),
                         "rmse": round(float(np.mean(rama_rmses)), 6)},
                "std": {"mae": round(float(np.std(rama_maes)), 6),
                        "r2": round(float(np.std(rama_r2s)), 6),
                        "rmse": round(float(np.std(rama_rmses)), 6)},
            },
        },
        "statistical_comparison": {
            "mae": mae_comparison,
            "r2": r2_comparison,
        },
        "safety_check": {
            "mae_within_5pct": mae_pct_diff < 5.0,
            "r2_not_catastrophic": r2_not_catastrophic,
            "mae_pct_diff": round(mae_pct_diff, 4),
        },
        "gate_analysis": gate_analysis,
        "timing": {
            "seconds_per_epoch_baseline": round(baseline_avg_time, 2),
            "seconds_per_epoch_rama": round(rama_avg_time, 2),
            "rama_overhead_pct": round(overhead_pct, 2),
        },
        "memory": {
            "peak_vram_gb_baseline": round(float(np.mean(baseline_vrams)), 2),
            "peak_vram_gb_rama": round(float(np.mean(rama_vrams)), 2),
        },
    }

    # Build examples in exp_gen_sol_out schema format
    # Each seed + aggregation variant becomes an example
    examples = []
    seeds = config['seeds']
    for seed in seeds:
        seed_str = str(seed)
        baseline_m = baseline_metrics.get(seed_str, {"mae": 0, "r2": 0, "rmse": 0})
        rama_m = rama_metrics.get(seed_str, {"mae": 0, "r2": 0, "rmse": 0})

        input_text = (
            f"=== RAMA Stress Test ===\n"
            f"Dataset: rel-avito\n"
            f"Task: ad-ctr (regression)\n"
            f"Seed: {seed}\n"
            f"Config: channels={config['channels']}, neighbors={config['num_neighbors']}, "
            f"layers={config['num_layers']}, epochs={config['epochs']}, lr={config['lr']}\n"
            f"Mean FK cardinality: 114,625 (extreme)\n"
            f"Hypothesis: RAMA gate closes on extreme-cardinality edges, "
            f"gracefully degrading to mean aggregation"
        )
        output_text = json.dumps({
            "baseline_mae": baseline_m['mae'],
            "baseline_r2": baseline_m['r2'],
            "rama_mae": rama_m['mae'],
            "rama_r2": rama_m['r2'],
            "mae_pct_diff": round(abs(rama_m['mae'] - baseline_m['mae']) / max(baseline_m['mae'], 1e-12) * 100, 4),
            "rama_safe": rama_m['r2'] > -1.0,
        })

        example = {
            "input": input_text,
            "output": output_text,
            "predict_baseline": json.dumps({"mae": baseline_m['mae'], "r2": baseline_m['r2'], "rmse": baseline_m['rmse']}),
            "predict_rama": json.dumps({"mae": rama_m['mae'], "r2": rama_m['r2'], "rmse": rama_m['rmse']}),
            "metadata_seed": seed,
            "metadata_dataset": "rel-avito",
            "metadata_task": "ad-ctr",
            "metadata_task_type": "regression",
            "metadata_aggr_baseline": "mean",
            "metadata_aggr_method": "rama",
            "metadata_channels": config['channels'],
            "metadata_num_neighbors": config['num_neighbors'],
            "metadata_epochs": config['epochs'],
            "metadata_phase": phase_name,
        }
        examples.append(example)

    output = {
        "metadata": metadata,
        "datasets": [
            {
                "dataset": "rel-avito__ad-ctr",
                "examples": examples,
            }
        ],
    }
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["mini", "full", "preview"], default="mini")
    args = parser.parse_args()

    logger.info(f"=== RAMA Avito Stress Test (phase={args.phase}) ===")

    # ── Phase configs ────────────────────────────────────────────────────────
    if args.phase == "mini":
        config = {
            'channels': 64,
            'num_neighbors': 32,
            'batch_size': 256,
            'epochs': 2,
            'max_steps_per_epoch': 50,
            'lr': 0.005,
            'num_layers': 2,
            'grad_clip': 1.0,
            'seeds': [42],
        }
    elif args.phase == "preview":
        config = {
            'channels': 64,
            'num_neighbors': 32,
            'batch_size': 256,
            'epochs': 3,
            'max_steps_per_epoch': 200,
            'lr': 0.005,
            'num_layers': 2,
            'grad_clip': 1.0,
            'seeds': [42, 123, 456],
        }
    else:  # full
        config = {
            'channels': 64,
            'num_neighbors': 64,
            'batch_size': 256,
            'epochs': 10,
            'max_steps_per_epoch': 2000,
            'lr': 0.005,
            'num_layers': 2,
            'grad_clip': 1.0,
            'seeds': [42, 123, 456, 789, 1024],
        }

    logger.info(f"Config: {config}")

    # ── Build data (shared across all seeds/methods) ─────────────────────────
    data_bundle = build_data_and_loaders(config, config['seeds'][0])

    # Log edge types for analysis
    logger.info(f"Edge types ({len(data_bundle['edge_types'])}):")
    for et in data_bundle['edge_types']:
        ei = data_bundle['data'][et].edge_index
        logger.info(f"  {et}: {ei.size(1)} edges")

    # ── Run experiments ──────────────────────────────────────────────────────
    all_results = {'mean': {}, 'rama': {}}

    for aggr_name in ['mean', 'rama']:
        for seed in config['seeds']:
            try:
                result = run_single_experiment(config, seed, aggr_name, data_bundle)
                all_results[aggr_name][seed] = result
            except Exception as e:
                logger.exception(f"Failed {aggr_name} seed={seed}: {e}")
                # Insert a failure placeholder
                all_results[aggr_name][seed] = {
                    'test_metrics': {"mae": float('inf'), "r2": -9999, "rmse": float('inf')},
                    'best_val_mae': float('inf'),
                    'epoch_times': [0],
                    'peak_vram_gb': 0,
                    'gate_summary': None,
                }

    # ── Build and save output ────────────────────────────────────────────────
    output = build_output(all_results, config, data_bundle, phase_name=args.phase)

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Summary
    meta = output['metadata']
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info(f"  Baseline MAE: {meta['results']['baseline_mean']['mean']['mae']:.4f} "
                f"± {meta['results']['baseline_mean']['std']['mae']:.4f}")
    logger.info(f"  RAMA MAE:     {meta['results']['rama']['mean']['mae']:.4f} "
                f"± {meta['results']['rama']['std']['mae']:.4f}")
    logger.info(f"  MAE within 5%: {meta['safety_check']['mae_within_5pct']}")
    logger.info(f"  R² not catastrophic: {meta['safety_check']['r2_not_catastrophic']}")
    logger.info(f"  Cohen's d (MAE): {meta['statistical_comparison']['mae']['cohen_d']}")
    if meta['gate_analysis']['per_edge_type']:
        logger.info(f"  Gate closes on extreme: {meta['gate_analysis'].get('gate_closes_on_extreme', 'N/A')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
