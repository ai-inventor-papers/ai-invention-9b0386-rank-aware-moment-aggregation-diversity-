#!/usr/bin/env python3
"""CAMA (Cardinality-Aware Moment Aggregation) experiment.

Runs 2 methods (HeteroSAGE+mean baseline, HeteroSAGE+CAMA) x 3 tasks x 5 seeds = 30 runs.
Tasks: rel-trial/study-outcome, rel-trial/study-adverse, rel-stack/user-engagement.
Reports per-task Cohen's d with bootstrap 95% CI and gate values per edge type.

MEMORY-SAFE: Processes one dataset at a time, frees graph data between datasets.
"""

import os
import sys
import math
import copy
import json
import time
import gc
import resource
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, L1Loss
from loguru import logger

# ============================================================================
# LOGGING SETUP
# ============================================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ============================================================================
# HARDWARE DETECTION & MEMORY LIMITS
# ============================================================================
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


def _get_current_ram_gb() -> float:
    """Read current RAM usage from cgroup."""
    for p in ["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"]:
        try:
            return int(Path(p).read_text().strip()) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return psutil.Process().memory_info().rss / 1e9


NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
            f"GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM)")

# Set RAM limit to 70% of container limit — leave 30% for agent + OS
RAM_BUDGET_GB = TOTAL_RAM_GB * 0.70
RAM_BUDGET = int(RAM_BUDGET_GB * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET_GB:.1f}GB (70% of {TOTAL_RAM_GB:.1f}GB)")

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.90)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))
    logger.info(f"VRAM budget: {VRAM_BUDGET / 1e9:.1f}GB (90% of {_total / 1e9:.1f}GB)")

# ============================================================================
# DELAYED IMPORTS (after hardware setup)
# ============================================================================
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
from scipy import stats

ROOT_DIR = os.path.expanduser("~/.cache/relbench_cama_exp")
SCRIPT_DIR = Path(__file__).parent


# ============================================================================
# STEP 1: IMPLEMENT CAMAAggregation
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

    def stop_recording(self):
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
# STEP 2: IMPLEMENT CAMAHeteroGraphSAGE
# ============================================================================
class CAMAHeteroGraphSAGE(nn.Module):
    """Modified HeteroGraphSAGE with CAMA aggregation per edge type per layer."""

    def __init__(self, node_types, edge_types, channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.cama_registry = {}

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
# STEP 3: IMPLEMENT WRAPPER MODEL
# ============================================================================
class Model(nn.Module):
    def __init__(self, data, col_stats_dict, num_layers, channels,
                 out_channels, use_cama=False):
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
        if use_cama:
            self.gnn = CAMAHeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                num_layers=num_layers,
            )
        else:
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
# STEP 4: TASK CONFIGURATIONS
# ============================================================================
# IMPORTANT: rel-trial tasks first (smaller, faster, less memory),
# then rel-stack (larger, slower, more memory).
TASK_CONFIGS = [
    {
        'dataset_name': 'rel-trial',
        'task_name': 'study-outcome',
        'task_type': 'binary_classification',
        'primary_metric': 'average_precision',
        'tune_metric': 'average_precision',
        'higher_is_better': True,
        'out_channels': 1,
        'loss_fn': BCEWithLogitsLoss(),
        'channels': 128,
        'lr': 0.001,
        'epochs': 20,
        'batch_size': 512,
        'num_neighbors': [64, 64],
        'max_steps_per_epoch': 500,
    },
    {
        'dataset_name': 'rel-trial',
        'task_name': 'study-adverse',
        'task_type': 'regression',
        'primary_metric': 'mae',
        'tune_metric': 'mae',
        'higher_is_better': False,
        'out_channels': 1,
        'loss_fn': L1Loss(),
        'channels': 128,
        'lr': 0.001,
        'epochs': 20,
        'batch_size': 512,
        'num_neighbors': [64, 64],
        'max_steps_per_epoch': 500,
    },
    {
        'dataset_name': 'rel-stack',
        'task_name': 'user-engagement',
        'task_type': 'binary_classification',
        'primary_metric': 'roc_auc',
        'tune_metric': 'roc_auc',
        'higher_is_better': True,
        'out_channels': 1,
        'loss_fn': BCEWithLogitsLoss(),
        'channels': 128,
        'lr': 0.005,
        'epochs': 10,
        'batch_size': 512,
        'num_neighbors': [64, 64],      # Reduced from [128,128] for memory/time
        'max_steps_per_epoch': 500,      # Reduced from 2000 for time
    },
]
SEEDS = [42, 123, 456, 789, 1024]
METHODS = ['baseline', 'cama']


# ============================================================================
# STEP 5: GLOVE TEXT EMBEDDING
# ============================================================================
class GloveTextEmbedding:
    """Uses sentence-transformers GloVe model for text embedding."""

    def __init__(self, device=None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences):
        return torch.from_numpy(
            self.model.encode(sentences, show_progress_bar=False)
        )


# ============================================================================
# STEP 6: TRAIN AND TEST FUNCTIONS
# ============================================================================
def train_epoch(model, loader, optimizer, loss_fn, entity_table,
                task_type, max_steps):
    model.train()
    loss_accum = count_accum = 0
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch, entity_table).view(-1)
        loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)
    return loss_accum / count_accum if count_accum > 0 else float('inf')


@torch.no_grad()
def test_predict(model, loader, entity_table, task_type,
                 clamp_min=None, clamp_max=None):
    model.eval()
    pred_list = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table).view(-1)
        if task_type == 'regression' and clamp_min is not None:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task_type == 'binary_classification':
            pred = torch.sigmoid(pred)
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


# ============================================================================
# STEP 7: COHEN'S D STATISTICS
# ============================================================================
def cohens_d_independent(group1, group2):
    """Independent-samples Cohen's d. Positive = group2 better."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled < 1e-12:
        return 0.0
    return (m2 - m1) / s_pooled


def bootstrap_cohens_d(baseline_scores, cama_scores,
                       n_bootstrap=10000, ci=0.95):
    """Bootstrap CI for Cohen's d."""
    rng = np.random.RandomState(42)
    n_b, n_c = len(baseline_scores), len(cama_scores)
    boot_ds = []
    for _ in range(n_bootstrap):
        idx_b = rng.choice(n_b, n_b, replace=True)
        idx_c = rng.choice(n_c, n_c, replace=True)
        boot_ds.append(
            cohens_d_independent(baseline_scores[idx_b], cama_scores[idx_c])
        )
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(boot_ds, 100 * alpha)),
        float(np.percentile(boot_ds, 100 * (1 - alpha))),
    )


def format_edge_type(et) -> str:
    """Convert PyG edge type tuple to readable string."""
    if isinstance(et, tuple):
        return "__".join(str(x) for x in et)
    return str(et)


# ============================================================================
# MAIN
# ============================================================================
@logger.catch
def main():
    os.makedirs(ROOT_DIR, exist_ok=True)
    out_path = SCRIPT_DIR / "method_out.json"
    partial_path = SCRIPT_DIR / "partial_results.json"
    global_t0 = time.time()

    # ------------------------------------------------------------------
    # DATASET LOADING AND EXPERIMENT LOOP
    # Process one dataset at a time — free graph before loading next.
    # ------------------------------------------------------------------
    all_results = _load_partial(partial_path)
    current_dataset = None
    graph_data = None
    col_stats = None
    text_embedder = None

    for tc_idx, tc in enumerate(TASK_CONFIGS):
        ds_name = tc['dataset_name']

        # Load dataset if not already loaded (or if different dataset)
        if ds_name != current_dataset:
            # Free previous dataset
            if graph_data is not None:
                logger.info(f"Freeing graph data for {current_dataset}")
                del graph_data, col_stats
                gc.collect()
                torch.cuda.empty_cache()

            logger.info(f"Loading dataset: {ds_name}")
            logger.info(f"  RAM: {_get_current_ram_gb():.1f}GB / {TOTAL_RAM_GB:.1f}GB")
            t0 = time.time()

            if text_embedder is None:
                text_embedder = GloveTextEmbedding(device=DEVICE)

            dataset = get_dataset(ds_name, download=True)
            db = dataset.get_db()
            col_to_stype_dict = get_stype_proposal(db)
            text_embedder_cfg = TextEmbedderConfig(
                text_embedder=text_embedder, batch_size=256,
            )
            cache_dir = os.path.join(ROOT_DIR, f"{ds_name}_cache")
            graph_data, col_stats = make_pkey_fkey_graph(
                db, col_to_stype_dict=col_to_stype_dict,
                text_embedder_cfg=text_embedder_cfg,
                cache_dir=cache_dir,
            )
            current_dataset = ds_name

            # Free temporary references
            del dataset, db, col_to_stype_dict
            gc.collect()

            logger.info(f"  Loaded in {time.time()-t0:.1f}s")
            logger.info(f"  Nodes: {graph_data.node_types}")
            et_strs = [format_edge_type(et) for et in graph_data.edge_types]
            logger.info(f"  Edges: {et_strs}")
            logger.info(f"  RAM: {_get_current_ram_gb():.1f}GB / {TOTAL_RAM_GB:.1f}GB")

        # Get task
        task = get_task(ds_name, tc['task_name'], download=True)
        entity_table = task.entity_table
        logger.info(f"=== Task {tc_idx+1}/{len(TASK_CONFIGS)}: "
                     f"{ds_name}/{tc['task_name']} ===")

        # Compute regression clamp bounds
        clamp_min, clamp_max = None, None
        if tc['task_type'] == 'regression':
            train_table = task.get_table("train")
            targets = train_table.df[task.target_col].to_numpy()
            clamp_min = float(np.percentile(targets, 2))
            clamp_max = float(np.percentile(targets, 98))
            logger.info(f"  Clamp: [{clamp_min:.4f}, {clamp_max:.4f}]")

        # Create data loaders
        loader_dict = {}
        for split in ["train", "val", "test"]:
            table = task.get_table(split)
            table_input = get_node_train_table_input(table=table, task=task)
            loader_dict[split] = NeighborLoader(
                graph_data,
                num_neighbors=tc['num_neighbors'],
                time_attr="time",
                input_nodes=table_input.nodes,
                input_time=table_input.time,
                transform=table_input.transform,
                batch_size=tc['batch_size'],
                temporal_strategy="uniform",
                shuffle=(split == "train"),
                num_workers=0,
                persistent_workers=False,
            )
        logger.info(f"  Loaders: train={len(loader_dict['train'])}, "
                     f"val={len(loader_dict['val'])}, "
                     f"test={len(loader_dict['test'])}")

        for method in METHODS:
            for seed in SEEDS:
                run_key = f"{tc['task_name']}__{method}__seed{seed}"
                # Skip if already completed in partial results
                if (tc['task_name'], method, seed) in all_results and \
                        all_results[(tc['task_name'], method, seed)].get('test_metrics'):
                    logger.info(f"--- Skipping {run_key} (already completed) ---")
                    continue
                logger.info(f"--- Run: {run_key} ---")
                seed_everything(seed)
                run_t0 = time.time()

                model = None
                optimizer = None
                best_state = None

                try:
                    use_cama = (method == 'cama')
                    model = Model(
                        data=graph_data, col_stats_dict=col_stats,
                        num_layers=2, channels=tc['channels'],
                        out_channels=tc['out_channels'], use_cama=use_cama,
                    ).to(DEVICE)
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=tc['lr'],
                    )

                    best_val = (
                        -math.inf if tc['higher_is_better'] else math.inf
                    )
                    best_epoch = 0

                    for epoch in range(1, tc['epochs'] + 1):
                        t0 = time.time()
                        train_loss = train_epoch(
                            model, loader_dict["train"], optimizer,
                            tc['loss_fn'], entity_table, tc['task_type'],
                            tc['max_steps_per_epoch'],
                        )
                        val_pred = test_predict(
                            model, loader_dict["val"], entity_table,
                            tc['task_type'], clamp_min, clamp_max,
                        )
                        val_metrics = task.evaluate(
                            val_pred, task.get_table("val"),
                        )
                        elapsed = time.time() - t0

                        val_score = val_metrics[tc['tune_metric']]
                        improved = (
                            (tc['higher_is_better'] and val_score >= best_val)
                            or (not tc['higher_is_better']
                                and val_score <= best_val)
                        )
                        if improved:
                            best_val = val_score
                            best_epoch = epoch
                            if best_state is not None:
                                del best_state
                            best_state = copy.deepcopy(model.state_dict())

                        logger.info(
                            f"  Epoch {epoch}: loss={train_loss:.4f}, "
                            f"val_{tc['tune_metric']}={val_score:.4f}, "
                            f"time={elapsed:.1f}s"
                        )

                    # Load best model and test
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    test_pred = test_predict(
                        model, loader_dict["test"], entity_table,
                        tc['task_type'], clamp_min, clamp_max,
                    )
                    test_metrics = task.evaluate(test_pred)
                    # Convert numpy values to Python floats
                    test_metrics = {
                        k: float(v) for k, v in test_metrics.items()
                    }
                    logger.info(f"  TEST: {test_metrics}")

                    # Study-adverse diagnostic
                    diagnostics = {}
                    if tc['task_name'] == 'study-adverse':
                        # Test table may not have target col (RelBench
                        # hides test labels); fall back to val table.
                        try:
                            test_df = task.get_table("test").df
                            if task.target_col in test_df.columns:
                                ref_y = test_df[task.target_col].to_numpy()
                            else:
                                ref_y = (
                                    task.get_table("val")
                                    .df[task.target_col].to_numpy()
                                )
                        except Exception:
                            ref_y = (
                                task.get_table("val")
                                .df[task.target_col].to_numpy()
                            )
                        diagnostics = {
                            'pred_variance': float(np.var(test_pred)),
                            'target_variance': float(np.var(ref_y)),
                            'pred_std': float(np.std(test_pred)),
                            'target_std': float(np.std(ref_y)),
                            'pred_range': float(np.ptp(test_pred)),
                            'target_range': float(np.ptp(ref_y)),
                            'pred_mean': float(np.mean(test_pred)),
                            'target_mean': float(np.mean(ref_y)),
                            'pred_median': float(np.median(test_pred)),
                            'pred_unique_ratio': float(
                                len(np.unique(np.round(test_pred, 4)))
                                / len(test_pred)
                            ),
                            'is_degenerate': bool(
                                np.std(test_pred) < 0.01 * np.std(ref_y)
                            ),
                        }
                        logger.info(
                            f"  DIAG: pred_std={diagnostics['pred_std']:.6f},"
                            f" target_std={diagnostics['target_std']:.4f},"
                            f" degenerate={diagnostics['is_degenerate']}"
                        )

                    # Gate extraction (CAMA only) — separate try/except
                    # so gate errors don't overwrite successful test results
                    gate_stats_per_edge = {}
                    if use_cama and isinstance(
                        model.gnn, CAMAHeteroGraphSAGE
                    ):
                        try:
                            for key, cama_mod in (
                                model.gnn.cama_registry.items()
                            ):
                                cama_mod.start_recording()
                            _ = test_predict(
                                model, loader_dict["test"], entity_table,
                                tc['task_type'], clamp_min, clamp_max,
                            )
                            for (layer_idx, edge_type), cama_mod in (
                                model.gnn.cama_registry.items()
                            ):
                                gs = cama_mod.stop_recording()
                                if gs is not None:
                                    et_str = format_edge_type(edge_type)
                                    gate_stats_per_edge[
                                        f"L{layer_idx}_{et_str}"
                                    ] = gs
                        except Exception:
                            logger.exception(
                                f"Gate extraction failed for {run_key}"
                            )

                    run_time = time.time() - run_t0
                    all_results[(tc['task_name'], method, seed)] = {
                        'test_metrics': test_metrics,
                        'best_val_metric': float(best_val),
                        'best_epoch': best_epoch,
                        'diagnostics': diagnostics,
                        'gate_stats': gate_stats_per_edge,
                        'run_time_s': float(run_time),
                    }
                    logger.info(f"  Run completed in {run_time:.1f}s")

                except Exception:
                    logger.exception(f"FAILED: {run_key}")
                    all_results[(tc['task_name'], method, seed)] = {
                        'test_metrics': {},
                        'best_val_metric': float('nan'),
                        'best_epoch': 0,
                        'diagnostics': {},
                        'gate_stats': {},
                        'run_time_s': 0.0,
                        'error': traceback.format_exc(),
                    }
                finally:
                    del model, optimizer, best_state
                    torch.cuda.empty_cache()
                    gc.collect()

        # Free loaders after all runs for this task
        del loader_dict
        gc.collect()

        # Log progress
        elapsed_total = time.time() - global_t0
        logger.info(
            f"  Total: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min), "
            f"RAM: {_get_current_ram_gb():.1f}GB / {TOTAL_RAM_GB:.1f}GB"
        )

        # Save intermediate results
        _save_partial(all_results, partial_path)

    # Free final graph data
    if graph_data is not None:
        del graph_data, col_stats
        gc.collect()
    if text_embedder is not None:
        del text_embedder
        gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # STEP 8: COHEN'S D WITH BOOTSTRAP 95% CI
    # ------------------------------------------------------------------
    logger.info("=== Computing Cohen's d statistics ===")
    task_analysis = {}
    for tc in TASK_CONFIGS:
        tn = tc['task_name']
        metric = tc['primary_metric']
        higher_better = tc['higher_is_better']

        baseline_scores = np.array([
            all_results.get((tn, 'baseline', s), {})
            .get('test_metrics', {}).get(metric, float('nan'))
            for s in SEEDS
        ])
        cama_scores = np.array([
            all_results.get((tn, 'cama', s), {})
            .get('test_metrics', {}).get(metric, float('nan'))
            for s in SEEDS
        ])

        logger.info(f"  {tn}: baseline={baseline_scores}, cama={cama_scores}")

        if np.any(np.isnan(baseline_scores)) or np.any(np.isnan(cama_scores)):
            logger.warning(f"  {tn}: NaN detected, skipping Cohen's d")
            task_analysis[tn] = {
                'metric': metric,
                'error': 'NaN values in scores',
                'baseline_per_seed': [float(x) for x in baseline_scores],
                'cama_per_seed': [float(x) for x in cama_scores],
            }
            continue

        # Positive d = CAMA better
        if not higher_better:
            d = cohens_d_independent(cama_scores, baseline_scores)
        else:
            d = cohens_d_independent(baseline_scores, cama_scores)

        ci_lo, ci_hi = bootstrap_cohens_d(
            baseline_scores if higher_better else cama_scores,
            cama_scores if higher_better else baseline_scores,
        )

        t_stat, p_val = stats.ttest_ind(
            baseline_scores, cama_scores, equal_var=False,
        )

        task_analysis[tn] = {
            'metric': metric,
            'baseline_mean': float(np.mean(baseline_scores)),
            'baseline_std': float(np.std(baseline_scores, ddof=1)),
            'baseline_per_seed': baseline_scores.tolist(),
            'cama_mean': float(np.mean(cama_scores)),
            'cama_std': float(np.std(cama_scores, ddof=1)),
            'cama_per_seed': cama_scores.tolist(),
            'cohens_d': float(d),
            'cohens_d_ci_lower': ci_lo,
            'cohens_d_ci_upper': ci_hi,
            'p_value': float(p_val),
            'significant_at_005': bool(p_val < 0.05),
            'higher_is_better': higher_better,
        }
        logger.info(
            f"  {tn}: d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={p_val:.4f}"
        )

    # ------------------------------------------------------------------
    # STEP 9: GATE ANALYSIS
    # ------------------------------------------------------------------
    gate_analysis = {}
    for tc in TASK_CONFIGS:
        tn = tc['task_name']
        edge_gate_means = {}
        for seed in SEEDS:
            gs = all_results.get(
                (tn, 'cama', seed), {},
            ).get('gate_stats', {})
            for edge_key, stats_dict in gs.items():
                edge_gate_means.setdefault(
                    edge_key, [],
                ).append(stats_dict['mean'])
        gate_analysis[tn] = {
            edge_key: {
                'mean_across_seeds': float(np.mean(vals)),
                'std_across_seeds': float(np.std(vals)),
                'per_seed': vals,
            }
            for edge_key, vals in edge_gate_means.items()
        }

    # ------------------------------------------------------------------
    # STEP 10: STUDY-ADVERSE DIAGNOSIS
    # ------------------------------------------------------------------
    adverse_diagnosis = {}
    for seed in SEEDS:
        diag = all_results.get(
            ('study-adverse', 'baseline', seed), {},
        ).get('diagnostics', {})
        adverse_diagnosis[f'seed_{seed}'] = diag
    degenerate_seeds = [
        s for s in SEEDS
        if all_results.get(('study-adverse', 'baseline', s), {})
        .get('diagnostics', {}).get('is_degenerate', False)
    ]
    adverse_diagnosis['degenerate_seeds'] = degenerate_seeds
    adverse_diagnosis['num_degenerate'] = len(degenerate_seeds)
    adverse_diagnosis['summary'] = (
        f"{len(degenerate_seeds)}/5 baseline seeds show degenerate "
        "(near-constant) predictions. This inflates Cohen's d artificially."
        if degenerate_seeds else
        "No degenerate seeds detected. Baseline produces diverse predictions."
    )

    # ------------------------------------------------------------------
    # STEP 11: BUILD OUTPUT IN exp_gen_sol_out SCHEMA FORMAT
    # ------------------------------------------------------------------
    logger.info("=== Building output ===")
    datasets_out = []
    for tc in TASK_CONFIGS:
        tn = tc['task_name']
        ds_name = tc['dataset_name']
        metric = tc['primary_metric']
        examples = []

        for seed in SEEDS:
            baseline_res = all_results.get((tn, 'baseline', seed), {})
            cama_res = all_results.get((tn, 'cama', seed), {})

            baseline_metric_val = baseline_res.get(
                'test_metrics', {},
            ).get(metric, None)
            cama_metric_val = cama_res.get(
                'test_metrics', {},
            ).get(metric, None)

            input_str = (
                f"Task: {ds_name}/{tn}\n"
                f"Type: {tc['task_type']}\n"
                f"Metric: {metric} "
                f"({'higher' if tc['higher_is_better'] else 'lower'}"
                f" is better)\n"
                f"Seed: {seed}\n"
                f"Channels: {tc['channels']}, LR: {tc['lr']}, "
                f"Epochs: {tc['epochs']}\n"
                f"Batch size: {tc['batch_size']}, "
                f"Neighbors: {tc['num_neighbors']}"
            )

            output_str = json.dumps({
                'baseline_test_metrics': baseline_res.get(
                    'test_metrics', {},
                ),
                'cama_test_metrics': cama_res.get('test_metrics', {}),
                'baseline_best_val': baseline_res.get(
                    'best_val_metric', None,
                ),
                'cama_best_val': cama_res.get('best_val_metric', None),
                'baseline_best_epoch': baseline_res.get('best_epoch', 0),
                'cama_best_epoch': cama_res.get('best_epoch', 0),
                'diagnostics': baseline_res.get('diagnostics', {}),
                'gate_stats': cama_res.get('gate_stats', {}),
            })

            example = {
                'input': input_str,
                'output': output_str,
                'predict_baseline': (
                    str(baseline_metric_val)
                    if baseline_metric_val is not None else "N/A"
                ),
                'predict_cama': (
                    str(cama_metric_val)
                    if cama_metric_val is not None else "N/A"
                ),
                'metadata_dataset_name': ds_name,
                'metadata_task_name': tn,
                'metadata_task_type': tc['task_type'],
                'metadata_metric': metric,
                'metadata_seed': seed,
                'metadata_method_baseline': 'HeteroSAGE+mean',
                'metadata_method_cama': 'HeteroSAGE+CAMA',
            }
            examples.append(example)

        datasets_out.append({
            'dataset': f"{ds_name}__{tn}",
            'examples': examples,
        })

    output = {
        'metadata': {
            'experiment': 'cama_5seed_3tasks',
            'methods': [
                'baseline (HeteroSAGE+mean)',
                'CAMA (HeteroSAGE+CAMAAggregation)',
            ],
            'seeds': SEEDS,
            'tasks': {
                tc['task_name']: {
                    'dataset': tc['dataset_name'],
                    'metric': tc['primary_metric'],
                    'task_type': tc['task_type'],
                }
                for tc in TASK_CONFIGS
            },
            'task_level_analysis': task_analysis,
            'gate_analysis': gate_analysis,
            'study_adverse_diagnosis': adverse_diagnosis,
        },
        'datasets': datasets_out,
    }

    out_path.write_text(json.dumps(output, indent=2))
    logger.info(
        f"Saved output to {out_path} "
        f"({out_path.stat().st_size / 1024:.1f} KB)"
    )

    # Clean up partial results
    if partial_path.exists():
        partial_path.unlink()

    total_time = time.time() - global_t0
    logger.info(f"=== DONE in {total_time:.0f}s ({total_time/60:.1f} min) ===")


def _save_partial(all_results: dict, path: Path):
    """Save partial results to disk for crash recovery."""
    try:
        partial = {}
        for (tn, method, seed), res in all_results.items():
            key = f"{tn}__{method}__seed{seed}"
            partial[key] = {
                'test_metrics': res.get('test_metrics', {}),
                'best_val_metric': res.get('best_val_metric', float('nan')),
                'run_time_s': res.get('run_time_s', 0),
                'diagnostics': res.get('diagnostics', {}),
                'gate_stats': res.get('gate_stats', {}),
            }
        Path(path).write_text(json.dumps(partial, indent=2))
        logger.debug(f"Saved partial results ({len(partial)} runs)")
    except Exception:
        logger.exception("Failed to save partial results")


def _load_partial(path: Path) -> dict:
    """Load partial results from a previous run for crash recovery."""
    all_results = {}
    if not path.exists():
        return all_results
    try:
        partial = json.loads(path.read_text())
        for key, res in partial.items():
            # Parse key: "task_name__method__seedN"
            parts = key.split("__")
            if len(parts) != 3:
                continue
            tn, method, seed_str = parts
            seed = int(seed_str.replace("seed", ""))
            # Only load if test_metrics is non-empty (successful run)
            if res.get('test_metrics'):
                all_results[(tn, method, seed)] = res
        logger.info(f"Loaded {len(all_results)} completed runs from "
                     f"partial results")
    except Exception:
        logger.exception("Failed to load partial results, starting fresh")
    return all_results


if __name__ == "__main__":
    main()
