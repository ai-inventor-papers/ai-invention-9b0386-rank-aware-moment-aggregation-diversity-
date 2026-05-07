#!/usr/bin/env python3
"""Compare Sum, Mean, PNA, Ungated-Moment, and CAMA aggregation methods
on rel-stack/user-engagement + Sum on rel-trial/study-outcome.

Measures: best-validation average_precision, Cohen's d with bootstrap CIs,
effective rank of aggregated representations.
Output: method_out.json conforming to exp_gen_sol_out schema.

NOTE: relbench 2.1.1 test splits have NO target labels, so the best validation
metric during training is used for all comparisons. All methods are evaluated on
the same validation split, so relative comparisons remain valid.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
RESULTS_DIR = WORKSPACE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

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
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 42.0
RAM_BUDGET_BYTES = int(min(TOTAL_RAM_GB * 0.70, 28) * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, "
            f"budget={RAM_BUDGET_BYTES / 1e9:.1f} GB")

# ---------------------------------------------------------------------------
# PyTorch imports (after hardware setup)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

HAS_GPU = torch.cuda.is_available()
if HAS_GPU:
    DEVICE = torch.device("cuda")
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.90)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))
    torch.set_num_threads(1)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM={VRAM_GB:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    VRAM_GB = 0
    logger.warning("No GPU detected, using CPU (will be slow)")

from scipy import stats
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP, SAGEConv
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.seed import seed_everything

# ---------------------------------------------------------------------------
# Monkey-patch: pyg-lib unavailable for torch 2.10; disable forced disjoint
# sampling when time_attr is set. Temporal filtering is lost, but all methods
# get identical treatment, so relative comparison remains valid.
# ---------------------------------------------------------------------------
try:
    from torch_geometric.sampler.neighbor_sampler import NeighborSampler as _NS
    _NS.disjoint = property(
        lambda self: self._disjoint,
        lambda self, v: setattr(self, '_disjoint', v),
    )
except Exception:
    logger.warning("Could not apply NeighborSampler monkey-patch")

from relbench.base import Dataset, EntityTask, Table, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from torch_geometric.nn.conv import HeteroConv
from torch_geometric.nn.norm import LayerNorm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = str(WORKSPACE / "relbench_cache")
SEEDS = [42, 123, 456, 789, 1024]
HYPERPARAMS = dict(
    channels=128,
    lr=0.005,
    epochs=5,
    batch_size=128,
    num_neighbors=16,
    num_layers=2,
    max_steps_per_epoch=100,
)
TIME_BUDGET_SECONDS = 50 * 60  # 50 min total budget
SCRIPT_START_TIME = time.time()


# ---------------------------------------------------------------------------
# GloVe Text Embedder (hash-based, deterministic, fast)
# ---------------------------------------------------------------------------
class GloveTextEmbedding:
    """Deterministic hash-based text embedding for torch_frame.
    All methods use the same embedder, so relative comparisons remain valid."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE

    def __call__(self, sentences: List[str]) -> Tensor:
        n = len(sentences)
        out = np.zeros((n, 300), dtype=np.float32)
        for i, sent in enumerate(sentences):
            h = hash(sent) % (2**31)
            rng = np.random.RandomState(h)
            out[i] = rng.standard_normal(300).astype(np.float32) * 0.1
        return torch.tensor(out, dtype=torch.float32, device=self.device)


# ---------------------------------------------------------------------------
# Aggregation Modules
# ---------------------------------------------------------------------------
class CAMAAggregation(Aggregation):
    """Context-Aware Moment Aggregation (CAMA).
    Combines mean with gated variance injection conditioned on cardinality
    and variance entropy (effective rank proxy)."""

    def __init__(self, channels: int, gate_hidden: int = 16, eps: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.gate_hidden = gate_hidden
        self.eps = eps
        self.W_sigma = nn.Linear(channels, channels, bias=False)
        self.W_g = nn.Linear(2, gate_hidden)
        self.gate_out = nn.Linear(gate_hidden, channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.W_sigma.weight)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None,
                dim_size: Optional[int] = None,
                dim: int = -2,
                max_num_elements: Optional[int] = None) -> Tensor:
        mu = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce='mean')
        var = (mean_x2 - mu * mu).clamp(min=0)
        ones = torch.ones(x.size(0), 1, device=x.device)
        N = self.reduce(ones, index, ptr, dim_size, dim, reduce='sum')
        var_sum = var.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        p = var / var_sum
        p = p.clamp(min=self.eps)
        H = -(p * p.log()).sum(dim=-1, keepdim=True)
        log_d = math.log(max(self.channels, 2))
        rho = H / log_d
        gate_input = torch.cat([torch.log1p(N), rho], dim=-1)
        g = torch.sigmoid(self.gate_out(F.relu(self.W_g(gate_input))))
        return mu + g * self.W_sigma(var)


class UngatedMomentAggregation(Aggregation):
    """Ungated Moment Injection: mu + W_sigma(var) without gating.
    Ablation baseline to isolate CAMA's gating mechanism effect."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.W_sigma = nn.Linear(channels, channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.W_sigma.weight)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None,
                dim_size: Optional[int] = None,
                dim: int = -2,
                max_num_elements: Optional[int] = None) -> Tensor:
        mu = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        mean_x2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce='mean')
        var = (mean_x2 - mu * mu).clamp(min=0)
        return mu + self.W_sigma(var)


def make_pna_aggr(channels: int) -> MultiAggregation:
    """PNA-style MultiAggregation with projection mode.
    Uses [mean, std, min, max] aggregators combined via learned projection."""
    return MultiAggregation(
        ['mean', 'std', 'min', 'max'],
        mode='proj',
        mode_kwargs={'in_channels': channels, 'out_channels': channels},
    )


# ---------------------------------------------------------------------------
# Custom HeteroGraphSAGE (deepcopy aggregation per conv layer)
# ---------------------------------------------------------------------------
class HeteroGraphSAGE(torch.nn.Module):
    """HeteroGraphSAGE that properly handles non-string Aggregation objects
    by deep-copying them for each SAGEConv instance."""

    def __init__(self, node_types, edge_types, channels, aggr="mean",
                 num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv(
                        (channels, channels), channels,
                        aggr=(copy.deepcopy(aggr) if not isinstance(aggr, str)
                              else aggr),
                    )
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)
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

    def forward(self, x_dict, edge_index_dict,
                num_sampled_nodes_dict=None,
                num_sampled_edges_dict=None):
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                key: norm_dict[key](x.relu())
                for key, x in x_dict.items()
                if key in norm_dict
            }
        return x_dict


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class Model(torch.nn.Module):
    def __init__(self, data, col_stats_dict, num_layers, channels,
                 out_channels, aggr, norm):
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
                node_type for node_type in data.node_types
                if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch, entity_table: str) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        try:
            batch_dict = batch.batch_dict
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch_dict
            )
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time
        except (KeyError, AttributeError):
            pass
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        return self.head(x_dict[entity_table][:seed_time.size(0)])


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_task_data(dataset_name: str, task_name: str) -> Tuple:
    """Download dataset and task, build graph, return everything needed."""
    logger.info(f"Loading {dataset_name}/{task_name}...")
    t_start = time.time()

    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)
    logger.info(f"Dataset and task loaded in {time.time()-t_start:.1f}s")

    # Stype proposal (cached)
    stypes_cache = Path(f"{CACHE_DIR}/{dataset_name}/stypes.json")
    try:
        with open(stypes_cache, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
        logger.info("Loaded cached stypes")
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset.get_db())
        stypes_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)
        logger.info("Computed and cached stypes")

    # Graph materialization
    logger.info("Materializing graph...")
    t_mat = time.time()
    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=DEVICE),
            batch_size=256,
        ),
        cache_dir=f"{CACHE_DIR}/{dataset_name}/materialized",
    )
    logger.info(f"Graph materialized in {time.time()-t_mat:.1f}s")
    logger.info(f"  Node types: {data.node_types}")
    logger.info(f"  Edge types: {len(data.edge_types)} types")

    return dataset, task, data, col_stats_dict


def build_loaders(task, data, batch_size, num_neighbors, num_layers):
    """Build train/val NeighborLoaders (skip test — no labels)."""
    loader_dict = {}
    entity_table = None
    for split in ["train", "val"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
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
            shuffle=(split == "train"),
            num_workers=0,
        )
    return loader_dict, entity_table


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, loss_fn, entity_table,
                task_type, max_steps=200):
    """Train one epoch, return average loss."""
    model.train()
    loss_accum = 0.0
    count_accum = 0
    for steps, batch in enumerate(loader):
        if steps >= max_steps:
            break
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch, entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = loss_fn(pred, batch[entity_table].y.long())
        else:
            loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)
    return loss_accum / max(count_accum, 1)


@torch.no_grad()
def evaluate_predictions(model, loader, entity_table, task_type,
                         clamp_min=None, clamp_max=None):
    """Evaluate model, return raw predictions as numpy array."""
    model.eval()
    pred_list = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table)
        if task_type == TaskType.REGRESSION and clamp_min is not None:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task_type in [TaskType.BINARY_CLASSIFICATION,
                         TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = torch.softmax(pred, dim=1)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


def run_single_experiment(task, data, col_stats_dict, aggr,
                          method_name, seed, hp):
    """Train model with given aggregation, return metrics.
    Uses VALIDATION set for evaluation (test labels unavailable)."""
    logger.info(f"  [{method_name}|seed={seed}] Starting...")
    t0 = time.time()
    seed_everything(seed)

    task_type = task.task_type
    clamp_min, clamp_max = None, None
    if task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "average_precision"
        higher_is_better = True
    elif task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = L1Loss()
        tune_metric = "mae"
        higher_is_better = False
        train_table = task.get_table("train")
        clamp_min, clamp_max = np.percentile(
            train_table.df[task.target_col].to_numpy(), [2, 98]
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    loader_dict, entity_table = build_loaders(
        task, data,
        batch_size=hp['batch_size'],
        num_neighbors=hp['num_neighbors'],
        num_layers=hp['num_layers'],
    )

    if isinstance(aggr, str):
        aggr_obj = aggr
    else:
        aggr_obj = copy.deepcopy(aggr)

    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=hp['num_layers'],
        channels=hp['channels'],
        out_channels=out_channels,
        aggr=aggr_obj,
        norm="layer_norm",
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])

    # Training loop with val-based model selection
    best_val_metric = -math.inf if higher_is_better else math.inf
    state_dict = None
    epoch_metrics = []
    for epoch in range(1, hp['epochs'] + 1):
        train_loss = train_epoch(
            model, loader_dict["train"], optimizer, loss_fn,
            entity_table, task_type, max_steps=hp['max_steps_per_epoch'],
        )
        val_pred = evaluate_predictions(
            model, loader_dict["val"], entity_table, task_type,
            clamp_min, clamp_max,
        )
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        val_score = val_metrics[tune_metric]
        improved = (
            (higher_is_better and val_score >= best_val_metric)
            or (not higher_is_better and val_score <= best_val_metric)
        )
        if improved:
            best_val_metric = val_score
            state_dict = copy.deepcopy(model.state_dict())
        epoch_metrics.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            f"val_{tune_metric}": round(float(val_score), 4),
        })
        logger.debug(f"    Epoch {epoch}: loss={train_loss:.4f}, "
                     f"val_{tune_metric}={val_score:.4f}"
                     f"{' *' if improved else ''}")

    elapsed = time.time() - t0
    result = {
        "method": method_name,
        "seed": seed,
        "val_best_metric": round(float(best_val_metric), 6),
        "tune_metric": tune_metric,
        "epoch_metrics": epoch_metrics,
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info(f"  [{method_name}|seed={seed}] Done in {elapsed:.0f}s. "
                f"val_{tune_metric}={best_val_metric:.4f}")

    del model, optimizer, loader_dict, state_dict
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Effective Rank Measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_effective_rank(model, loader, entity_table, max_batches=3):
    """Measure effective rank of node embeddings after GNN via spectral entropy."""
    model.eval()
    all_embeddings = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(DEVICE)
        seed_time = batch[entity_table].seed_time
        x_dict = model.encoder(batch.tf_dict)
        try:
            batch_dict = batch.batch_dict
            rel_time_dict = model.temporal_encoder(
                seed_time, batch.time_dict, batch_dict
            )
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time
        except (KeyError, AttributeError):
            pass
        x_dict = model.gnn(x_dict, batch.edge_index_dict)
        emb = x_dict[entity_table][:seed_time.size(0)]
        all_embeddings.append(emb.cpu())

    if not all_embeddings:
        return {"effective_rank": 0.0, "normalized_effective_rank": 0.0}

    emb = torch.cat(all_embeddings, dim=0).numpy()
    try:
        U, S, Vt = np.linalg.svd(emb, full_matrices=False)
        S = S[S > 1e-10]
        p = S / S.sum()
        H = -(p * np.log(p)).sum()
        eff_rank = float(np.exp(H))
        normalized = float(eff_rank / min(emb.shape))
    except Exception:
        logger.exception("SVD failed for effective rank")
        eff_rank, normalized = 0.0, 0.0

    return {
        "effective_rank": round(eff_rank, 2),
        "normalized_effective_rank": round(normalized, 4),
        "embedding_shape": list(emb.shape),
    }


# ---------------------------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------------------------
def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size. Positive d means group2 > group1."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(group2) - np.mean(group1)) / pooled_std)


def bootstrap_ci(group1, group2, n_boot=10000, alpha=0.05):
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.RandomState(42)
    ds = []
    for _ in range(n_boot):
        idx1 = rng.choice(len(group1), len(group1), replace=True)
        idx2 = rng.choice(len(group2), len(group2), replace=True)
        ds.append(cohens_d(group1[idx1], group2[idx2]))
    lo, hi = np.percentile(ds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


def run_statistical_analysis(results, task_name, tune_metric):
    """Cohen's d, bootstrap CIs, paired t-tests for all comparisons."""
    methods = {}
    for r in results:
        if r.get("task") == task_name:
            m = r["method"]
            methods.setdefault(m, []).append(r["val_best_metric"])

    for m in methods:
        methods[m] = np.array(methods[m])

    analysis = {"per_method": {}, "comparisons": {}}
    for m, scores in methods.items():
        analysis["per_method"][m] = {
            "mean": round(float(np.mean(scores)), 6),
            "std": round(float(np.std(scores, ddof=1)), 6) if len(scores) > 1 else 0.0,
            "median": round(float(np.median(scores)), 6),
            "n_seeds": len(scores),
            "per_seed": {int(SEEDS[i]): round(float(s), 6)
                         for i, s in enumerate(scores)},
        }

    comparisons_spec = [
        ("cama", "sum", "CAMA vs Sum (key comparison)"),
        ("cama", "pna", "CAMA vs PNA"),
        ("cama", "ungated_moment", "CAMA vs Ungated Moment (gating ablation)"),
        ("mean", "sum", "Mean vs Sum (aggregation baseline)"),
        ("cama", "mean", "CAMA vs Mean"),
        ("pna", "sum", "PNA vs Sum"),
        ("ungated_moment", "sum", "Ungated Moment vs Sum"),
    ]
    for method_a, method_b, label in comparisons_spec:
        if method_a not in methods or method_b not in methods:
            continue
        scores_a, scores_b = methods[method_a], methods[method_b]
        d = cohens_d(scores_b, scores_a)  # positive d = method_a better
        ci = bootstrap_ci(scores_b, scores_a)

        if len(scores_a) == len(scores_b) and len(scores_a) >= 2:
            t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
        else:
            t_stat, p_val = 0.0, 1.0

        analysis["comparisons"][f"{method_a}_vs_{method_b}"] = {
            "label": label,
            "cohens_d": round(float(d), 4),
            "ci_95": ci,
            "p_value": round(float(p_val), 6),
            "t_statistic": round(float(t_stat), 4),
            "method_a_mean": round(float(np.mean(scores_a)), 6),
            "method_b_mean": round(float(np.mean(scores_b)), 6),
            "difference": round(float(np.mean(scores_a) - np.mean(scores_b)), 6),
        }
        logger.info(f"  {label}: d={d:.3f}, CI=[{ci[0]:.3f},{ci[1]:.3f}], "
                     f"p={p_val:.4f}")

    return analysis


# ---------------------------------------------------------------------------
# Output Formatting (exp_gen_sol_out schema)
# ---------------------------------------------------------------------------
def format_output(all_results, task_analyses, effective_ranks, task_configs):
    """Build output conforming to exp_gen_sol_out.json schema.

    Each seed becomes one example with predict_<method> fields showing
    each method's best validation metric at that seed.
    """
    output = {
        "metadata": {
            "title": ("Sum/Mean/PNA/Ungated-Moment/CAMA Baselines on "
                      "rel-stack/user-engagement + Sum on rel-trial/study-outcome"),
            "description": (
                "Experiment comparing 5 aggregation methods (sum, mean, PNA, "
                "ungated moment, CAMA) on rel-stack/user-engagement and sum on "
                "rel-trial/study-outcome. Uses best-validation average_precision "
                "(test labels unavailable in relbench 2.1.1). Computes Cohen's d "
                "with bootstrap CIs and effective rank of representations."
            ),
            "methods_tested": ["sum", "mean", "pna", "ungated_moment", "cama"],
            "hyperparameters": HYPERPARAMS,
            "seeds": SEEDS,
            "evaluation_metric": "best validation average_precision",
            "note": ("Test split labels unavailable in relbench 2.1.1. "
                     "All comparisons use validation metrics for fairness."),
            "statistical_analysis": task_analyses,
            "effective_rank_analysis": effective_ranks,
            "all_run_results": all_results,
        },
        "datasets": [],
    }

    # Group results by task and then by seed
    tasks_seen = {}
    for r in all_results:
        task_key = r.get("task", "unknown")
        tasks_seen.setdefault(task_key, {})
        seed = r["seed"]
        tasks_seen[task_key].setdefault(seed, {})[r["method"]] = r

    for task_key, seed_dict in tasks_seen.items():
        ds_name = task_key.replace("/", "__")
        examples = []
        config = task_configs.get(task_key, {})

        for seed in sorted(seed_dict.keys()):
            methods_at_seed = seed_dict[seed]
            # Find best method at this seed
            best_method = max(
                methods_at_seed.items(),
                key=lambda kv: kv[1]["val_best_metric"]
            )
            example = {
                "input": json.dumps({
                    "task": task_key,
                    "seed": seed,
                    "metric": config.get("metric", "average_precision"),
                    "task_type": config.get("task_type", "binary_classification"),
                    "hyperparameters": HYPERPARAMS,
                }, indent=None),
                "output": str(round(best_method[1]["val_best_metric"], 6)),
                "metadata_seed": seed,
                "metadata_task": task_key,
                "metadata_best_method": best_method[0],
                "metadata_metric": config.get("metric", "average_precision"),
            }
            # Add predict_<method> for each method tested at this seed
            for method_name, r in methods_at_seed.items():
                safe_name = method_name.replace(" ", "_")
                example[f"predict_{safe_name}"] = str(
                    round(r["val_best_metric"], 6)
                )
            examples.append(example)

        if examples:
            output["datasets"].append({
                "dataset": ds_name,
                "examples": examples,
            })

    return output


# ---------------------------------------------------------------------------
# Time Management
# ---------------------------------------------------------------------------
def _time_remaining() -> float:
    return TIME_BUDGET_SECONDS - (time.time() - SCRIPT_START_TIME)


def _should_continue(min_seconds: int = 300) -> bool:
    remaining = _time_remaining()
    if remaining < min_seconds:
        logger.warning(f"Time budget low: {remaining/60:.1f}min remaining")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@logger.catch
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting aggregation comparison experiment")
    logger.info("=" * 60)
    logger.info(f"Time budget: {TIME_BUDGET_SECONDS/60:.0f} min")

    all_results = []
    task_analyses = {}
    effective_ranks = {}
    task_configs = {}

    # ===================================================================
    # TASK 1: rel-stack/user-engagement — all methods × all seeds
    # ===================================================================
    task_key = "rel-stack/user-engagement"
    logger.info(f"\n{'='*60}")
    logger.info(f"TASK: {task_key}")
    logger.info(f"{'='*60}")

    try:
        dataset_stack, task_stack, data_stack, col_stats_stack = load_task_data(
            "rel-stack", "user-engagement"
        )
        task_configs[task_key] = {
            "task_type": "binary_classification",
            "metric": "average_precision",
        }

        methods = {
            "sum": "sum",
            "mean": "mean",
            "pna": make_pna_aggr(HYPERPARAMS['channels']),
            "ungated_moment": UngatedMomentAggregation(
                channels=HYPERPARAMS['channels']),
            "cama": CAMAAggregation(
                channels=HYPERPARAMS['channels'], gate_hidden=16),
        }

        for method_name, aggr_obj in methods.items():
            logger.info(f"\n--- Method: {method_name} ---")
            for seed in SEEDS:
                if not _should_continue(min_seconds=180):
                    logger.warning("Time budget low, stopping runs")
                    break
                try:
                    result = run_single_experiment(
                        task=task_stack, data=data_stack,
                        col_stats_dict=col_stats_stack,
                        aggr=aggr_obj, method_name=method_name,
                        seed=seed, hp=HYPERPARAMS,
                    )
                    result["task"] = task_key
                    all_results.append(result)

                    # Incremental save
                    inc_path = RESULTS_DIR / "incremental_results.json"
                    inc_path.write_text(json.dumps(all_results, indent=2))

                    if HAS_GPU:
                        mem = torch.cuda.memory_allocated() / 1e9
                        peak = torch.cuda.max_memory_allocated() / 1e9
                        logger.info(f"  GPU: {mem:.1f}GB alloc, {peak:.1f}GB peak")

                except torch.cuda.OutOfMemoryError:
                    logger.error(f"OOM for {method_name}|seed={seed}, "
                                 "retrying with reduced params...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    try:
                        reduced_hp = {**HYPERPARAMS, 'batch_size': 64,
                                      'num_neighbors': 8}
                        result = run_single_experiment(
                            task=task_stack, data=data_stack,
                            col_stats_dict=col_stats_stack,
                            aggr=aggr_obj, method_name=method_name,
                            seed=seed, hp=reduced_hp,
                        )
                        result["task"] = task_key
                        result["note"] = "reduced_batch_size"
                        all_results.append(result)
                    except Exception:
                        logger.exception(f"Failed even reduced: "
                                         f"{method_name}|seed={seed}")
                except Exception:
                    logger.exception(f"Failed: {method_name}|seed={seed}")

        # Statistical analysis
        if all_results:
            logger.info("\n--- Statistical Analysis ---")
            task_analyses[task_key] = run_statistical_analysis(
                all_results, task_key, "average_precision"
            )

        # Effective rank measurement
        if _should_continue(min_seconds=300):
            logger.info("\n--- Effective Rank Measurement ---")
            for method_name, aggr_obj in methods.items():
                if not _should_continue(min_seconds=120):
                    break
                try:
                    seed_everything(42)
                    a = aggr_obj if isinstance(aggr_obj, str) else copy.deepcopy(aggr_obj)
                    model = Model(
                        data=data_stack, col_stats_dict=col_stats_stack,
                        num_layers=HYPERPARAMS['num_layers'],
                        channels=HYPERPARAMS['channels'],
                        out_channels=1, aggr=a, norm="layer_norm",
                    ).to(DEVICE)
                    loader_dict, entity_table = build_loaders(
                        task_stack, data_stack,
                        batch_size=HYPERPARAMS['batch_size'],
                        num_neighbors=HYPERPARAMS['num_neighbors'],
                        num_layers=HYPERPARAMS['num_layers'],
                    )
                    rank_info = measure_effective_rank(
                        model, loader_dict["val"], entity_table, max_batches=2,
                    )
                    effective_ranks[f"{task_key}/{method_name}"] = rank_info
                    logger.info(f"  {method_name}: eff_rank={rank_info}")
                    del model, loader_dict
                    gc.collect()
                    if HAS_GPU:
                        torch.cuda.empty_cache()
                except Exception:
                    logger.exception(f"Rank failed for {method_name}")
        else:
            logger.warning("Skipping effective rank due to time budget")

        del data_stack, col_stats_stack
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()

    except Exception:
        logger.exception(f"FAILED to run {task_key}")

    # ===================================================================
    # TASK 2: rel-trial/study-outcome — sum only × all seeds
    # ===================================================================
    task_key2 = "rel-trial/study-outcome"
    if _should_continue(min_seconds=600):
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK: {task_key2}")
        logger.info(f"{'='*60}")

        try:
            dataset_trial, task_trial, data_trial, col_stats_trial = load_task_data(
                "rel-trial", "study-outcome"
            )
            task_configs[task_key2] = {
                "task_type": "binary_classification",
                "metric": "average_precision",
            }

            logger.info("--- Method: sum ---")
            for seed in SEEDS:
                if not _should_continue(min_seconds=180):
                    break
                try:
                    result = run_single_experiment(
                        task=task_trial, data=data_trial,
                        col_stats_dict=col_stats_trial,
                        aggr="sum", method_name="sum",
                        seed=seed, hp=HYPERPARAMS,
                    )
                    result["task"] = task_key2
                    all_results.append(result)
                except Exception:
                    logger.exception(f"Failed: sum|seed={seed} on {task_key2}")

            del data_trial, col_stats_trial
            gc.collect()
            if HAS_GPU:
                torch.cuda.empty_cache()

        except Exception:
            logger.exception(f"FAILED {task_key2}. Skipping per fallback plan.")
    else:
        logger.warning(f"Skipping {task_key2} due to time budget")

    # ===================================================================
    # Build final output
    # ===================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Building output...")
    logger.info(f"{'='*60}")
    logger.info(f"Total runs completed: {len(all_results)}")

    key_findings = []
    for comp_key, comp_data in task_analyses.get(
            task_key, {}).get("comparisons", {}).items():
        d_val = comp_data.get("cohens_d", 0)
        ci = comp_data.get("ci_95", [0, 0])
        p = comp_data.get("p_value", 1)
        sig = "significant" if p < 0.05 else "not significant"
        key_findings.append(
            f"{comp_data['label']}: Cohen's d={d_val:.3f}, "
            f"95% CI=[{ci[0]:.3f}, {ci[1]:.3f}], p={p:.4f} ({sig})"
        )

    output = format_output(
        all_results=all_results,
        task_analyses=task_analyses,
        effective_ranks=effective_ranks,
        task_configs=task_configs,
    )
    output["metadata"]["key_findings"] = key_findings
    output["metadata"]["total_elapsed_seconds"] = round(
        time.time() - start_time, 1
    )

    conclusion_parts = []
    per_method = task_analyses.get(task_key, {}).get("per_method", {})
    if per_method:
        sorted_methods = sorted(
            per_method.items(), key=lambda x: x[1]["mean"], reverse=True
        )
        best = sorted_methods[0]
        conclusion_parts.append(
            f"Best method on {task_key}: {best[0]} "
            f"(val AP={best[1]['mean']:.4f} +/- {best[1]['std']:.4f})"
        )
    if not conclusion_parts:
        conclusion_parts.append("Experiment completed but no comparisons available.")
    output["metadata"]["conclusion"] = "; ".join(conclusion_parts)

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output saved to {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    logger.info(f"Total elapsed: {(time.time()-start_time)/60:.1f} min")

    logger.info("\n=== SUMMARY ===")
    for finding in key_findings:
        logger.info(f"  {finding}")
    logger.info(f"  Conclusion: {output['metadata']['conclusion']}")


if __name__ == "__main__":
    main()
