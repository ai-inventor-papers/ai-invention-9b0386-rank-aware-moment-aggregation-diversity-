#!/usr/bin/env python3
"""Controlled RAMA vs Mean vs RAMA-no-rank on rel-f1 and rel-amazon.

Phase A: rel-f1 (driver-position regression + driver-dnf classification, 3 methods x 5 seeds)
Phase B: rel-amazon/item-ltv (3 methods x 5 seeds)

Resolves the 14% MAE discrepancy from iteration 2, fills the classification
evidence gap, and tests whether rank conditioning matters.
"""

import copy
import gc
import json
import math
import os
import resource
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from loguru import logger
from scipy import stats as sp_stats

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


def _current_ram_gb() -> float:
    for p in [
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ]:
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

logger.info(
    f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
    f"GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM)"
)
logger.info(f"Current RAM usage: {_current_ram_gb():.1f}GB")

# ---------------------------------------------------------------------------
# Memory limits
# ---------------------------------------------------------------------------
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.75 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
logger.info(f"RAM budget: {RAM_BUDGET_BYTES / 1e9:.1f}GB (75% of {TOTAL_RAM_GB:.1f}GB)")

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(0.90)
    logger.info(f"GPU memory: {_free / 1e9:.1f}GB free / {_total / 1e9:.1f}GB total")
    torch.set_num_threads(max(1, NUM_CPUS // 2))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 10
BATCH_SIZE = 512
CHANNELS = 128
NUM_LAYERS = 2
NUM_NEIGHBORS = 128
LR = 0.005
MAX_STEPS_F1 = 99999  # no limit for small F1
MAX_STEPS_AMAZON = 2000  # cap for large amazon

METHODS = ["standard_mean", "rama_full", "rama_no_rank"]

TASKS_F1 = [
    ("driver-position", "regression"),
    ("driver-dnf", "binary_classification"),
]

# Relbench cache
EXISTING_CACHE = Path(
    "/ai-inventor/aii_pipeline/runs/temp-debug-test_sbr/"
    "3_invention_loop/iter_1/gen_art/data_id3_it1__opus/"
    "temp/relbench_cache"
)
CACHE_DIR = str(EXISTING_CACHE)

# Track deviations
DEVIATIONS: List[str] = []


def log_memory(label: str = "") -> float:
    ram_gb = _current_ram_gb()
    gpu_info = ""
    if HAS_GPU:
        gpu_used = torch.cuda.memory_allocated() / 1e9
        gpu_info = f", GPU={gpu_used:.1f}GB"
    logger.info(f"[MEM {label}] RAM={ram_gb:.1f}/{TOTAL_RAM_GB:.0f}GB{gpu_info}")
    if ram_gb > TOTAL_RAM_GB * 0.85:
        logger.warning(
            f"RAM usage at {ram_gb / TOTAL_RAM_GB * 100:.0f}% - approaching limit!"
        )
    return ram_gb


# ===========================================================================
# Aggregation factory
# ===========================================================================
def make_aggr(method_name: str, channels: int):
    """Create fresh aggregation object for a method."""
    from rama import RAMAAggregation, RAMANoRankAggregation

    if method_name == "standard_mean":
        return "mean"
    elif method_name == "rama_full":
        return RAMAAggregation(channels=channels, gate_hidden=16, eps=1e-8)
    elif method_name == "rama_no_rank":
        return RAMANoRankAggregation(channels=channels, gate_hidden=16, eps=1e-8)
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ===========================================================================
# Unit tests
# ===========================================================================
def run_unit_tests():
    """Run all unit tests for aggregation modules."""
    from torch_geometric.nn.aggr import MeanAggregation
    from torch_geometric.nn.conv import SAGEConv

    from rama import RAMAAggregation, RAMANoRankAggregation

    logger.info("=" * 60)
    logger.info("UNIT TESTS")
    logger.info("=" * 60)

    ch = 128
    n_edges = 100
    n_groups = 20
    x = torch.randn(n_edges, ch)
    index = torch.randint(0, n_groups, (n_edges,))

    # Test all 3 methods
    for name, aggr_cls in [
        ("RAMAAggregation", RAMAAggregation),
        ("RAMANoRankAggregation", RAMANoRankAggregation),
    ]:
        logger.info(f"Testing {name}...")
        aggr = aggr_cls(channels=ch, gate_hidden=16)

        # a) Shape test
        out = aggr(x, index=index, dim_size=n_groups)
        assert out.shape == (
            n_groups,
            ch,
        ), f"{name}: Expected ({n_groups},{ch}), got {out.shape}"
        logger.info(f"  Shape OK: {out.shape}")

        # b) No NaN
        assert not torch.isnan(out).any(), f"{name}: NaN in output!"
        logger.info("  No NaN OK")

        # c) Zero-init test: should match mean
        mean_aggr = MeanAggregation()
        mean_out = mean_aggr(x, index=index, dim_size=n_groups)
        diff = (out - mean_out).abs().max().item()
        assert diff < 1e-3, f"{name}: init should match mean, diff={diff:.6f}"
        logger.info(f"  Zero-init == mean OK (diff={diff:.6f})")

        # d) Gradient flow
        out_grad = aggr(x, index=index, dim_size=n_groups)
        loss = out_grad.sum()
        loss.backward()
        for pname, p in aggr.named_parameters():
            assert p.grad is not None, f"{name}: No gradient for {pname}"
        logger.info("  Gradient flow OK")

        # e) SAGEConv integration
        conv = SAGEConv((ch, ch), ch, aggr=aggr_cls(channels=ch, gate_hidden=16))
        x_src = torch.randn(50, ch)
        x_dst = torch.randn(30, ch)
        edge_index = torch.stack(
            [torch.randint(0, 50, (80,)), torch.randint(0, 30, (80,))]
        )
        conv_out = conv((x_src, x_dst), edge_index)
        assert conv_out.shape == (30, ch), f"{name}: SAGEConv shape wrong"
        assert not torch.isnan(conv_out).any(), f"{name}: SAGEConv NaN"
        logger.info("  SAGEConv integration OK")

        # f) Edge cases
        # Single child per group
        x_single = torch.randn(5, ch)
        idx_single = torch.arange(5)
        out_single = aggr_cls(channels=ch)(x_single, index=idx_single, dim_size=5)
        assert not torch.isnan(out_single).any(), f"{name}: single-child NaN"
        logger.info("  Edge case: single child OK")

        # Identical children
        x_ident = torch.ones(10, ch) * 3.0
        idx_ident = torch.zeros(10, dtype=torch.long)
        out_ident = aggr_cls(channels=ch)(x_ident, index=idx_ident, dim_size=1)
        assert not torch.isnan(out_ident).any(), f"{name}: identical children NaN"
        logger.info("  Edge case: identical children OK")

    # Standard mean test (string)
    logger.info("Testing standard_mean (string 'mean')...")
    conv_mean = SAGEConv((ch, ch), ch, aggr="mean")
    conv_mean_out = conv_mean((x_src, x_dst), edge_index)
    assert conv_mean_out.shape == (30, ch)
    assert not torch.isnan(conv_mean_out).any()
    logger.info("  standard_mean OK")

    logger.info("ALL UNIT TESTS PASSED!")
    # Cleanup
    del x, index, x_src, x_dst, edge_index
    gc.collect()


# ===========================================================================
# NaN preprocessing for TensorFrame data
# ===========================================================================
def fill_nan_in_graph(data: Any) -> Any:
    """Fill NaN values in TensorFrame features with 0 to prevent encoder NaN.

    This must be called AFTER make_pkey_fkey_graph and BEFORE training.
    Fixes: torch_frame numerical encoders propagate NaN through linear layers.
    """
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
                    logger.debug(
                        f"  Filled {cnt} NaN in {nt}.tf.feat_dict[{key}]"
                    )
    if nan_count > 0:
        logger.info(f"Filled {nan_count} total NaN values in TensorFrame features")
    return data


# ===========================================================================
# Dataset loading
# ===========================================================================
def load_f1_data() -> Tuple[Any, Dict]:
    """Load rel-f1 dataset and build graph. No text embeddings needed."""
    from relbench.datasets import get_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from torch_frame import stype as st

    logger.info("Loading rel-f1 dataset...")
    t0 = time.time()

    os.environ["RELBENCH_CACHE_DIR"] = CACHE_DIR
    dataset = get_dataset("rel-f1", download=True)
    db = dataset.get_db()

    logger.info(f"Dataset loaded in {time.time() - t0:.0f}s")
    logger.info(f"Tables: {list(db.table_dict.keys())}")
    log_memory("after f1 dataset load")

    # Get stype proposal and remove text_embedded columns
    col_to_stype_dict = get_stype_proposal(db)
    for table_name in col_to_stype_dict:
        col_to_stype_dict[table_name] = {
            c: s
            for c, s in col_to_stype_dict[table_name].items()
            if s != st.text_embedded
        }

    logger.info("Building rel-f1 graph (no text embeddings)...")
    mat_cache = str(WS / "mat_cache" / "rel-f1" / "materialized")
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

    # Fill NaN in TensorFrame features
    data = fill_nan_in_graph(data)
    log_memory("after f1 graph build")

    del db
    gc.collect()
    return data, col_stats_dict


def load_amazon_data() -> Tuple[Any, Dict]:
    """Load rel-amazon dataset with OOM-safe text exclusions."""
    from relbench.datasets import get_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from torch_frame import stype as st

    logger.info("Loading rel-amazon dataset...")
    t0 = time.time()

    os.environ["RELBENCH_CACHE_DIR"] = CACHE_DIR
    dataset = get_dataset("rel-amazon", download=True)
    db = dataset.get_db()

    logger.info(f"Dataset loaded in {time.time() - t0:.0f}s")
    logger.info(f"Tables: {list(db.table_dict.keys())}")
    log_memory("after amazon dataset load")

    # Get stypes
    stypes_cache_path = Path(f"{CACHE_DIR}/rel-amazon/stypes.json")
    if stypes_cache_path.exists():
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table in col_to_stype_dict:
            for col, stype_str in list(col_to_stype_dict[table].items()):
                col_to_stype_dict[table][col] = st(stype_str)
        logger.info("Loaded cached stypes")
    else:
        logger.info("Computing stype proposal...")
        col_to_stype_dict = get_stype_proposal(db)

    # CRITICAL: Remove text_embedded from review table (20.8M rows)
    if "review" in col_to_stype_dict:
        for text_col in ["review_text", "summary"]:
            if text_col in col_to_stype_dict["review"]:
                removed = col_to_stype_dict["review"].pop(text_col)
                logger.warning(
                    f"REMOVED review.{text_col} ({removed}) - 20.8M rows OOM risk"
                )
        DEVIATIONS.append(
            "Removed review_text and summary from review table to prevent OOM"
        )

    # Remove customer_name (1.85M rows)
    if "customer" in col_to_stype_dict:
        if "customer_name" in col_to_stype_dict["customer"]:
            col_to_stype_dict["customer"].pop("customer_name")
            logger.warning("REMOVED customer.customer_name")
            DEVIATIONS.append("Removed customer_name from customer table")

    # Remove ALL text_embedded to be safe (fallback 1)
    for table_name in col_to_stype_dict:
        removed_cols = [
            c
            for c, s in col_to_stype_dict[table_name].items()
            if s == st.text_embedded
        ]
        for c in removed_cols:
            col_to_stype_dict[table_name].pop(c)
            logger.warning(f"REMOVED {table_name}.{c} (text_embedded)")
    DEVIATIONS.append(
        "Excluded ALL text_embedded columns from all tables for OOM safety"
    )

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

    # Fill NaN in TensorFrame features
    data = fill_nan_in_graph(data)
    log_memory("after amazon graph build")

    del db
    gc.collect()
    return data, col_stats_dict


# ===========================================================================
# Loaders
# ===========================================================================
def build_loaders(
    dataset_name: str,
    task_name: str,
    data: Any,
    num_neighbors: int,
    batch_size: int,
) -> Tuple[Dict[str, Any], Any]:
    """Build train/val/test neighbor loaders. Returns (loaders, task)."""
    from relbench.modeling.graph import get_node_train_table_input
    from relbench.tasks import get_task
    from torch_geometric.loader import NeighborLoader

    os.environ["RELBENCH_CACHE_DIR"] = CACHE_DIR
    task = get_task(dataset_name, task_name, download=True)

    loader_dict = {}
    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[int(num_neighbors / 2**i) for i in range(NUM_LAYERS)],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=batch_size,
            temporal_strategy="uniform",
            shuffle=(split == "train"),
            num_workers=0,
        )
    return loader_dict, task


# ===========================================================================
# Model creation
# ===========================================================================
def create_model(data: Any, col_stats_dict: Any, method_name: str) -> Any:
    from model import Model

    aggr = make_aggr(method_name, CHANNELS)

    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=NUM_LAYERS,
        channels=CHANNELS,
        out_channels=1,
        aggr=aggr,
        norm="batch_norm",
    ).to(DEVICE)
    return model


# ===========================================================================
# Training
# ===========================================================================
def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    entity_table: str,
    task_type: str,
    max_steps: int,
) -> float:
    model.train()
    loss_accum = count_accum = 0
    steps = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch, entity_table).view(-1)
        y = batch[entity_table].y.float()

        # Match shapes: only take first batch_size elements
        bs = min(pred.size(0), y.size(0))
        pred = pred[:bs]
        y = y[:bs]

        # Mask out NaN targets (and NaN predictions for safety)
        mask = ~torch.isnan(y) & ~torch.isnan(pred)
        if not mask.any():
            continue

        pred_m = pred[mask]
        y_m = y[mask]

        # NOTE: Do NOT clamp during training - kills gradients when
        # initial predictions are outside clamp range.
        # Clamping is applied only during evaluation.

        loss = loss_fn(pred_m.float(), y_m)

        if torch.isnan(loss):
            logger.warning("NaN loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_accum += loss.detach().item() * pred_m.size(0)
        count_accum += pred_m.size(0)
        steps += 1
        if steps >= max_steps:
            break
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
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table).view(-1)
        if clamp_min is not None:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        # Replace NaN predictions with default
        pred = torch.nan_to_num(pred, nan=default_val)
        preds.append(pred.cpu())
    result = torch.cat(preds, dim=0).numpy()
    # Final NaN safety
    result = np.nan_to_num(result, nan=default_val)
    return result


# ===========================================================================
# Gate statistics
# ===========================================================================
def extract_gate_stats(model: torch.nn.Module) -> Dict[str, Any]:
    from rama import RAMAAggregation, RAMANoRankAggregation

    def _safe_float(t: torch.Tensor) -> float:
        v = float(t)
        return 0.0 if (v != v) else v  # NaN check

    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (RAMAAggregation, RAMANoRankAggregation)):
            try:
                with torch.no_grad():
                    gate_bias = module.gate_out.bias.data.detach().cpu().float()
                    w_sigma = module.W_sigma.weight.data.detach().cpu().float()
                    w_g = module.W_g.weight.data.detach().cpu().float()
                    gate_bias = torch.nan_to_num(gate_bias, nan=0.0)
                    stats[name] = {
                        "gate_bias_mean": _safe_float(gate_bias.mean()),
                        "gate_bias_std": _safe_float(gate_bias.std()),
                        "gate_bias_sigmoid_mean": _safe_float(
                            torch.sigmoid(gate_bias).mean()
                        ),
                        "W_sigma_norm": _safe_float(w_sigma.norm()),
                        "W_g_norm": _safe_float(w_g.norm()),
                        "type": type(module).__name__,
                    }
            except Exception as e:
                stats[name] = {"error": str(e), "type": type(module).__name__}
    return stats


# ===========================================================================
# Single run
# ===========================================================================
def run_single(
    dataset_name: str,
    task_name: str,
    task_type: str,
    method_name: str,
    seed: int,
    data: Any,
    col_stats_dict: Any,
    loader_dict: Dict[str, Any],
    task: Any,
    epochs: int,
    max_steps: int,
    instrumented: bool = False,
) -> Dict[str, Any]:
    from torch_geometric.seed import seed_everything

    logger.info(
        f"  RUN: {dataset_name}/{task_name} | {method_name} | seed={seed} | "
        f"epochs={epochs}"
    )

    seed_everything(seed)

    model = create_model(data, col_stats_dict, method_name)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"    Params: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    entity_table = task.entity_table
    train_table = task.get_table("train")

    # Setup loss and clamping
    if task_type == "regression":
        loss_fn = torch.nn.L1Loss()
        targets = train_table.df[task.target_col].dropna().to_numpy().astype(float)
        clamp_min = float(np.percentile(targets, 2))
        clamp_max = float(np.percentile(targets, 98))
        default_pred = float(np.median(targets))
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
        clamp_min = clamp_max = None
        default_pred = 0.0

    best_val_metric = None
    best_state = None
    epoch_times = []
    train_losses = []
    higher_is_better = task_type != "regression"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            loader_dict["train"],
            optimizer,
            loss_fn,
            entity_table,
            task_type,
            max_steps,
        )
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        train_losses.append(train_loss)

        # Validation
        val_pred = evaluate_model(
            model, loader_dict["val"], entity_table, clamp_min, clamp_max,
            default_val=default_pred,
        )
        val_table = task.get_table("val")
        try:
            val_metrics = task.evaluate(val_pred, val_table)
        except Exception as e:
            logger.warning(f"    Val eval failed: {e}, using fallback")
            val_metrics = {"mae": float("inf"), "average_precision": 0.0}

        if task_type == "regression":
            val_m = val_metrics.get("mae", float("inf"))
            is_better = best_val_metric is None or val_m < best_val_metric
        else:
            val_m = val_metrics.get(
                "average_precision",
                val_metrics.get("roc_auc", val_metrics.get("f1", 0.0)),
            )
            is_better = best_val_metric is None or val_m > best_val_metric

        if is_better:
            best_val_metric = val_m
            best_state = copy.deepcopy(model.state_dict())

        logger.info(
            f"    epoch={epoch}/{epochs} loss={train_loss:.4f} "
            f"val_metric={val_m:.4f} best={best_val_metric:.4f} "
            f"time={epoch_time:.1f}s"
        )

    # Test evaluation with best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test split; fall back to val if test lacks targets
    test_metrics = None
    for eval_split, eval_loader_key in [("test", "test"), ("val", "val")]:
        try:
            eval_table = task.get_table(eval_split)
            # Check if target column exists in the table
            if task.target_col not in eval_table.df.columns:
                logger.info(f"    {eval_split} table lacks '{task.target_col}', skipping")
                continue
            eval_pred = evaluate_model(
                model, loader_dict[eval_loader_key], entity_table,
                clamp_min, clamp_max, default_val=default_pred,
            )
            test_metrics = task.evaluate(eval_pred, eval_table)
            test_metrics = {k: float(v) for k, v in test_metrics.items()}
            if eval_split == "val":
                test_metrics["_note"] = "evaluated on val split (test lacks targets)"
                DEVIATIONS.append(f"Used val split for test eval on {task_name}")
            break
        except Exception:
            logger.exception(f"    {eval_split} evaluation failed")
            continue

    if test_metrics is None:
        logger.warning("    Both test and val evaluation failed, using fallback")
        if best_val_metric is not None:
            test_metrics = {"fallback_val_metric": float(best_val_metric)}
        else:
            test_metrics = {"mae": float("inf"), "average_precision": 0.0}

    # Gate stats for RAMA methods on first seed
    gate_stats = {}
    if instrumented and method_name.startswith("rama"):
        gate_stats = extract_gate_stats(model)

    # Get predictions for the split we evaluated on (for examples)
    final_pred = evaluate_model(
        model, loader_dict["val"], entity_table, clamp_min, clamp_max,
        default_val=default_pred,
    )

    result = {
        "dataset": dataset_name,
        "task": task_name,
        "task_type": task_type,
        "method": method_name,
        "seed": seed,
        "test_metrics": test_metrics,
        "best_val_metric": float(best_val_metric) if best_val_metric else None,
        "train_losses": [float(x) for x in train_losses],
        "epoch_times": [float(x) for x in epoch_times],
        "mean_epoch_time": float(np.mean(epoch_times)),
        "total_time": float(sum(epoch_times)),
        "gate_stats": gate_stats,
        "num_params": param_count,
        "test_predictions": final_pred.tolist(),
    }

    del model, best_state
    if HAS_GPU:
        torch.cuda.empty_cache()
    gc.collect()
    return result


# ===========================================================================
# Incremental save
# ===========================================================================
def save_incremental(results: List[Dict], label: str = "incremental"):
    out_path = WS / f"results_{label}.json"
    safe = [
        {k: v for k, v in r.items() if k != "test_predictions"} for r in results
    ]
    out_path.write_text(json.dumps(safe, indent=2))
    logger.info(f"Saved {len(safe)} results to {out_path.name}")


# ===========================================================================
# Statistical analysis
# ===========================================================================
def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Cohen's d. Positive means group2 > group1."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return 0.0
    return float((m2 - m1) / pooled)


def ci_cohens_d(d: float, n1: int, n2: int, alpha: float = 0.05) -> List[float]:
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    z = sp_stats.norm.ppf(1 - alpha / 2)
    return [float(d - z * se), float(d + z * se)]


def analyze_results(
    all_results: List[Dict],
) -> Dict[str, Any]:
    """Full statistical analysis across all tasks and methods."""
    analysis: Dict[str, Any] = {}

    # Group results by task
    tasks = set((r["dataset"], r["task"], r["task_type"]) for r in all_results)

    per_task: Dict[str, Any] = {}
    all_ds_full_vs_mean = []
    all_ds_norank_vs_mean = []
    all_ds_full_vs_norank = []
    all_ns = []

    for ds, task_name, task_type in sorted(tasks):
        task_key = f"{ds}/{task_name}"
        task_results = [
            r for r in all_results if r["dataset"] == ds and r["task"] == task_name
        ]

        if task_type == "regression":
            metric_key = "mae"
            # For regression, lower is better
            # Positive d = method is better (lower MAE)
            sign = -1  # flip: baseline - method, positive = method better
        else:
            metric_key = "average_precision"
            sign = 1  # positive = method better (higher AP)

        def get_metric_vals(method: str) -> List[float]:
            vals = []
            for r in task_results:
                if r["method"] == method:
                    v = r["test_metrics"].get(metric_key)
                    if v is not None and not np.isnan(v):
                        vals.append(float(v))
            return vals

        baseline_vals = get_metric_vals("standard_mean")
        rama_full_vals = get_metric_vals("rama_full")
        rama_norank_vals = get_metric_vals("rama_no_rank")

        comparisons: Dict[str, Any] = {}
        n = min(len(baseline_vals), len(rama_full_vals))

        # rama_full vs mean
        if len(baseline_vals) >= 2 and len(rama_full_vals) >= 2:
            raw_d = cohens_d(baseline_vals, rama_full_vals)
            d = raw_d * sign
            ci = ci_cohens_d(d, len(baseline_vals), len(rama_full_vals))
            t_stat, p_val = (
                sp_stats.ttest_rel(baseline_vals, rama_full_vals)
                if len(baseline_vals) == len(rama_full_vals)
                else sp_stats.ttest_ind(baseline_vals, rama_full_vals)
            )
            try:
                w_stat, w_p = sp_stats.wilcoxon(
                    baseline_vals, rama_full_vals
                )
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")

            diffs = np.array(rama_full_vals) - np.array(baseline_vals)
            se_diff = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
            diff_ci = [
                float(np.mean(diffs) - 1.96 * se_diff),
                float(np.mean(diffs) + 1.96 * se_diff),
            ]

            comparisons["rama_full_vs_mean"] = {
                "cohens_d": float(d),
                "cohens_d_ci_95": ci,
                "p_value": float(p_val),
                "t_statistic": float(t_stat),
                "wilcoxon_p": float(w_p),
                "mean_difference": float(np.mean(diffs)),
                "difference_ci_95": diff_ci,
                "baseline_mean": float(np.mean(baseline_vals)),
                "baseline_std": float(np.std(baseline_vals, ddof=1)),
                "rama_full_mean": float(np.mean(rama_full_vals)),
                "rama_full_std": float(np.std(rama_full_vals, ddof=1)),
            }
            all_ds_full_vs_mean.append(d)
            all_ns.append(n)

        # rama_no_rank vs mean
        if len(baseline_vals) >= 2 and len(rama_norank_vals) >= 2:
            raw_d = cohens_d(baseline_vals, rama_norank_vals)
            d = raw_d * sign
            ci = ci_cohens_d(d, len(baseline_vals), len(rama_norank_vals))
            t_stat, p_val = (
                sp_stats.ttest_rel(baseline_vals, rama_norank_vals)
                if len(baseline_vals) == len(rama_norank_vals)
                else sp_stats.ttest_ind(baseline_vals, rama_norank_vals)
            )
            diffs = np.array(rama_norank_vals) - np.array(baseline_vals)
            se_diff = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
            diff_ci = [
                float(np.mean(diffs) - 1.96 * se_diff),
                float(np.mean(diffs) + 1.96 * se_diff),
            ]
            comparisons["rama_no_rank_vs_mean"] = {
                "cohens_d": float(d),
                "cohens_d_ci_95": ci,
                "p_value": float(p_val),
                "mean_difference": float(np.mean(diffs)),
                "difference_ci_95": diff_ci,
                "rama_no_rank_mean": float(np.mean(rama_norank_vals)),
                "rama_no_rank_std": float(np.std(rama_norank_vals, ddof=1)),
            }
            all_ds_norank_vs_mean.append(d)

        # rama_full vs rama_no_rank (rank ablation)
        if len(rama_full_vals) >= 2 and len(rama_norank_vals) >= 2:
            raw_d = cohens_d(rama_norank_vals, rama_full_vals)
            d = raw_d * sign
            ci = ci_cohens_d(d, len(rama_norank_vals), len(rama_full_vals))
            t_stat, p_val = (
                sp_stats.ttest_rel(rama_norank_vals, rama_full_vals)
                if len(rama_norank_vals) == len(rama_full_vals)
                else sp_stats.ttest_ind(rama_norank_vals, rama_full_vals)
            )
            comparisons["rama_full_vs_no_rank"] = {
                "cohens_d": float(d),
                "cohens_d_ci_95": ci,
                "p_value": float(p_val),
                "interpretation": (
                    "Positive d means full RAMA (with rank) is better than no-rank"
                ),
            }
            all_ds_full_vs_norank.append(d)

        per_task[task_key] = {
            "metric_key": metric_key,
            "higher_is_better": task_type != "regression",
            "n_seeds": n,
            "comparisons": comparisons,
            "raw_values": {
                "baseline": baseline_vals,
                "rama_full": rama_full_vals,
                "rama_no_rank": rama_norank_vals,
            },
        }

    analysis["per_task"] = per_task

    # DerSimonian-Laird pooled effects
    def pool_effects(ds_list: List[float], ns_list: List[int]) -> Dict:
        if not ds_list or not ns_list:
            return {"pooled_d": None, "pooled_ci_95": None}
        all_vars = [2 / max(n, 2) + d**2 / (2 * max(n, 2)) for d, n in zip(ds_list, ns_list)]
        weights = [1.0 / max(v, 1e-12) for v in all_vars]
        w_sum = sum(weights)
        pooled_d = sum(d * w for d, w in zip(ds_list, weights)) / max(w_sum, 1e-12)
        pooled_var = 1.0 / max(w_sum, 1e-12)
        pooled_ci = [
            float(pooled_d - 1.96 * np.sqrt(pooled_var)),
            float(pooled_d + 1.96 * np.sqrt(pooled_var)),
        ]
        return {
            "pooled_d": float(pooled_d),
            "pooled_ci_95": pooled_ci,
            "per_task_ds": [float(d) for d in ds_list],
            "n_tasks": len(ds_list),
        }

    analysis["pooled_effects"] = {
        "rama_full_vs_mean": pool_effects(all_ds_full_vs_mean, all_ns),
        "rama_no_rank_vs_mean": pool_effects(all_ds_norank_vs_mean, all_ns),
        "rama_full_vs_no_rank": pool_effects(all_ds_full_vs_norank, all_ns),
    }

    # Rank ablation summary
    rank_helps = [d > 0 for d in all_ds_full_vs_norank]
    analysis["rank_ablation_summary"] = {
        "per_task_d": [float(d) for d in all_ds_full_vs_norank],
        "rank_helps_count": sum(rank_helps),
        "total_tasks": len(all_ds_full_vs_norank),
        "conclusion": (
            "rank conditioning helps"
            if sum(rank_helps) > len(rank_helps) / 2
            else "rank conditioning does not clearly help"
        ),
    }

    return analysis


# ===========================================================================
# Build output JSON (exp_gen_sol_out schema)
# ===========================================================================
def build_output(
    all_results: List[Dict],
    analysis: Dict[str, Any],
    timing: Dict[str, float],
    mini: bool = False,
) -> Dict[str, Any]:
    """Build method_out.json conforming to exp_gen_sol_out schema."""

    # Load dependency examples for datasets section
    dep_data3_path = Path(
        "/ai-inventor/aii_pipeline/runs/leskovec-predictive-residual-message-passing-v2_sti/"
        "3_invention_loop/iter_1/gen_art/data_id3_it1__opus/mini_data_out.json"
    )
    dep_data4_path = Path(
        "/ai-inventor/aii_pipeline/runs/leskovec-predictive-residual-message-passing-v2_sti/"
        "3_invention_loop/iter_1/gen_art/data_id4_it1__opus/mini_data_out.json"
    )

    dep_examples: Dict[str, List[Dict]] = {}
    for dep_path, dep_name in [
        (dep_data3_path, "data_id3"),
        (dep_data4_path, "data_id4"),
    ]:
        try:
            dep_data = json.loads(dep_path.read_text())
            for ds_entry in dep_data.get("datasets", []):
                ds_name = ds_entry.get("dataset", "")
                dep_examples[ds_name] = ds_entry.get("examples", [])[:3]
        except Exception:
            logger.exception(f"Failed to load dependency {dep_name}")

    # Group results by dataset/task
    task_groups: Dict[str, List[Dict]] = {}
    for r in all_results:
        key = f"{r['dataset']}/{r['task']}"
        if key not in task_groups:
            task_groups[key] = []
        task_groups[key].append(r)

    # Build datasets section
    datasets_out = []

    for task_key, task_results in sorted(task_groups.items()):
        ds_name, tname = task_key.split("/", 1)

        # Find best prediction per method for examples
        best_preds: Dict[str, List[float]] = {}
        for method in METHODS:
            method_runs = [r for r in task_results if r["method"] == method]
            if method_runs:
                # Best by val metric
                task_type = method_runs[0]["task_type"]
                if task_type == "regression":
                    best = min(method_runs, key=lambda r: r.get("best_val_metric", float("inf")))
                else:
                    best = max(method_runs, key=lambda r: r.get("best_val_metric", 0))
                if "test_predictions" in best and best["test_predictions"]:
                    best_preds[method] = best["test_predictions"]

        # Get dependency examples for this task
        dep_key_options = [
            task_key,
            f"{ds_name}/{tname}",
            f"{ds_name}__{tname}",
            f"rel-{ds_name.replace('rel-', '')}/{tname}",
        ]
        source_examples = []
        for dk in dep_key_options:
            if dk in dep_examples:
                source_examples = dep_examples[dk]
                break

        # Build examples: use test predictions from each method
        examples = []
        n_examples = min(
            500,
            len(best_preds.get("standard_mean", [])) if best_preds else 0,
        )

        if source_examples:
            # Use dependency examples as base
            for i, ex in enumerate(source_examples):
                example: Dict[str, Any] = {
                    "input": ex.get("input", ""),
                    "output": ex.get("output", ""),
                }
                for method in METHODS:
                    if method in best_preds and i < len(best_preds[method]):
                        pred_val = best_preds[method][i]
                        example[f"predict_{method}"] = str(round(float(pred_val), 6))
                # Copy metadata
                for k, v in ex.items():
                    if k.startswith("metadata_"):
                        example[k] = v
                examples.append(example)
        elif n_examples > 0:
            # Build from test predictions directly
            for i in range(n_examples):
                example = {
                    "input": json.dumps({"index": i, "task": task_key}),
                    "output": "0",  # placeholder
                    "metadata_task_type": task_results[0]["task_type"],
                    "metadata_dataset": ds_name,
                    "metadata_task": tname,
                }
                for method in METHODS:
                    if method in best_preds and i < len(best_preds[method]):
                        pred_val = best_preds[method][i]
                        example[f"predict_{method}"] = str(
                            round(float(pred_val), 6)
                        )
                examples.append(example)

        if examples:
            datasets_out.append({"dataset": task_key, "examples": examples})

    # Gate analysis
    gate_analysis: Dict[str, Any] = {}
    for r in all_results:
        if r.get("gate_stats"):
            key = f"{r['dataset']}/{r['task']}/{r['method']}/seed_{r['seed']}"
            gate_analysis[key] = r["gate_stats"]

    # Per-run results (without test_predictions to save space)
    per_run = [
        {k: v for k, v in r.items() if k != "test_predictions"} for r in all_results
    ]

    output = {
        "metadata": {
            "method_name": "RAMA Controlled Comparison",
            "description": (
                "Controlled experiment: 3 aggregation methods (standard_mean, rama_full, "
                "rama_no_rank) on rel-f1 (driver-position + driver-dnf, 5 seeds) and "
                "rel-amazon/item-ltv (5 seeds). Tests whether rank conditioning matters "
                "and fills the classification evidence gap."
            ),
            "methods": {
                "standard_mean": "Standard mean aggregation (baseline)",
                "rama_full": "RAMA with rank + cardinality gating",
                "rama_no_rank": "RAMA ablation with fixed rho=0.5 (cardinality-only)",
            },
            "hyperparameters": {
                "channels": CHANNELS,
                "lr": LR,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "num_neighbors": NUM_NEIGHBORS,
                "num_layers": NUM_LAYERS,
                "rama_gate_hidden": 16,
                "max_steps_f1": MAX_STEPS_F1,
                "max_steps_amazon": MAX_STEPS_AMAZON,
            },
            "seeds": SEEDS if not mini else SEEDS[:1],
            "statistical_analysis": analysis,
            "gate_analysis": gate_analysis,
            "timing": timing,
            "per_run_results": per_run,
            "deviations_from_plan": DEVIATIONS,
        },
        "datasets": datasets_out,
    }
    return output


# ===========================================================================
# Phase A: rel-f1
# ===========================================================================
def run_phase_a(
    mini: bool = False,
) -> List[Dict]:
    """Phase A: rel-f1 with both tasks."""
    logger.info("=" * 60)
    logger.info("PHASE A: rel-f1 (driver-position + driver-dnf)")
    logger.info("=" * 60)

    seeds = SEEDS[:1] if mini else SEEDS
    epochs = 3 if mini else EPOCHS

    # Load data once
    data, col_stats_dict = load_f1_data()

    results: List[Dict] = []

    for task_name, task_type in TASKS_F1:
        logger.info(f"\n--- Task: {task_name} ({task_type}) ---")

        # Build loaders for this task
        try:
            loader_dict, task = build_loaders(
                "rel-f1", task_name, data, NUM_NEIGHBORS, BATCH_SIZE
            )
        except Exception:
            logger.exception(f"Failed to build loaders for {task_name}")
            continue

        for method in METHODS:
            for seed in seeds:
                instrumented = seed == seeds[0]
                try:
                    r = run_single(
                        dataset_name="rel-f1",
                        task_name=task_name,
                        task_type=task_type,
                        method_name=method,
                        seed=seed,
                        data=data,
                        col_stats_dict=col_stats_dict,
                        loader_dict=loader_dict,
                        task=task,
                        epochs=epochs,
                        max_steps=MAX_STEPS_F1,
                        instrumented=instrumented,
                    )
                    results.append(r)
                    save_incremental(results, "phase_a")
                    log_memory(f"after {task_name}/{method}/s{seed}")

                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        f"GPU OOM: {task_name}/{method}/s{seed}, skipping"
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    DEVIATIONS.append(
                        f"GPU OOM skipped: {task_name}/{method}/s{seed}"
                    )
                except Exception:
                    logger.exception(
                        f"Failed: {task_name}/{method}/s{seed}"
                    )
                    DEVIATIONS.append(
                        f"Exception skipped: {task_name}/{method}/s{seed}"
                    )

                gc.collect()
                if HAS_GPU:
                    torch.cuda.empty_cache()

    # Free F1 data
    del data, col_stats_dict
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()
    log_memory("after phase A cleanup")

    return results


# ===========================================================================
# Phase B: rel-amazon
# ===========================================================================
def run_phase_b(
    mini: bool = False,
) -> List[Dict]:
    """Phase B: rel-amazon/item-ltv."""
    logger.info("=" * 60)
    logger.info("PHASE B: rel-amazon/item-ltv")
    logger.info("=" * 60)

    seeds = SEEDS[:1] if mini else SEEDS
    epochs = 3 if mini else EPOCHS

    # Load data once
    data, col_stats_dict = load_amazon_data()

    # Build loaders
    actual_neighbors = NUM_NEIGHBORS
    actual_batch = BATCH_SIZE
    try:
        loader_dict, task = build_loaders(
            "rel-amazon", "item-ltv", data, actual_neighbors, actual_batch
        )
    except Exception:
        logger.exception("Failed to build Amazon loaders, trying reduced settings")
        actual_neighbors = 64
        actual_batch = 256
        DEVIATIONS.append(
            f"Amazon loaders: reduced neighbors={actual_neighbors}, batch={actual_batch}"
        )
        loader_dict, task = build_loaders(
            "rel-amazon", "item-ltv", data, actual_neighbors, actual_batch
        )

    results: List[Dict] = []

    for method in METHODS:
        for seed in seeds:
            instrumented = seed == seeds[0]
            try:
                r = run_single(
                    dataset_name="rel-amazon",
                    task_name="item-ltv",
                    task_type="regression",
                    method_name=method,
                    seed=seed,
                    data=data,
                    col_stats_dict=col_stats_dict,
                    loader_dict=loader_dict,
                    task=task,
                    epochs=epochs,
                    max_steps=MAX_STEPS_AMAZON,
                    instrumented=instrumented,
                )
                results.append(r)
                save_incremental(results, "phase_b")
                log_memory(f"after amazon/{method}/s{seed}")

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"GPU OOM: amazon/{method}/s{seed}")
                torch.cuda.empty_cache()
                gc.collect()

                if actual_neighbors > 64:
                    logger.info("Retrying with reduced neighbors=64, batch=256")
                    actual_neighbors = 64
                    actual_batch = 256
                    DEVIATIONS.append(
                        f"Amazon OOM fallback: neighbors={actual_neighbors}, "
                        f"batch={actual_batch}"
                    )
                    loader_dict, task = build_loaders(
                        "rel-amazon",
                        "item-ltv",
                        data,
                        actual_neighbors,
                        actual_batch,
                    )
                    try:
                        r = run_single(
                            dataset_name="rel-amazon",
                            task_name="item-ltv",
                            task_type="regression",
                            method_name=method,
                            seed=seed,
                            data=data,
                            col_stats_dict=col_stats_dict,
                            loader_dict=loader_dict,
                            task=task,
                            epochs=epochs,
                            max_steps=MAX_STEPS_AMAZON,
                            instrumented=instrumented,
                        )
                        results.append(r)
                        save_incremental(results, "phase_b")
                    except Exception:
                        logger.exception(
                            f"Retry failed: amazon/{method}/s{seed}"
                        )
                        DEVIATIONS.append(
                            f"Retry failed: amazon/{method}/s{seed}"
                        )
                else:
                    DEVIATIONS.append(
                        f"GPU OOM skipped (already reduced): amazon/{method}/s{seed}"
                    )
            except Exception:
                logger.exception(f"Failed: amazon/{method}/s{seed}")
                DEVIATIONS.append(f"Exception skipped: amazon/{method}/s{seed}")

            gc.collect()
            if HAS_GPU:
                torch.cuda.empty_cache()

    del data, col_stats_dict
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()
    log_memory("after phase B cleanup")

    return results


# ===========================================================================
# Main
# ===========================================================================
@logger.catch
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--mini", action="store_true", help="Mini run: 1 seed, 3 epochs"
    )
    parser.add_argument("--phase-a-only", action="store_true")
    parser.add_argument("--phase-b-only", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    TIME_BUDGET_SEC = 50 * 60  # 50 minutes per script invocation

    def remaining_min() -> float:
        return (TIME_BUDGET_SEC - (time.time() - start_time)) / 60

    logger.info("=" * 60)
    logger.info("RAMA CONTROLLED COMPARISON EXPERIMENT")
    logger.info(f"Mini={args.mini}, PhaseA={'skip' if args.phase_b_only else 'run'}, "
                f"PhaseB={'skip' if args.phase_a_only else 'run'}")
    logger.info("=" * 60)

    # Phase 0: Unit tests
    run_unit_tests()

    # Phase A: rel-f1
    results_a: List[Dict] = []
    if not args.phase_b_only:
        t_a = time.time()
        results_a = run_phase_a(mini=args.mini)
        phase_a_time = time.time() - t_a
        logger.info(
            f"Phase A complete: {len(results_a)} runs in {phase_a_time / 60:.1f} min"
        )
        save_incremental(results_a, "phase_a_final")

        # Confirmation checks
        for task_name, _ in TASKS_F1:
            mean_runs = [
                r
                for r in results_a
                if r["task"] == task_name and r["method"] == "standard_mean"
            ]
            rama_runs = [
                r
                for r in results_a
                if r["task"] == task_name and r["method"] == "rama_full"
            ]
            if mean_runs:
                metric = "mae" if mean_runs[0]["task_type"] == "regression" else "average_precision"
                mean_vals = [r["test_metrics"].get(metric, float("nan")) for r in mean_runs]
                rama_vals = [r["test_metrics"].get(metric, float("nan")) for r in rama_runs]
                logger.info(
                    f"Phase A {task_name}: mean_{metric}={np.nanmean(mean_vals):.4f}, "
                    f"rama_{metric}={np.nanmean(rama_vals):.4f}"
                )

    # Check time budget before Phase B
    if not args.phase_a_only and remaining_min() < 5:
        logger.warning(
            f"Only {remaining_min():.1f}min left, skipping Phase B"
        )
        DEVIATIONS.append(
            f"Phase B skipped: only {remaining_min():.1f}min remaining"
        )
        args.phase_a_only = True

    # Phase B: rel-amazon
    results_b: List[Dict] = []
    if not args.phase_a_only:
        t_b = time.time()
        results_b = run_phase_b(mini=args.mini)
        phase_b_time = time.time() - t_b
        logger.info(
            f"Phase B complete: {len(results_b)} runs in {phase_b_time / 60:.1f} min"
        )
        save_incremental(results_b, "phase_b_final")

    # Combine and analyze
    all_results = results_a + results_b
    total_time = time.time() - start_time

    if not all_results:
        logger.error("No results collected! Aborting.")
        return

    logger.info("=" * 60)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("=" * 60)

    analysis = analyze_results(all_results)

    timing = {
        "total_seconds": total_time,
        "total_minutes": total_time / 60,
        "phase_a_runs": len(results_a),
        "phase_b_runs": len(results_b),
    }
    if results_a:
        timing["phase_a_seconds"] = sum(
            r.get("total_time", 0) for r in results_a
        )
    if results_b:
        timing["phase_b_seconds"] = sum(
            r.get("total_time", 0) for r in results_b
        )

    # Build output
    output = build_output(all_results, analysis, timing, mini=args.mini)

    # Save
    out_path = WS / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved final output to {out_path}")

    # Check file size
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output size: {size_mb:.1f}MB")

    if size_mb > 45:
        logger.warning(f"Output too large ({size_mb:.1f}MB), truncating predictions")
        # Remove test_predictions from per_run_results and rebuild
        for ds in output["datasets"]:
            if len(ds["examples"]) > 500:
                ds["examples"] = ds["examples"][:500]
        out_path.write_text(json.dumps(output, indent=2))
        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info(f"Truncated output size: {size_mb:.1f}MB")

    # Summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total runs: {len(all_results)}")
    logger.info(f"Total time: {total_time / 60:.1f} min")

    if "pooled_effects" in analysis:
        pe = analysis["pooled_effects"]
        for comp_name, comp_data in pe.items():
            if comp_data.get("pooled_d") is not None:
                logger.info(
                    f"  {comp_name}: pooled d={comp_data['pooled_d']:.3f}, "
                    f"CI={comp_data['pooled_ci_95']}"
                )

    if DEVIATIONS:
        logger.info(f"Deviations: {len(DEVIATIONS)}")
        for d in DEVIATIONS:
            logger.info(f"  - {d}")

    logger.info("DONE!")


if __name__ == "__main__":
    main()
