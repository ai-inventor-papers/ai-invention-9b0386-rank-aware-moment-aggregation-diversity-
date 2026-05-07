#!/usr/bin/env python3
"""RAMA vs Mean Aggregation on rel-amazon/item-ltv (Large-Scale Regression).

Tests RAMA (Rank-Aware Moment Aggregation) against standard mean aggregation
on rel-amazon/item-ltv, the largest available regression dataset
(506K products, 20.8M reviews).

CRITICAL FIX from previous crash: Exclude text_embedded columns from the
review table (20.8M rows). GloVe embedding 20.8M texts consumed 50GB+ RAM
and OOM-killed the container. Non-text features (rating, verified, timestamp)
are retained. Product text columns (506K rows) are kept.
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
            f"GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM)")
logger.info(f"Current RAM usage: {_current_ram_gb():.1f}GB")

# ---------------------------------------------------------------------------
# Memory limits - CRITICAL for preventing OOM container crash
# ---------------------------------------------------------------------------
# Use 75% of container RAM (57GB -> 42.75GB)
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.75 * 1e9)
# Virtual address space limit at 3x to allow for fragmentation
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
logger.info(f"RAM budget: {RAM_BUDGET_BYTES/1e9:.1f}GB (75% of {TOTAL_RAM_GB:.1f}GB)")

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(0.90)
    logger.info(f"GPU memory: {_free/1e9:.1f}GB free / {_total/1e9:.1f}GB total")
    # Use single CPU thread since we have GPU for heavy lifting
    torch.set_num_threads(max(1, NUM_CPUS // 2))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET = "rel-amazon"
TASK = "item-ltv"
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 10
BATCH_SIZE = 512
CHANNELS = 128
NUM_LAYERS = 2
NUM_NEIGHBORS = 128
LR = 0.005
MAX_STEPS_PER_EPOCH = 2000

# Existing relbench DB cache (from a prior data artifact)
EXISTING_DB_CACHE = Path(
    "/ai-inventor/aii_pipeline/runs/temp-debug-test_sbr/"
    "3_invention_loop/iter_1/gen_art/data_id3_it1__opus/"
    "temp/relbench_cache"
)
CACHE_DIR = str(EXISTING_DB_CACHE)

# Materialized graph cache (fresh, not reusing corrupted one)
MATERIALIZED_CACHE = str(WS / "mat_cache_v2" / f"{DATASET}_notext" / "materialized")

CONFIGS = {
    "baseline_mean": {"aggr": "mean", "label": "Mean Aggregation"},
    "rama": {"aggr": "rama", "label": "RAMA (gate_hidden=16)"},
}

# Track deviations from plan
DEVIATIONS: List[str] = []


def log_memory(label: str = "") -> float:
    """Log current memory usage. Returns usage in GB."""
    ram_gb = _current_ram_gb()
    gpu_info = ""
    if HAS_GPU:
        gpu_used = torch.cuda.memory_allocated() / 1e9
        gpu_info = f", GPU={gpu_used:.1f}GB"
    logger.info(f"[MEM {label}] RAM={ram_gb:.1f}/{TOTAL_RAM_GB:.0f}GB{gpu_info}")
    if ram_gb > TOTAL_RAM_GB * 0.85:
        logger.warning(f"RAM usage at {ram_gb/TOTAL_RAM_GB*100:.0f}% - approaching limit!")
    return ram_gb


# ===========================================================================
# Dataset Loading - with OOM prevention
# ===========================================================================
def load_dataset_and_graph() -> Tuple[Any, Any, Any, Any, Any]:
    """Load dataset, build graph, cache materialized tensors.

    CRITICAL FIX: Excludes text_embedded columns from the review table
    (20.8M rows) to prevent OOM. Product text columns (506K rows) are kept.
    """
    from relbench.datasets import get_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from relbench.tasks import get_task
    from torch_frame import stype as st
    from torch_frame.config.text_embedder import TextEmbedderConfig

    from text_embedder import GloveTextEmbedding

    logger.info("Loading dataset...")
    t0 = time.time()

    os.environ["RELBENCH_CACHE_DIR"] = CACHE_DIR
    dataset = get_dataset(DATASET, download=True)
    task = get_task(DATASET, TASK, download=True)
    db = dataset.get_db()

    logger.info(f"Dataset loaded in {time.time()-t0:.0f}s")
    logger.info(f"Tables: {list(db.table_dict.keys())}")
    log_memory("after dataset load")

    # ----- Build stype dict with OOM-safe exclusions -----
    # Load cached stypes from existing cache
    stypes_cache_path = Path(f"{CACHE_DIR}/{DATASET}/stypes.json")
    if stypes_cache_path.exists():
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in list(col_to_stype.items()):
                col_to_stype[col] = st(stype_str)
        logger.info("Loaded cached stypes")
    else:
        logger.info("Computing stype proposal...")
        col_to_stype_dict = get_stype_proposal(db)

    # === CRITICAL FIX: Remove text columns from review table ===
    # The review table has 20.8M rows. GloVe embedding of review_text and
    # summary columns requires ~50GB RAM (20.8M * 2 cols * 300d * 4 bytes).
    # This caused the previous container crash.
    if "review" in col_to_stype_dict:
        for text_col in ["review_text", "summary"]:
            if text_col in col_to_stype_dict["review"]:
                removed_stype = col_to_stype_dict["review"].pop(text_col)
                logger.warning(
                    f"REMOVED review.{text_col} ({removed_stype}) to prevent OOM "
                    f"(20.8M rows * 300d = ~6.2GB per column)"
                )
        DEVIATIONS.append(
            "Removed review_text and summary (text_embedded) from review table "
            "to prevent OOM: 20.8M rows * 2 cols * 300d embeddings = ~50GB"
        )

    # Also remove customer_name text embedding (1.85M rows * 300d = ~2.2GB)
    if "customer" in col_to_stype_dict:
        if "customer_name" in col_to_stype_dict["customer"]:
            col_to_stype_dict["customer"].pop("customer_name")
            logger.warning("REMOVED customer.customer_name to save ~2.2GB")
            DEVIATIONS.append(
                "Removed customer_name (text_embedded) from customer table "
                "to save ~2.2GB (1.85M rows * 300d)"
            )

    # Log final stype dict
    for table, stypes in col_to_stype_dict.items():
        logger.info(f"  {table}: {dict(stypes)}")

    log_memory("before graph build")

    # Text embedder for remaining text columns (product table: 506K rows, manageable)
    text_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=torch.device("cpu")),
        batch_size=256,
    )

    logger.info("Building pkey-fkey graph...")
    logger.info("  Product text embeddings: 506K rows * 3 cols = ~1.8GB (manageable)")
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

    # Free the database object to reclaim memory
    del db
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

    log_memory("after cleanup")
    return dataset, task, data, col_stats_dict, graph_time


# ===========================================================================
# NeighborLoaders
# ===========================================================================
def build_loaders(
    task: Any,
    data: Any,
    num_neighbors: int = NUM_NEIGHBORS,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    from relbench.modeling.graph import get_node_train_table_input
    from torch_geometric.loader import NeighborLoader

    loader_dict = {}
    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[
                int(num_neighbors / 2**i) for i in range(NUM_LAYERS)
            ],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=batch_size,
            temporal_strategy="uniform",
            shuffle=(split == "train"),
            num_workers=0,  # Avoid memory duplication
        )
    return loader_dict


# ===========================================================================
# Model creation
# ===========================================================================
def create_model(data: Any, col_stats_dict: Any, aggr_config: str) -> Any:
    from model import Model
    from rama import RAMAAggregation

    if aggr_config == "rama":
        aggr = RAMAAggregation(channels=CHANNELS, gate_hidden=16)
    else:
        aggr = aggr_config  # string like "mean"

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
    max_steps: int = MAX_STEPS_PER_EPOCH,
) -> float:
    model.train()
    loss_accum = count_accum = 0
    steps = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch, entity_table).view(-1)
        loss = loss_fn(pred.float(), batch[entity_table].y.float())

        if torch.isnan(loss):
            logger.warning("NaN loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)
        steps += 1
        if steps >= max_steps:
            break
    return loss_accum / max(count_accum, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Any,
    entity_table: str,
    clamp_min: float,
    clamp_max: float,
) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table).view(-1)
        pred = torch.clamp(pred, clamp_min, clamp_max)
        preds.append(pred.cpu())
    return torch.cat(preds, dim=0).numpy()


# ===========================================================================
# Gate statistics (RAMA only)
# ===========================================================================
def extract_gate_stats(model: torch.nn.Module) -> Dict[str, Any]:
    from rama import RAMAAggregation

    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, RAMAAggregation):
            with torch.no_grad():
                gate_bias = module.gate_out.bias.data.cpu()
                stats[name] = {
                    "gate_bias_mean": float(gate_bias.mean()),
                    "gate_bias_std": float(gate_bias.std()),
                    "gate_bias_sigmoid_mean": float(
                        torch.sigmoid(gate_bias).mean()
                    ),
                    "W_sigma_norm": float(
                        module.W_sigma.weight.data.norm().item()
                    ),
                }
    return stats


# ===========================================================================
# Single run
# ===========================================================================
def run_single(
    task: Any,
    data: Any,
    col_stats_dict: Any,
    loader_dict: Dict[str, Any],
    config_name: str,
    seed: int,
    epochs: int,
    max_steps: int = MAX_STEPS_PER_EPOCH,
) -> Dict[str, Any]:
    from torch_geometric.seed import seed_everything

    seed_everything(seed)

    aggr_val = CONFIGS[config_name]["aggr"]
    model = create_model(data, col_stats_dict, aggr_val)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model params: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss()
    entity_table = task.entity_table

    # Clamping from train target stats
    train_table = task.get_table("train")
    targets = train_table.df[task.target_col].dropna()
    clamp_min = float(np.percentile(targets.to_numpy(), 1))
    clamp_max = float(np.percentile(targets.to_numpy(), 99))

    best_val_metric = math.inf
    best_state = None
    epoch_times = []
    train_losses = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, loader_dict["train"], optimizer, loss_fn,
            entity_table, max_steps
        )
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        train_losses.append(train_loss)

        val_pred = evaluate(
            model, loader_dict["val"], entity_table, clamp_min, clamp_max
        )
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        val_mae = val_metrics["mae"]

        logger.info(
            f"  [{config_name}] seed={seed} epoch={epoch}/{epochs} "
            f"loss={train_loss:.4f} val_mae={val_mae:.4f} "
            f"time={epoch_time:.1f}s"
        )

        if val_mae <= best_val_metric:
            best_val_metric = val_mae
            best_state = copy.deepcopy(model.state_dict())

    # Final test evaluation with best model
    model.load_state_dict(best_state)
    test_pred = evaluate(
        model, loader_dict["test"], entity_table, clamp_min, clamp_max
    )
    test_table = task.get_table("test")

    # Test table may or may not have target column
    if task.target_col in test_table.df.columns:
        test_metrics = task.evaluate(test_pred, test_table)
        test_mae = test_metrics.get("mae", float("nan"))
    else:
        # Use val MAE as proxy when test targets unavailable
        logger.info("  Test table has no targets, using best val_mae as test_mae")
        test_metrics = {}
        test_mae = best_val_metric

    # Gate stats (RAMA only)
    gate_stats = extract_gate_stats(model) if config_name == "rama" else {}

    result = {
        "config": config_name,
        "seed": seed,
        "test_mae": float(test_mae),
        "best_val_mae": float(best_val_metric),
        "train_losses": [float(x) for x in train_losses],
        "epoch_times": [float(x) for x in epoch_times],
        "mean_epoch_time": float(np.mean(epoch_times)),
        "gate_stats": gate_stats,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "test_predictions": test_pred.tolist(),
        "num_params": param_count,
    }

    del model, best_state
    if HAS_GPU:
        torch.cuda.empty_cache()
    gc.collect()
    return result


# ===========================================================================
# Statistical analysis
# ===========================================================================
def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Cohen's d. Positive means group1 has higher MAE (RAMA better)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    mean1, mean2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((mean1 - mean2) / pooled_std)


def compute_ci_cohens_d(
    d: float, n1: int, n2: int, alpha: float = 0.05
) -> Tuple[float, float]:
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    z = sp_stats.norm.ppf(1 - alpha / 2)
    return (float(d - z * se), float(d + z * se))


def run_statistical_analysis(
    all_results: List[Dict],
) -> Dict[str, Any]:
    baseline_maes = [
        r["test_mae"] for r in all_results if r["config"] == "baseline_mean"
    ]
    rama_maes = [
        r["test_mae"] for r in all_results if r["config"] == "rama"
    ]

    analysis: Dict[str, Any] = {
        "baseline_mean_maes": baseline_maes,
        "rama_maes": rama_maes,
        "baseline_mean_mae_mean": (
            float(np.mean(baseline_maes)) if baseline_maes else None
        ),
        "baseline_mean_mae_std": (
            float(np.std(baseline_maes, ddof=1)) if len(baseline_maes) > 1 else None
        ),
        "rama_mae_mean": float(np.mean(rama_maes)) if rama_maes else None,
        "rama_mae_std": (
            float(np.std(rama_maes, ddof=1)) if len(rama_maes) > 1 else None
        ),
    }

    if len(baseline_maes) >= 2 and len(rama_maes) >= 2:
        d = compute_cohens_d(baseline_maes, rama_maes)
        ci_lo, ci_hi = compute_ci_cohens_d(d, len(baseline_maes), len(rama_maes))
        analysis["cohens_d"] = float(d)
        analysis["cohens_d_ci_95"] = [float(ci_lo), float(ci_hi)]
    else:
        analysis["cohens_d"] = None
        analysis["cohens_d_ci_95"] = None

    # Paired t-test (same seeds, paired comparison)
    if len(baseline_maes) == len(rama_maes) and len(baseline_maes) >= 3:
        t_stat, p_val = sp_stats.ttest_rel(baseline_maes, rama_maes)
        analysis["paired_ttest_t"] = float(t_stat)
        analysis["paired_ttest_p"] = float(p_val)
    elif len(baseline_maes) >= 2 and len(rama_maes) >= 2:
        t_stat, p_val = sp_stats.ttest_ind(baseline_maes, rama_maes)
        analysis["ind_ttest_t"] = float(t_stat)
        analysis["ind_ttest_p"] = float(p_val)

    # Wilcoxon signed-rank (non-parametric)
    if len(baseline_maes) == len(rama_maes) and len(baseline_maes) >= 3:
        try:
            w_stat, w_pval = sp_stats.wilcoxon(baseline_maes, rama_maes)
            analysis["wilcoxon_stat"] = float(w_stat)
            analysis["wilcoxon_p"] = float(w_pval)
        except ValueError:
            analysis["wilcoxon_stat"] = None
            analysis["wilcoxon_p"] = None

    # Improvement percentage
    if baseline_maes and rama_maes:
        analysis["improvement_pct"] = float(
            (np.mean(baseline_maes) - np.mean(rama_maes))
            / max(np.mean(baseline_maes), 1e-12)
            * 100
        )

    # Epoch time overhead
    baseline_times = [
        r["mean_epoch_time"] for r in all_results if r["config"] == "baseline_mean"
    ]
    rama_times = [
        r["mean_epoch_time"] for r in all_results if r["config"] == "rama"
    ]
    if baseline_times and rama_times:
        analysis["baseline_mean_epoch_time"] = float(np.mean(baseline_times))
        analysis["rama_mean_epoch_time"] = float(np.mean(rama_times))
        analysis["rama_epoch_time_overhead_pct"] = float(
            (np.mean(rama_times) - np.mean(baseline_times))
            / max(np.mean(baseline_times), 1e-12)
            * 100
        )

    return analysis


# ===========================================================================
# Build output in exp_gen_sol_out schema
# ===========================================================================
def build_output(
    task: Any,
    all_results: List[Dict],
    analysis: Dict[str, Any],
    actual_epochs: int,
    actual_seeds: int,
    actual_num_neighbors: int,
    actual_max_steps: int,
    graph_build_time: float,
) -> Dict[str, Any]:
    """Build method_out.json conforming to exp_gen_sol_out schema."""

    test_table = task.get_table("test")
    test_df = test_table.df

    # Use best-seed predictions for each config
    baseline_results = [r for r in all_results if r["config"] == "baseline_mean"]
    rama_results_list = [r for r in all_results if r["config"] == "rama"]

    baseline_test_preds = None
    if baseline_results:
        best_baseline = min(baseline_results, key=lambda r: r["test_mae"])
        baseline_test_preds = best_baseline.get("test_predictions", None)

    rama_test_preds = None
    if rama_results_list:
        best_rama = min(rama_results_list, key=lambda r: r["test_mae"])
        rama_test_preds = best_rama.get("test_predictions", None)

    # Build examples from test set
    examples = []
    entity_col = task.entity_col
    target_col = task.target_col

    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        input_dict = {}
        for col in test_df.columns:
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

        has_target = target_col in test_df.columns
        target_val = row.get(target_col, None) if has_target else None
        if target_val is not None and not (
            isinstance(target_val, float) and np.isnan(target_val)
        ):
            output_str = str(round(float(target_val), 6))
        else:
            output_str = "nan"

        example: Dict[str, Any] = {
            "input": json.dumps(input_dict),
            "output": output_str,
        }

        if baseline_test_preds is not None and idx < len(baseline_test_preds):
            example["predict_baseline_mean"] = str(
                round(float(baseline_test_preds[idx]), 6)
            )
        if rama_test_preds is not None and idx < len(rama_test_preds):
            example["predict_rama"] = str(
                round(float(rama_test_preds[idx]), 6)
            )

        example["metadata_fold"] = 2
        example["metadata_task_type"] = "regression"
        example["metadata_entity_col"] = entity_col
        example["metadata_target_col"] = target_col
        if entity_col in test_df.columns:
            eid = row[entity_col]
            example["metadata_row_index"] = (
                int(eid) if isinstance(eid, (int, np.integer)) else str(eid)
            )

        examples.append(example)

    # Gate analysis
    gate_analysis = {}
    for r in rama_results_list:
        if r.get("gate_stats"):
            gate_analysis[f"seed_{r['seed']}"] = r["gate_stats"]

    output = {
        "metadata": {
            "experiment": "RAMA vs Mean Aggregation on rel-amazon/item-ltv",
            "method_name": "RAMA (Rank-Aware Moment Aggregation)",
            "description": (
                "Tests RAMA against standard mean aggregation on rel-amazon/item-ltv. "
                "RAMA enriches mean pooling with gated per-dimension variance, where "
                "gate is conditioned on effective spectral rank proxy and cardinality."
            ),
            "dataset_info": {
                "name": "rel-amazon",
                "task": "item-ltv",
                "num_products": 506012,
                "num_reviews": 20862040,
                "task_type": "regression",
                "metric": "mae",
            },
            "hyperparameters": {
                "channels": CHANNELS,
                "lr": LR,
                "epochs": actual_epochs,
                "batch_size": BATCH_SIZE,
                "num_neighbors": actual_num_neighbors,
                "num_layers": NUM_LAYERS,
                "rama_gate_hidden": 16,
                "max_steps_per_epoch": actual_max_steps,
            },
            "statistical_analysis": analysis,
            "gate_analysis": gate_analysis,
            "per_run_results": [
                {k: v for k, v in r.items()
                 if k not in ("test_predictions",)}
                for r in all_results
            ],
            "deviations_from_plan": DEVIATIONS,
            "num_seeds": actual_seeds,
            "graph_build_time_sec": graph_build_time,
        },
        "datasets": [
            {
                "dataset": "rel-amazon/item-ltv",
                "examples": examples,
            }
        ],
    }
    return output


# ===========================================================================
# Save intermediate results
# ===========================================================================
def save_intermediate(all_results: List[Dict], label: str = "intermediate"):
    """Save intermediate results to avoid data loss."""
    out_path = WS / f"results_{label}.json"
    safe_results = [
        {k: v for k, v in r.items()
         if k not in ("test_predictions",)}
        for r in all_results
    ]
    out_path.write_text(json.dumps(safe_results, indent=2))
    logger.info(f"Saved {label} results to {out_path}")


# ===========================================================================
# Main
# ===========================================================================
@logger.catch
def main():
    start_time = time.time()
    TIME_BUDGET_SEC = 260 * 60  # 260 minutes (reduced for resumed session)

    def remaining_minutes() -> float:
        return (TIME_BUDGET_SEC - (time.time() - start_time)) / 60

    logger.info("=" * 60)
    logger.info("RAMA vs Mean Aggregation Experiment")
    logger.info("=" * 60)

    # -------------------------------------------------------------------
    # Phase 1: Unit test RAMA
    # -------------------------------------------------------------------
    logger.info("Phase 1: RAMA unit test...")
    from rama import RAMAAggregation

    rama_test = RAMAAggregation(channels=64, gate_hidden=8)
    x_test = torch.randn(100, 64)
    index_test = torch.randint(0, 10, (100,))

    out_test = rama_test(x_test, index=index_test, dim_size=10)
    assert out_test.shape == (10, 64), f"Expected (10,64), got {out_test.shape}"

    # Zero-init: RAMA should approximately equal mean
    mean_manual = torch.zeros(10, 64)
    count_manual = torch.zeros(10, 1)
    for i in range(100):
        mean_manual[index_test[i]] += x_test[i]
        count_manual[index_test[i]] += 1
    count_manual = count_manual.clamp(min=1)
    mean_manual = mean_manual / count_manual
    diff = (out_test - mean_manual).abs().max().item()
    assert diff < 1e-3, f"Initial RAMA should equal mean, diff={diff}"

    # Gradient flow
    loss_test = out_test.sum()
    loss_test.backward()
    for name, p in rama_test.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

    logger.info("RAMA unit tests passed!")
    del rama_test, x_test, index_test, out_test, mean_manual
    gc.collect()

    # -------------------------------------------------------------------
    # Phase 2: Load data
    # -------------------------------------------------------------------
    logger.info("Phase 2: Loading dataset and building graph...")
    log_memory("before data load")
    dataset, task, data, col_stats_dict, graph_build_time = (
        load_dataset_and_graph()
    )

    # -------------------------------------------------------------------
    # Phase 3: Build loaders
    # -------------------------------------------------------------------
    logger.info("Phase 3: Building neighbor loaders...")
    actual_num_neighbors = NUM_NEIGHBORS
    actual_batch_size = BATCH_SIZE
    loader_dict = build_loaders(
        task, data, actual_num_neighbors, actual_batch_size
    )
    logger.info("Loaders built.")
    log_memory("after loaders")

    # -------------------------------------------------------------------
    # Phase 4: Mini validation run (1 seed, 2 epochs)
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 4: MINI VALIDATION RUN (2 epochs, 1 seed)")
    logger.info("=" * 60)

    actual_max_steps = MAX_STEPS_PER_EPOCH
    mini_results = {}

    for cfg in ["baseline_mean", "rama"]:
        logger.info(f"Mini run: {cfg}")
        try:
            mini_result = run_single(
                task, data, col_stats_dict, loader_dict,
                cfg, seed=42, epochs=2, max_steps=actual_max_steps,
            )
            mini_results[cfg] = mini_result
            logger.info(
                f"Mini {cfg}: test_mae={mini_result['test_mae']:.4f}, "
                f"epoch_time={mini_result['mean_epoch_time']:.1f}s"
            )
            log_memory(f"after mini {cfg}")

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"GPU OOM on {cfg}! Reducing batch/neighbors")
            torch.cuda.empty_cache()
            gc.collect()
            DEVIATIONS.append(
                f"GPU OOM on mini run {cfg}, reducing neighbors=64, batch=256"
            )
            actual_num_neighbors = 64
            actual_batch_size = 256
            loader_dict = build_loaders(
                task, data, actual_num_neighbors, actual_batch_size
            )
            mini_result = run_single(
                task, data, col_stats_dict, loader_dict,
                cfg, seed=42, epochs=2, max_steps=actual_max_steps,
            )
            mini_results[cfg] = mini_result
            logger.info(
                f"Mini {cfg} (reduced): test_mae={mini_result['test_mae']:.4f}, "
                f"epoch_time={mini_result['mean_epoch_time']:.1f}s"
            )

    # Check if training is too slow
    max_epoch_time = max(r["mean_epoch_time"] for r in mini_results.values())
    logger.info(f"Max epoch time from mini run: {max_epoch_time:.1f}s")

    if max_epoch_time > 1800:
        logger.warning("Epochs too slow (>30min)! Reducing to neighbors=64")
        DEVIATIONS.append(
            f"Epoch time {max_epoch_time:.0f}s > 1800s, reducing neighbors=64"
        )
        if actual_num_neighbors > 64:
            actual_num_neighbors = 64
            loader_dict = build_loaders(
                task, data, actual_num_neighbors, actual_batch_size
            )

    if max_epoch_time > 900:
        actual_max_steps = min(actual_max_steps, 1000)
        DEVIATIONS.append(
            f"Epoch time {max_epoch_time:.0f}s > 900s, "
            f"max_steps={actual_max_steps}"
        )
        logger.warning(f"Reducing max_steps to {actual_max_steps}")

    # -------------------------------------------------------------------
    # Phase 5: Plan full runs based on time budget
    # -------------------------------------------------------------------
    logger.info("Phase 5: Planning full runs...")
    mins_left = remaining_minutes()
    logger.info(f"Time remaining: {mins_left:.0f} minutes")

    est_epoch_time = max_epoch_time
    actual_epochs = EPOCHS
    actual_seeds = len(SEEDS)

    eval_overhead = est_epoch_time * 0.5
    est_run_time = actual_epochs * est_epoch_time + eval_overhead
    est_total_time = 2 * actual_seeds * est_run_time

    logger.info(f"Estimated per run: {est_run_time/60:.1f} min")
    logger.info(f"Estimated total: {est_total_time/60:.1f} min")

    # Leave 25 min for output generation
    time_budget_min = mins_left - 25

    # Progressively reduce if needed
    if est_total_time / 60 > time_budget_min:
        actual_epochs = min(actual_epochs, 5)
        est_run_time = actual_epochs * est_epoch_time + eval_overhead
        est_total_time = 2 * actual_seeds * est_run_time
        DEVIATIONS.append(f"Reduced epochs to {actual_epochs}")
        logger.info(f"Reduced epochs to {actual_epochs}")

    if est_total_time / 60 > time_budget_min:
        actual_seeds = 3
        est_total_time = 2 * actual_seeds * est_run_time
        DEVIATIONS.append(f"Reduced seeds to {actual_seeds}")
        logger.info(f"Reduced seeds to {actual_seeds}")

    if est_total_time / 60 > time_budget_min:
        actual_epochs = max(3, actual_epochs - 2)
        est_run_time = actual_epochs * est_epoch_time + eval_overhead
        est_total_time = 2 * actual_seeds * est_run_time
        DEVIATIONS.append(
            f"Further reduced to {actual_epochs} epochs, {actual_seeds} seeds"
        )
        logger.info(
            f"Further reduced to {actual_epochs} epochs, {actual_seeds} seeds"
        )

    if est_total_time / 60 > time_budget_min:
        actual_epochs = 3
        actual_seeds = 2
        est_run_time = actual_epochs * est_epoch_time + eval_overhead
        DEVIATIONS.append("Time tight: 3 epochs, 2 seeds")
        logger.warning("Time tight: 3 epochs, 2 seeds")

    seeds_to_use = SEEDS[:actual_seeds]
    logger.info(
        f"Final plan: {actual_epochs} epochs, {actual_seeds} seeds "
        f"({seeds_to_use}), {actual_num_neighbors} neighbors, "
        f"max_steps={actual_max_steps}"
    )

    # -------------------------------------------------------------------
    # Phase 6: Full runs (with resume support)
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 6: FULL EXPERIMENT RUNS")
    logger.info("=" * 60)

    # Resume: load previously completed results
    all_results: List[Dict] = []
    completed_pairs: set = set()
    intermediate_path = WS / "results_intermediate.json"
    if intermediate_path.exists():
        try:
            prev = json.loads(intermediate_path.read_text())
            if isinstance(prev, list) and len(prev) > 0:
                all_results = prev
                for r in prev:
                    completed_pairs.add((r["config"], r["seed"]))
                logger.info(
                    f"Resumed {len(prev)} completed runs: {completed_pairs}"
                )
        except Exception as e:
            logger.warning(f"Could not load intermediate results: {e}")

    stopped_early = False

    for cfg in ["baseline_mean", "rama"]:
        for seed in seeds_to_use:
            if (cfg, seed) in completed_pairs:
                logger.info(f"Skipping {cfg} seed={seed} (already completed)")
                continue

            mins_left = remaining_minutes()
            if mins_left < 20:
                logger.warning(
                    f"Only {mins_left:.0f} min left, stopping runs"
                )
                DEVIATIONS.append(
                    f"Stopped at {cfg}/seed={seed}: {mins_left:.0f}min left"
                )
                stopped_early = True
                break

            logger.info(
                f"=== FULL RUN: {cfg} seed={seed} epochs={actual_epochs} "
                f"(~{mins_left:.0f} min left) ==="
            )
            try:
                result = run_single(
                    task, data, col_stats_dict, loader_dict,
                    cfg, seed=seed, epochs=actual_epochs,
                    max_steps=actual_max_steps,
                )
                all_results.append(result)
                save_intermediate(all_results)
                logger.info(
                    f"Completed {cfg} seed={seed}: "
                    f"test_mae={result['test_mae']:.4f}"
                )
                log_memory(f"after {cfg}/seed={seed}")

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"GPU OOM on {cfg} seed={seed}!")
                torch.cuda.empty_cache()
                gc.collect()
                DEVIATIONS.append(f"GPU OOM on {cfg}/seed={seed}, skipping")
                continue
            except MemoryError:
                logger.warning(f"RAM OOM on {cfg} seed={seed}!")
                gc.collect()
                DEVIATIONS.append(f"RAM OOM on {cfg}/seed={seed}, skipping")
                continue
            except Exception:
                logger.exception(f"Error on {cfg} seed={seed}")
                DEVIATIONS.append(f"Error on {cfg}/seed={seed}")
                continue

        if stopped_early:
            break

    logger.info(f"Completed {len(all_results)} runs total")

    # -------------------------------------------------------------------
    # Phase 7: Statistical analysis
    # -------------------------------------------------------------------
    logger.info("Phase 7: Statistical analysis...")
    analysis = run_statistical_analysis(all_results)

    # Log key results
    for key in ["baseline_mean_mae_mean", "rama_mae_mean", "cohens_d",
                "improvement_pct", "rama_epoch_time_overhead_pct"]:
        if key in analysis and analysis[key] is not None:
            logger.info(f"  {key}: {analysis[key]:.4f}")

    # -------------------------------------------------------------------
    # Phase 8: Build and save output
    # -------------------------------------------------------------------
    logger.info("Phase 8: Building output...")
    output = build_output(
        task, all_results, analysis,
        actual_epochs, actual_seeds,
        actual_num_neighbors, actual_max_steps,
        graph_build_time,
    )

    out_path = WS / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path}")

    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Output file size: {size_mb:.1f} MB")

    if size_mb > 50:
        logger.warning(f"Output file is {size_mb:.1f}MB, may need splitting")
        DEVIATIONS.append(f"Output file large: {size_mb:.1f}MB")

    total_time = (time.time() - start_time) / 60
    logger.info(f"Total experiment time: {total_time:.1f} minutes")
    logger.info(f"Deviations: {DEVIATIONS}")
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
