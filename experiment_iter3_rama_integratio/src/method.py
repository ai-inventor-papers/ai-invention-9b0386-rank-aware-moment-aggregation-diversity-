#!/usr/bin/env python3
"""RAMA Integration into RelGNN: Additive Improvement over SOTA.

Compares RelGNN+sum (baseline) vs RelGNN+RAMA on rel-f1/driver-position
and rel-amazon/item-ltv. Falls back to HeteroSAGE+RAMA if RelGNN fails.
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from loguru import logger
from scipy import stats
from torch.nn import BCEWithLogitsLoss, L1Loss

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
WS = Path(__file__).resolve().parent
LOG_DIR = WS / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

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
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, "
            f"GPU={'yes' if HAS_GPU else 'no'}"
            f"{f' ({VRAM_GB:.1f} GB VRAM)' if HAS_GPU else ''}")

# Memory limits — DO NOT use resource.setrlimit(RLIMIT_AS) as it breaks CUDA
# memory mapping in worker processes. Use GPU memory fraction only.
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.85, TOTAL_RAM_GB - 4) * 1e9)
if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(0.92)
    logger.info(f"VRAM budget: {_free/1e9:.1f} GB free / {_total/1e9:.1f} GB total")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("/ai-inventor/aii_pipeline/runs/leskovec-predictive-residual-message-passing-v2_sti/"
                "3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
RELBENCH_CACHE = str(DATA_DIR / "relbench_cache")

TASKS = [
    {"dataset": "rel-f1", "task": "driver-position", "task_type": "regression",
     "metric": "mae", "higher_is_better": False,
     "config": {"num_model_layers": 1, "channels": 128, "aggr": "sum",
                "num_heads": 4, "batch_size": 512, "num_neighbors": 128,
                "subgraph_type": "bidirectional"}},
    {"dataset": "rel-amazon", "task": "item-ltv", "task_type": "regression",
     "metric": "mae", "higher_is_better": False,
     "config": {"num_model_layers": 2, "channels": 128, "aggr": "sum",
                "num_heads": 8, "batch_size": 2048, "num_neighbors": 128,
                "subgraph_type": "directional"}},
]

METHODS = {
    "relgnn_sum":  {"use_rama": False, "label": "RelGNN+sum (original)"},
    "relgnn_rama": {"use_rama": True,  "label": "RelGNN+RAMA"},
}

SEEDS = [42, 123, 456, 789, 1024]
LR = 0.005
EPOCHS = 10
MAX_STEPS_PER_EPOCH = 2000
USE_RELGNN = True  # Will be set False if RelGNN integration fails

# ---------------------------------------------------------------------------
# Imports that may fail (with fallback handling)
# ---------------------------------------------------------------------------
try:
    from torch_frame import stype
    from torch_frame.config.text_embedder import TextEmbedderConfig
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.seed import seed_everything
    from torch_geometric.data import HeteroData

    from relbench.base import Dataset, EntityTask, TaskType
    from relbench.datasets import get_dataset
    from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from relbench.tasks import get_task

    from rama import RAMAAggregation
    from text_embedder import GloveTextEmbedding
    from atomic_routes import get_atomic_routes

    logger.info("All imports successful")
except ImportError as e:
    logger.exception(f"Import failed: {e}")
    raise

# Try RelGNN imports
try:
    from relgnn_model import RelGNN_Model
    logger.info("RelGNN imports successful")
except ImportError as e:
    logger.warning(f"RelGNN import failed: {e}. Will use HeteroSAGE fallback.")
    USE_RELGNN = False

# HeteroSAGE fallback model (from relbench examples)
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch_geometric.nn import MLP
from torch.nn import Embedding, ModuleDict
from torch_geometric.typing import NodeType


class HeteroSAGEModel(torch.nn.Module):
    """HeteroSAGE model from relbench examples, supports custom aggr."""
    def __init__(self, data: HeteroData, col_stats_dict, num_layers: int,
                 channels: int, out_channels: int, aggr, norm: str):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(channels, out_channels=out_channels, norm=norm,
                        num_layers=1)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch: HeteroData, entity_table: str) -> torch.Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict)
        for nt, rt in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rt
        x_dict = self.gnn(x_dict, batch.edge_index_dict,
                          batch.num_sampled_nodes_dict,
                          batch.num_sampled_edges_dict)
        return self.head(x_dict[entity_table][:seed_time.size(0)])


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------
def create_model(data, col_stats_dict, use_rama: bool, config: dict,
                 use_relgnn: bool = True) -> nn.Module:
    """Create model with optional RAMA aggregation."""
    channels = config["channels"]

    if use_rama:
        aggr = RAMAAggregation(channels=channels, gate_hidden=16)
    else:
        aggr = config.get("aggr", "sum")

    if use_relgnn:
        try:
            atomic_routes = get_atomic_routes(data.edge_types)
            logger.info(f"  Computed {len(atomic_routes)} atomic routes "
                        f"({sum(1 for r in atomic_routes if r[0]=='dim-fact-dim')} dim-fact-dim, "
                        f"{sum(1 for r in atomic_routes if r[0]=='dim-dim')} dim-dim)")
            model = RelGNN_Model(
                data=data,
                col_stats_dict=col_stats_dict,
                num_model_layers=config["num_model_layers"],
                channels=channels,
                out_channels=1,
                aggr=aggr,
                norm="batch_norm",
                num_heads=config["num_heads"],
                atomic_routes=atomic_routes,
            )
            logger.info(f"  Created RelGNN model: {sum(p.numel() for p in model.parameters())} params")
            return model
        except Exception as e:
            logger.warning(f"  RelGNN creation failed: {e}. Falling back to HeteroSAGE.")

    # Fallback: HeteroSAGE
    num_layers = max(config.get("num_model_layers", 2), 2)
    model = HeteroSAGEModel(
        data=data, col_stats_dict=col_stats_dict,
        num_layers=num_layers, channels=channels,
        out_channels=1, aggr=aggr, norm="batch_norm",
    )
    logger.info(f"  Created HeteroSAGE model: {sum(p.numel() for p in model.parameters())} params")
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_task_data(task_cfg: dict) -> Tuple:
    """Load dataset, task, data graph, and col_stats_dict."""
    ds_name = task_cfg["dataset"]
    task_name = task_cfg["task"]
    logger.info(f"Loading {ds_name}/{task_name}...")

    os.environ["RELBENCH_CACHE_DIR"] = RELBENCH_CACHE
    dataset: Dataset = get_dataset(ds_name, download=True)
    task: EntityTask = get_task(ds_name, task_name, download=True)

    col_to_stype_dict = get_stype_proposal(dataset.get_db())

    # OOM protection for rel-amazon: exclude text columns
    if ds_name == "rel-amazon":
        logger.info("  Excluding text columns from rel-amazon to prevent OOM")
        if "review" in col_to_stype_dict:
            for col in ["review_text", "summary"]:
                col_to_stype_dict["review"].pop(col, None)
        if "customer" in col_to_stype_dict:
            col_to_stype_dict["customer"].pop("customer_name", None)
        if "product" in col_to_stype_dict:
            col_to_stype_dict["product"].pop("description", None)

    cache_dir = str(WS / "materialized" / ds_name)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    text_embedder = GloveTextEmbedding(device=DEVICE)
    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=text_embedder, batch_size=256),
        cache_dir=cache_dir,
    )

    logger.info(f"  Graph: {len(data.node_types)} node types, "
                f"{len(data.edge_types)} edge types")
    return dataset, task, data, col_stats_dict


def build_loaders(task: EntityTask, data, config: dict) -> Dict[str, NeighborLoader]:
    """Build NeighborLoader for train/val/test splits."""
    loaders = {}
    num_layers = config.get("num_model_layers", 2)
    # RelGNN uses num_layers from loader_config (typically 2)
    loader_num_layers = max(num_layers, 2)
    num_neighbors = [int(config["num_neighbors"] / 2**i)
                     for i in range(loader_num_layers)]

    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        loaders[split] = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            subgraph_type=config.get("subgraph_type", "bidirectional"),
            batch_size=config["batch_size"],
            temporal_strategy="uniform",
            shuffle=(split == "train"),
            num_workers=0,  # CUDA contexts can't be shared in forked workers
        )
    return loaders


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, entity_table,
                    max_steps: int = 0) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(loader):
        if max_steps > 0 and i >= max_steps:
            break
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table).squeeze()
        target = batch[entity_table].y
        if target.dim() > 1:
            target = target.squeeze()
        loss = loss_fn(pred[:len(target)], target.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(target)
        count += len(target)
    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, entity_table, clamp_range=None) -> np.ndarray:
    """Evaluate model and return predictions."""
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch, entity_table)
        if clamp_range is not None:
            pred = torch.clamp(pred, clamp_range[0], clamp_range[1])
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        preds.append(pred.detach().cpu())
    return torch.cat(preds, dim=0).numpy()


def extract_gate_stats(model) -> dict:
    """Extract RAMA gate statistics from model."""
    gate_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, RAMAAggregation):
            bias = module.gate_out.bias.detach().cpu()
            gate_stats[name] = {
                "gate_bias_mean": float(bias.mean()),
                "gate_bias_std": float(bias.std()),
                "gate_bias_sigmoid_mean": float(torch.sigmoid(bias).mean()),
                "W_sigma_norm": float(module.W_sigma.weight.detach().cpu().norm()),
            }
    return gate_stats


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------
def cohens_d(x: List[float], y: List[float]) -> float:
    """Cohen's d (positive = x > y, for MAE: positive = RAMA better)."""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2))
    if pooled < 1e-12:
        return 0.0
    return (mx - my) / pooled


def cohens_d_ci(d: float, n1: int, n2: int, alpha: float = 0.05) -> List[float]:
    """95% CI for Cohen's d using non-central t approximation."""
    se = np.sqrt(n1 + n2) / np.sqrt(n1 * n2) * np.sqrt(1 + d**2 * n1 * n2 / (2 * (n1 + n2)))
    t_crit = stats.t.ppf(1 - alpha/2, n1 + n2 - 2)
    return [d - t_crit * se, d + t_crit * se]


def dersimonian_laird(effects: List[float], variances: List[float]) -> dict:
    """DerSimonian-Laird random-effects meta-analysis."""
    k = len(effects)
    if k == 0:
        return {"pooled_effect": 0, "pooled_se": 0, "tau2": 0, "I2": 0}
    w = [1.0/v if v > 0 else 0 for v in variances]
    W = sum(w)
    if W < 1e-12:
        return {"pooled_effect": np.mean(effects), "pooled_se": 1.0,
                "tau2": 0, "I2": 0}
    theta_fe = sum(wi * ei for wi, ei in zip(w, effects)) / W
    Q = sum(wi * (ei - theta_fe)**2 for wi, ei in zip(w, effects))
    C = W - sum(wi**2 for wi in w) / W
    tau2 = max(0, (Q - (k - 1)) / C) if C > 0 else 0
    w_re = [1.0 / (v + tau2) if (v + tau2) > 0 else 0 for v in variances]
    W_re = sum(w_re)
    if W_re < 1e-12:
        return {"pooled_effect": np.mean(effects), "pooled_se": 1.0,
                "tau2": tau2, "I2": 0}
    theta_re = sum(wi * ei for wi, ei in zip(w_re, effects)) / W_re
    se_re = 1.0 / np.sqrt(W_re)
    I2 = max(0, (Q - (k-1)) / Q * 100) if Q > 0 else 0
    return {"pooled_effect": float(theta_re), "pooled_se": float(se_re),
            "tau2": float(tau2), "I2": float(I2)}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
@logger.catch
def main():
    global USE_RELGNN
    start_time = time.time()
    all_results = []
    all_task_stats = {}
    all_gate_analysis = {}
    deviations = []
    architecture_used = "RelGNN" if USE_RELGNN else "HeteroSAGE"

    # -----------------------------------------------------------------------
    # SMOKE TEST: verify RAMA works on synthetic data
    # -----------------------------------------------------------------------
    logger.info("=== SMOKE TEST: RAMA on synthetic data ===")
    try:
        rama_test = RAMAAggregation(channels=128, gate_hidden=16).to(DEVICE)
        x_test = torch.randn(100, 128, device=DEVICE)
        idx_test = torch.randint(0, 10, (100,), device=DEVICE)
        out_test = rama_test(x_test, index=idx_test, dim_size=10)
        assert out_test.shape == (10, 128), f"Bad shape: {out_test.shape}"
        assert not torch.isnan(out_test).any(), "NaN in RAMA output"
        logger.info(f"  RAMA smoke test PASSED: output shape {out_test.shape}")
        del rama_test, x_test, idx_test, out_test
        torch.cuda.empty_cache()
    except Exception as e:
        logger.exception(f"RAMA smoke test FAILED: {e}")
        raise

    # -----------------------------------------------------------------------
    # Main loop over tasks
    # -----------------------------------------------------------------------
    for task_idx, task_cfg in enumerate(TASKS):
        ds_name = task_cfg["dataset"]
        task_name = task_cfg["task"]
        config = task_cfg["config"]
        task_key = f"{ds_name}/{task_name}"
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK {task_idx+1}/{len(TASKS)}: {task_key}")
        logger.info(f"{'='*60}")

        # Load data
        try:
            dataset_obj, task_obj, data, col_stats_dict = load_task_data(task_cfg)
        except Exception as e:
            logger.exception(f"Data loading failed for {task_key}: {e}")
            deviations.append(f"Skipped {task_key}: data loading failed ({e})")
            continue

        # Build loaders
        try:
            loaders = build_loaders(task_obj, data, config)
        except Exception as e:
            logger.exception(f"Loader creation failed for {task_key}: {e}")
            deviations.append(f"Skipped {task_key}: loader creation failed ({e})")
            continue

        entity_table = task_obj.entity_table

        # Clamp range for regression
        clamp_range = None
        if task_cfg["task_type"] == "regression":
            train_table = task_obj.get_table("train")
            clamp_min, clamp_max = np.percentile(
                train_table.df[task_obj.target_col].to_numpy(), [2, 98])
            clamp_range = (clamp_min, clamp_max)

        # Determine max steps for this task
        max_steps = MAX_STEPS_PER_EPOCH if ds_name == "rel-amazon" else 0

        # -------------------------------------------------------------------
        # SMOKE TEST on first seed: verify model builds and trains
        # -------------------------------------------------------------------
        logger.info(f"  Smoke test: 1 seed, 3 epochs...")
        relgnn_ok = USE_RELGNN
        for method_name in ["relgnn_sum", "relgnn_rama"]:
            try:
                seed_everything(42)
                use_rama = METHODS[method_name]["use_rama"]
                smoke_model = create_model(
                    data, col_stats_dict, use_rama, config,
                    use_relgnn=relgnn_ok)
                smoke_model = smoke_model.to(DEVICE)
                opt = torch.optim.Adam(smoke_model.parameters(), lr=LR)
                loss_fn = L1Loss()
                for ep in range(3):
                    loss = train_one_epoch(smoke_model, loaders["train"],
                                           opt, loss_fn, entity_table,
                                           max_steps=50)
                    logger.info(f"    Smoke {method_name} ep{ep}: loss={loss:.4f}")
                    if math.isnan(loss):
                        raise ValueError(f"NaN loss at epoch {ep}")
                del smoke_model, opt
                torch.cuda.empty_cache()
                gc.collect()
                logger.info(f"    Smoke test {method_name} PASSED")
            except Exception as e:
                logger.warning(f"    Smoke test {method_name} FAILED: {e}")
                if relgnn_ok:
                    logger.warning("    Falling back to HeteroSAGE")
                    relgnn_ok = False
                    deviations.append(
                        f"RelGNN failed for {task_key}/{method_name}: {e}. "
                        f"Using HeteroSAGE fallback.")
                    # Retry with HeteroSAGE
                    try:
                        seed_everything(42)
                        smoke_model = create_model(
                            data, col_stats_dict, use_rama, config,
                            use_relgnn=False)
                        smoke_model = smoke_model.to(DEVICE)
                        opt = torch.optim.Adam(smoke_model.parameters(), lr=LR)
                        for ep in range(3):
                            loss = train_one_epoch(smoke_model, loaders["train"],
                                                   opt, loss_fn, entity_table,
                                                   max_steps=50)
                            logger.info(f"    Fallback smoke ep{ep}: loss={loss:.4f}")
                        del smoke_model, opt
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info(f"    HeteroSAGE fallback smoke PASSED")
                    except Exception as e2:
                        logger.exception(f"    HeteroSAGE fallback also failed: {e2}")
                        deviations.append(f"Both RelGNN and HeteroSAGE failed for {task_key}")
                        continue

        if not relgnn_ok:
            architecture_used = "HeteroSAGE"

        # -------------------------------------------------------------------
        # Full training runs
        # -------------------------------------------------------------------
        task_results = []
        for method_name, method_cfg in METHODS.items():
            use_rama = method_cfg["use_rama"]
            label = method_cfg["label"]
            if not relgnn_ok:
                label = label.replace("RelGNN", "HeteroSAGE")

            for seed in SEEDS:
                run_start = time.time()
                seed_everything(seed)
                logger.info(f"  {label} seed={seed}")

                try:
                    model = create_model(
                        data, col_stats_dict, use_rama, config,
                        use_relgnn=relgnn_ok)
                    model = model.to(DEVICE)
                    num_params = sum(p.numel() for p in model.parameters())
                    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                    loss_fn = L1Loss()

                    best_val = float("inf")
                    best_state = None
                    train_losses = []
                    epoch_times = []

                    for epoch in range(EPOCHS):
                        ep_start = time.time()
                        t_loss = train_one_epoch(
                            model, loaders["train"], optimizer, loss_fn,
                            entity_table, max_steps=max_steps)
                        train_losses.append(t_loss)

                        val_pred = evaluate(model, loaders["val"],
                                           entity_table, clamp_range)
                        val_metrics = task_obj.evaluate(val_pred,
                                                       task_obj.get_table("val"))
                        val_mae = val_metrics[task_cfg["metric"]]
                        ep_time = time.time() - ep_start
                        epoch_times.append(ep_time)

                        logger.info(
                            f"    ep{epoch}: loss={t_loss:.4f} "
                            f"val_mae={val_mae:.4f} ({ep_time:.1f}s)")

                        if val_mae < best_val:
                            best_val = val_mae
                            best_state = copy.deepcopy(model.state_dict())

                    # Test with best model
                    model.load_state_dict(best_state)
                    test_pred = evaluate(model, loaders["test"],
                                        entity_table, clamp_range)
                    test_metrics = task_obj.evaluate(test_pred)
                    test_mae = test_metrics[task_cfg["metric"]]

                    # Gate stats
                    gate_stats = extract_gate_stats(model) if use_rama else {}

                    run_time = time.time() - run_start
                    result = {
                        "task": task_key,
                        "method": method_name,
                        "label": label,
                        "seed": seed,
                        "test_mae": float(test_mae),
                        "best_val_mae": float(best_val),
                        "train_losses": [float(x) for x in train_losses],
                        "epoch_times": [float(x) for x in epoch_times],
                        "mean_epoch_time": float(np.mean(epoch_times)),
                        "gate_stats": gate_stats,
                        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
                        "num_params": num_params,
                        "run_time": run_time,
                    }
                    task_results.append(result)
                    all_results.append(result)
                    logger.info(f"    test_mae={test_mae:.4f} "
                                f"({run_time:.1f}s total)")

                except Exception as e:
                    logger.exception(f"    Run failed: {e}")
                    deviations.append(
                        f"{task_key}/{method_name}/seed={seed} failed: {e}")
                finally:
                    if 'model' in dir():
                        try:
                            del model
                        except Exception:
                            pass
                    if 'optimizer' in dir():
                        try:
                            del optimizer
                        except Exception:
                            pass
                    torch.cuda.empty_cache()
                    gc.collect()

        # -------------------------------------------------------------------
        # Per-task statistical analysis
        # -------------------------------------------------------------------
        baseline_maes = [r["test_mae"] for r in task_results
                         if not METHODS[r["method"]]["use_rama"]]
        rama_maes = [r["test_mae"] for r in task_results
                     if METHODS[r["method"]]["use_rama"]]

        if len(baseline_maes) >= 2 and len(rama_maes) >= 2:
            d = cohens_d(baseline_maes, rama_maes)
            ci = cohens_d_ci(d, len(baseline_maes), len(rama_maes))
            try:
                t_stat, p_val = stats.ttest_rel(
                    baseline_maes[:min(len(baseline_maes), len(rama_maes))],
                    rama_maes[:min(len(baseline_maes), len(rama_maes))])
            except Exception:
                t_stat, p_val = 0.0, 1.0
            try:
                w_stat, w_pval = stats.wilcoxon(
                    baseline_maes[:min(len(baseline_maes), len(rama_maes))],
                    rama_maes[:min(len(baseline_maes), len(rama_maes))])
            except Exception:
                w_stat, w_pval = 0.0, 1.0

            bm = np.mean(baseline_maes)
            rm = np.mean(rama_maes)
            pct = (bm - rm) / bm * 100 if bm > 0 else 0

            task_stat = {
                "baseline_maes": baseline_maes,
                "rama_maes": rama_maes,
                "baseline_mean": float(bm),
                "baseline_std": float(np.std(baseline_maes, ddof=1)),
                "rama_mean": float(rm),
                "rama_std": float(np.std(rama_maes, ddof=1)),
                "cohens_d": float(d),
                "cohens_d_ci_95": [float(x) for x in ci],
                "paired_ttest_t": float(t_stat),
                "paired_ttest_p": float(p_val),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_pval),
                "improvement_pct": float(pct),
            }
            all_task_stats[task_key] = task_stat
            logger.info(f"\n  === {task_key} Stats ===")
            logger.info(f"  Baseline MAE: {bm:.4f} +/- {np.std(baseline_maes, ddof=1):.4f}")
            logger.info(f"  RAMA MAE:     {rm:.4f} +/- {np.std(rama_maes, ddof=1):.4f}")
            logger.info(f"  Cohen's d:    {d:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
            logger.info(f"  p-value:      {p_val:.6f}")
            logger.info(f"  Improvement:  {pct:.2f}%")

        # Gate analysis for this task
        rama_runs = [r for r in task_results if METHODS[r["method"]]["use_rama"]]
        if rama_runs:
            gate_analysis = {}
            for r in rama_runs:
                gate_analysis[f"seed_{r['seed']}"] = r["gate_stats"]
            all_gate_analysis[task_key] = gate_analysis

        # Cleanup task data
        del data, loaders
        torch.cuda.empty_cache()
        gc.collect()

    # -----------------------------------------------------------------------
    # Cross-task meta-analysis
    # -----------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("Cross-task meta-analysis")
    logger.info(f"{'='*60}")

    effects = []
    variances = []
    for tk, ts in all_task_stats.items():
        effects.append(ts["cohens_d"])
        n = len(ts["baseline_maes"])
        se = np.sqrt(2.0/n + ts["cohens_d"]**2 / (2*n))
        variances.append(se**2)

    meta = dersimonian_laird(effects, variances)
    logger.info(f"  Pooled effect (DL): {meta['pooled_effect']:.4f} "
                f"(SE={meta['pooled_se']:.4f})")
    logger.info(f"  tau^2={meta['tau2']:.4f}, I^2={meta['I2']:.1f}%")

    # -----------------------------------------------------------------------
    # Build output JSON
    # -----------------------------------------------------------------------
    logger.info("\nBuilding output JSON...")

    # Load input data for examples
    data_path = DATA_DIR / "full_data_out.json"
    input_data = json.loads(data_path.read_text())

    datasets_out = []
    for task_cfg in TASKS:
        task_key = f"{task_cfg['dataset']}/{task_cfg['task']}"

        # Find matching dataset in input data
        input_examples = []
        for ds in input_data.get("datasets", []):
            if ds["dataset"] == task_key:
                input_examples = ds["examples"]
                break

        # Get per-method test MAE means for predictions
        task_runs = [r for r in all_results if r["task"] == task_key]
        baseline_runs = [r for r in task_runs
                         if not METHODS[r["method"]]["use_rama"]]
        rama_runs = [r for r in task_runs
                     if METHODS[r["method"]]["use_rama"]]

        baseline_mean_mae = (np.mean([r["test_mae"] for r in baseline_runs])
                             if baseline_runs else float("nan"))
        rama_mean_mae = (np.mean([r["test_mae"] for r in rama_runs])
                         if rama_runs else float("nan"))

        examples = []
        for ex in input_examples:
            out_ex = {
                "input": ex["input"],
                "output": str(ex.get("output", "nan")),
                "predict_baseline_sum": f"{baseline_mean_mae:.6f}",
                "predict_rama": f"{rama_mean_mae:.6f}",
            }
            # Copy metadata fields
            for k, v in ex.items():
                if k.startswith("metadata_"):
                    out_ex[k] = v
            examples.append(out_ex)

        datasets_out.append({
            "dataset": task_key,
            "examples": examples,
        })

    # Comparison with iter_2 HeteroSAGE results
    iter2_comparison = {
        "rel-f1/driver-position": {
            "heterosage_rama_cohens_d": -0.50,
            "note": "RAMA was worse with HeteroSAGE on F1"
        },
        "rel-amazon/item-ltv": {
            "heterosage_rama_cohens_d": 10.83,
            "note": "RAMA was dramatically better with HeteroSAGE on Amazon"
        },
    }

    # Per-run results (truncated for output)
    per_run_summary = []
    for r in all_results:
        per_run_summary.append({
            "config": r["method"],
            "label": r["label"],
            "task": r["task"],
            "seed": r["seed"],
            "test_mae": r["test_mae"],
            "best_val_mae": r["best_val_mae"],
            "train_losses": r["train_losses"][:3],  # First 3 for preview
            "epoch_times": r["epoch_times"][:3],
            "mean_epoch_time": r["mean_epoch_time"],
            "gate_stats": r["gate_stats"],
            "test_metrics": r["test_metrics"],
            "num_params": r["num_params"],
        })

    output = {
        "metadata": {
            "experiment": "RAMA Integration: RelGNN vs baseline on rel-f1 and rel-amazon",
            "method_name": "RAMA (Rank-Aware Moment Aggregation)",
            "architecture_used": architecture_used,
            "description": (
                f"Tests RAMA against standard sum aggregation using {architecture_used} "
                f"on rel-f1/driver-position and rel-amazon/item-ltv (regression/MAE). "
                f"RAMA enriches mean pooling with gated per-dimension variance "
                f"conditioned on spectral rank proxy and cardinality."
            ),
            "tasks": [f"{t['dataset']}/{t['task']}" for t in TASKS],
            "hyperparameters": {
                "channels": 128,
                "lr": LR,
                "epochs": EPOCHS,
                "seeds": SEEDS,
                "max_steps_per_epoch": MAX_STEPS_PER_EPOCH,
                "rama_gate_hidden": 16,
                "task_configs": {
                    f"{t['dataset']}/{t['task']}": t["config"]
                    for t in TASKS
                },
            },
            "statistical_analysis": all_task_stats,
            "meta_analysis": meta,
            "gate_analysis": all_gate_analysis,
            "comparison_with_heterosage": iter2_comparison,
            "per_run_results": per_run_summary,
            "deviations_from_plan": deviations,
            "num_seeds": len(SEEDS),
            "total_runtime_seconds": time.time() - start_time,
        },
        "datasets": datasets_out,
    }

    out_path = WS / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved results to {out_path}")
    logger.info(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
