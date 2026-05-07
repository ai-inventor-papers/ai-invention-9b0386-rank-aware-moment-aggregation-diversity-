#!/usr/bin/env python3
"""Synthetic Testbed Validation: Rank Collapse Measurement, RAMA vs Baselines, and Gate Behavior Analysis.

Three-phase experiment on the 3,750-example synthetic testbed:
  (A) Measure rank collapse at FK-join aggregation and validate the variance-entropy proxy
  (B) Compare 5 aggregation methods (mean, ungated mean+var, RAMA, PNA-style, VPA) on
      regression and classification across 12 conditions with 5 seeds
  (C) Analyze RAMA's learned gate values to verify they correlate with child-set diversity
"""

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
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy import stats as scipy_stats
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

# ============================================================
# SETUP
# ============================================================
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
PLOT_DIR = WORKSPACE / "plots"
LOG_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ============================================================
# CONSTANTS
# ============================================================
CHILD_DIM = 16
PARENT_DIM = 8
HIDDEN_DIM = 64
NUM_SEEDS = 5
NUM_EPOCHS = 100
LR = 0.001
PATIENCE = 15
BATCH_SIZE = 64
EPS = 1e-8

DATA_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/leskovec-predictive-residual-message-passing-v2_sti"
    "/3_invention_loop/iter_1/gen_art/data_id5_it1__opus"
)
METHOD_NAMES = ["mean", "ungated_meanvar", "rama", "pna", "vpa"]
TASK_NAMES = ["regression", "classification"]
CONDITIONS = [
    "low_rank2", "low_rank8", "low_rank16",
    "medium_rank2", "medium_rank8", "medium_rank16",
    "high_rank2", "high_rank8", "high_rank16",
    "extreme_rank2", "extreme_rank8", "extreme_rank16",
]
REGIMES = ["low", "medium", "high", "extreme"]
RANKS = [2, 8, 16]

# ============================================================
# HARDWARE & MEMORY
# ============================================================
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
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.7 * 1e9)

try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except (ValueError, resource.error):
    pass
try:
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
except (ValueError, resource.error):
    pass

torch.set_num_threads(max(1, NUM_CPUS))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET_BYTES/1e9:.1f} GB")

# ============================================================
# DATA LOADING
# ============================================================
def load_data(data_path: Path, max_examples: Optional[int] = None) -> dict:
    """Load and parse the synthetic testbed dataset."""
    logger.info(f"Loading data from {data_path} ({data_path.stat().st_size / 1e6:.1f} MB)")
    t0 = time.time()
    raw = json.loads(data_path.read_text())
    logger.info(f"JSON parsed in {time.time()-t0:.1f}s")

    metadata = raw.get("metadata", {})
    examples_raw = raw["datasets"][0]["examples"]
    if max_examples is not None:
        examples_raw = examples_raw[:max_examples]

    logger.info(f"Parsing {len(examples_raw)} examples into numpy arrays...")
    examples = []
    for i, ex in enumerate(examples_raw):
        try:
            inp = json.loads(ex["input"])
            out = json.loads(ex["output"])
            examples.append({
                "idx": i,
                "parent_features": np.array(inp["parent_features"], dtype=np.float32),
                "child_features": np.array(inp["child_features"], dtype=np.float32),
                "num_children": int(inp["num_children"]),
                "regression_target": float(out["regression_target"]),
                "classification_target": int(out["classification_target"]),
                "fold": ex.get("metadata_fold", "train"),
                "condition": ex.get("metadata_condition", "unknown"),
                "cardinality_regime": ex.get("metadata_cardinality_regime", ""),
                "cardinality": int(ex.get("metadata_cardinality", 1)),
                "projection_rank": int(ex.get("metadata_projection_rank", 1)),
                "erank_svd": float(ex.get("metadata_erank_svd", 1.0)),
                "erank_proxy": float(ex.get("metadata_erank_proxy", 0.5)),
                "child_mean": np.array(ex.get("metadata_child_mean", [0.0]*CHILD_DIM), dtype=np.float32),
                "child_variance": np.array(ex.get("metadata_child_variance", [0.0]*CHILD_DIM), dtype=np.float32),
                "original_input": ex["input"],
                "original_output": ex["output"],
                "original_metadata": {k: v for k, v in ex.items() if k.startswith("metadata_")},
            })
        except Exception:
            logger.exception(f"Failed to parse example {i}")
            continue

    # Free raw JSON
    del raw
    gc.collect()

    logger.info(f"Parsed {len(examples)} examples in {time.time()-t0:.1f}s total")

    by_fold = defaultdict(list)
    by_condition = defaultdict(list)
    for ex in examples:
        by_fold[ex["fold"]].append(ex)
        by_condition[ex["condition"]].append(ex)

    logger.info(f"Folds: {', '.join(f'{k}={len(v)}' for k, v in sorted(by_fold.items()))}")
    logger.info(f"Conditions: {len(by_condition)} ({', '.join(sorted(by_condition.keys())[:4])}...)")

    return {
        "metadata": metadata,
        "examples": examples,
        "by_fold": dict(by_fold),
        "by_condition": dict(by_condition),
    }


# ============================================================
# SCATTER OPERATIONS (pure PyTorch, no torch_scatter)
# ============================================================
def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean aggregation: (E, d), (E,) -> (dim_size, d)."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    idx_exp = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx_exp, src)
    ones = torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(1), ones)
    return out / count.clamp(min=1)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum aggregation: (E, d), (E,) -> (dim_size, d)."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    idx_exp = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx_exp, src)
    return out


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Max aggregation: (E, d), (E,) -> (dim_size, d). Differentiable."""
    try:
        out = src.new_full((dim_size, src.size(1)), float('-inf'))
        idx_exp = index.unsqueeze(1).expand_as(src)
        out.scatter_reduce_(0, idx_exp, src, reduce="amax")
        return out
    except Exception:
        parts = []
        for j in range(dim_size):
            mask = (index == j)
            if mask.any():
                parts.append(src[mask].max(dim=0).values)
            else:
                parts.append(src.new_full((src.size(1),), float('-inf')))
        return torch.stack(parts, dim=0)


def scatter_count(index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Count elements per group: (E,) -> (dim_size, 1)."""
    ones = torch.ones(index.size(0), 1, device=index.device)
    count = torch.zeros(dim_size, 1, device=index.device)
    count.scatter_add_(0, index.unsqueeze(1), ones)
    return count


# ============================================================
# MODEL DEFINITIONS
# ============================================================
class ChildEncoder(nn.Module):
    """Linear(16, 64) + ReLU."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(CHILD_DIM, HIDDEN_DIM)

    def forward(self, x):
        return F.relu(self.fc(x))


class ParentEncoder(nn.Module):
    """Linear(8, 64) + ReLU."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(PARENT_DIM, HIDDEN_DIM)

    def forward(self, x):
        return F.relu(self.fc(x))


class PredictionHead(nn.Module):
    """MLP: Linear(in, 64) + ReLU + Linear(64, 1)."""
    def __init__(self, input_dim: int = 2 * HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class MeanAggModel(nn.Module):
    """Baseline: mean aggregation of child embeddings."""
    def __init__(self):
        super().__init__()
        self.child_enc = ChildEncoder()
        self.parent_enc = ParentEncoder()
        self.head = PredictionHead(2 * HIDDEN_DIM)

    def forward(self, x_children, parent_idx, x_parents, dim_size):
        h = self.child_enc(x_children)
        agg = scatter_mean(h, parent_idx, dim_size)
        hp = self.parent_enc(x_parents)
        return self.head(torch.cat([agg, hp], dim=1))


class UngatedMeanVarModel(nn.Module):
    """Baseline: concatenate mean and variance, project down."""
    def __init__(self):
        super().__init__()
        self.child_enc = ChildEncoder()
        self.parent_enc = ParentEncoder()
        self.proj = nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM)
        self.head = PredictionHead(2 * HIDDEN_DIM)

    def forward(self, x_children, parent_idx, x_parents, dim_size):
        h = self.child_enc(x_children)
        mu = scatter_mean(h, parent_idx, dim_size)
        mean_x2 = scatter_mean(h * h, parent_idx, dim_size)
        var = (mean_x2 - mu * mu).clamp(min=0)
        agg = F.relu(self.proj(torch.cat([mu, var], dim=1)))
        hp = self.parent_enc(x_parents)
        return self.head(torch.cat([agg, hp], dim=1))


class RAMAModel(nn.Module):
    """RAMA: Rank-Aware Moment Aggregation (our method)."""
    def __init__(self, gate_hidden: int = 16):
        super().__init__()
        self.child_enc = ChildEncoder()
        self.parent_enc = ParentEncoder()

        # RAMA-specific layers
        self.W_sigma = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.W_g = nn.Linear(2, gate_hidden)
        self.gate_out = nn.Linear(gate_hidden, HIDDEN_DIM)
        self.head = PredictionHead(2 * HIDDEN_DIM)

        # Zero-init so RAMA starts as pure mean aggregation
        nn.init.zeros_(self.W_sigma.weight)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)

    def _rama_aggregate(self, h_child, parent_idx, dim_size):
        """Compute RAMA aggregation. Returns (agg, gate, rho, N)."""
        mu = scatter_mean(h_child, parent_idx, dim_size)
        mean_x2 = scatter_mean(h_child * h_child, parent_idx, dim_size)
        var = (mean_x2 - mu * mu).clamp(min=0)

        N = scatter_count(parent_idx, dim_size)

        # Effective rank proxy: normalized variance entropy
        var_sum = var.sum(dim=-1, keepdim=True).clamp(min=EPS)
        p = (var / var_sum).clamp(min=EPS)
        H = -(p * p.log()).sum(dim=-1, keepdim=True)
        log_d = math.log(HIDDEN_DIM)
        rho = H / log_d  # ~[0, 1]

        # Gating network: inputs are log(N+1) and rho
        gate_input = torch.cat([torch.log1p(N), rho], dim=-1)
        g = torch.sigmoid(self.gate_out(F.relu(self.W_g(gate_input))))

        # RAMA output: mean + gated variance transform
        agg = mu + g * self.W_sigma(var)
        return agg, g, rho, N

    def forward(self, x_children, parent_idx, x_parents, dim_size):
        h = self.child_enc(x_children)
        agg, _, _, _ = self._rama_aggregate(h, parent_idx, dim_size)
        hp = self.parent_enc(x_parents)
        return self.head(torch.cat([agg, hp], dim=1))

    def get_gate_values(self, x_children, parent_idx, x_parents, dim_size):
        """Extract gate values for Phase C analysis."""
        h = self.child_enc(x_children)
        _, g, rho, N = self._rama_aggregate(h, parent_idx, dim_size)
        return g, rho, N


class PNAModel(nn.Module):
    """PNA-style: mean + std + max with degree scaling."""
    def __init__(self):
        super().__init__()
        self.child_enc = ChildEncoder()
        self.parent_enc = ParentEncoder()
        self.proj = nn.Linear(3 * HIDDEN_DIM, HIDDEN_DIM)
        self.head = PredictionHead(2 * HIDDEN_DIM)
        self.register_buffer('avg_deg_log', torch.tensor(1.0))

    def set_avg_degree(self, train_degrees: list):
        self.avg_deg_log = torch.tensor(float(np.log(np.mean(train_degrees) + 1)))

    def forward(self, x_children, parent_idx, x_parents, dim_size):
        h = self.child_enc(x_children)
        mu = scatter_mean(h, parent_idx, dim_size)
        mean_x2 = scatter_mean(h * h, parent_idx, dim_size)
        var = (mean_x2 - mu * mu).clamp(min=0)
        std = torch.sqrt(var + 1e-8)
        mx = scatter_max(h, parent_idx, dim_size)

        N = scatter_count(parent_idx, dim_size)
        scaler = torch.log(N + 1) / self.avg_deg_log.clamp(min=1e-8)

        agg = F.relu(self.proj(torch.cat([mu, std, mx], dim=1))) * scaler
        hp = self.parent_enc(x_parents)
        return self.head(torch.cat([agg, hp], dim=1))


class VPAModel(nn.Module):
    """VPA: Variance-Preserving Aggregation (sum / sqrt(N))."""
    def __init__(self):
        super().__init__()
        self.child_enc = ChildEncoder()
        self.parent_enc = ParentEncoder()
        self.head = PredictionHead(2 * HIDDEN_DIM)

    def forward(self, x_children, parent_idx, x_parents, dim_size):
        h = self.child_enc(x_children)
        s = scatter_sum(h, parent_idx, dim_size)
        N = scatter_count(parent_idx, dim_size)
        agg = s / torch.sqrt(N.clamp(min=1))
        hp = self.parent_enc(x_parents)
        return self.head(torch.cat([agg, hp], dim=1))


MODEL_CLASSES = {
    "mean": MeanAggModel,
    "ungated_meanvar": UngatedMeanVarModel,
    "rama": RAMAModel,
    "pna": PNAModel,
    "vpa": VPAModel,
}


# ============================================================
# DATASET & DATALOADER
# ============================================================
class RelationalDataset(Dataset):
    def __init__(self, examples: list):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "parent_features": torch.from_numpy(ex["parent_features"]),
            "child_features": torch.from_numpy(ex["child_features"]),
            "regression_target": torch.tensor(ex["regression_target"], dtype=torch.float32),
            "classification_target": torch.tensor(float(ex["classification_target"]), dtype=torch.float32),
            "idx": ex["idx"],
        }


def collate_fn(batch):
    """Concatenate variable-length child sets and build parent_idx."""
    parents = torch.stack([b["parent_features"] for b in batch])
    y_reg = torch.stack([b["regression_target"] for b in batch])
    y_cls = torch.stack([b["classification_target"] for b in batch])
    indices = [b["idx"] for b in batch]

    children_list = []
    pidx_list = []
    for i, b in enumerate(batch):
        c = b["child_features"]
        children_list.append(c)
        pidx_list.append(torch.full((c.size(0),), i, dtype=torch.long))

    return {
        "x_children": torch.cat(children_list, dim=0),
        "parent_idx": torch.cat(pidx_list, dim=0),
        "x_parents": parents,
        "y_reg": y_reg,
        "y_cls": y_cls,
        "indices": indices,
    }


def make_loader(examples: list, shuffle: bool = False) -> DataLoader:
    bs = min(BATCH_SIZE, len(examples))
    return DataLoader(
        RelationalDataset(examples), batch_size=bs, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )


# ============================================================
# TRAINING INFRASTRUCTURE
# ============================================================
def train_one_model(
    method_name: str, task: str, train_examples: list, val_examples: list,
    seed: int, train_degrees: list,
) -> tuple:
    """Train one model. Returns (best_state_dict, best_val_loss, epochs_trained)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MODEL_CLASSES[method_name]()
    if method_name == "pna":
        model.set_avg_degree(train_degrees)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()

    train_loader = make_loader(train_examples, shuffle=True)
    val_loader = make_loader(val_examples, shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    patience_ctr = 0

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(
                batch["x_children"], batch["parent_idx"],
                batch["x_parents"], batch["x_parents"].size(0),
            ).squeeze(-1)
            target = batch["y_reg"] if task == "regression" else batch["y_cls"]
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch["x_children"], batch["parent_idx"],
                    batch["x_parents"], batch["x_parents"].size(0),
                ).squeeze(-1)
                target = batch["y_reg"] if task == "regression" else batch["y_cls"]
                val_loss_sum += criterion(pred, target).item() * pred.size(0)
                val_n += pred.size(0)

        val_loss = val_loss_sum / max(val_n, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    return best_state, best_val_loss, epoch + 1


def predict_all(model: nn.Module, examples: list, task: str) -> dict:
    """Predict on all examples. Returns {idx: prediction_value}."""
    model.eval()
    loader = make_loader(examples, shuffle=False)
    predictions = {}
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["x_children"], batch["parent_idx"],
                batch["x_parents"], batch["x_parents"].size(0),
            ).squeeze(-1)
            for i, idx in enumerate(batch["indices"]):
                if task == "regression":
                    predictions[idx] = pred[i].item()
                else:
                    predictions[idx] = torch.sigmoid(pred[i]).item()
    return predictions


def evaluate_predictions(predictions: dict, examples: list, task: str) -> dict:
    """Compute metrics for predictions against ground truth."""
    y_true, y_pred = [], []
    for ex in examples:
        if ex["idx"] in predictions:
            y_true.append(ex["regression_target"] if task == "regression" else ex["classification_target"])
            y_pred.append(predictions[ex["idx"]])
    if not y_true:
        return {"mse": float('nan'), "r2": float('nan')} if task == "regression" else {"accuracy": float('nan'), "auroc": float('nan')}

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if task == "regression":
        mse = float(np.mean((y_true - y_pred) ** 2))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
        return {"mse": mse, "r2": r2}
    else:
        acc = float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
        try:
            auroc = float(roc_auc_score(y_true, y_pred))
        except ValueError:
            auroc = 0.5
        return {"accuracy": acc, "auroc": auroc}


# ============================================================
# PHASE A: RANK COLLAPSE ANALYSIS
# ============================================================
def phase_a_rank_analysis(data: dict) -> dict:
    """Measure rank collapse and validate variance-entropy proxy."""
    logger.info("=" * 60)
    logger.info("PHASE A: Rank Collapse Analysis")
    logger.info("=" * 60)

    examples = data["examples"]
    by_condition = data["by_condition"]

    # --- A1: Overall proxy vs SVD correlation ---
    all_svd = np.array([ex["erank_svd"] for ex in examples])
    all_proxy = np.array([ex["erank_proxy"] for ex in examples])

    if len(all_svd) > 2:
        sp_all = scipy_stats.spearmanr(all_svd, all_proxy)
        pe_all = scipy_stats.pearsonr(all_svd, all_proxy)
        sp_rho, sp_p = float(sp_all.statistic), float(sp_all.pvalue)
        pe_r, pe_p = float(pe_all.statistic), float(pe_all.pvalue)
    else:
        sp_rho, sp_p, pe_r, pe_p = 0.0, 1.0, 0.0, 1.0
    logger.info(f"Proxy vs SVD overall: Spearman={sp_rho:.4f} Pearson={pe_r:.4f}")

    # --- A1b: Per-cardinality correlations ---
    per_card_corr = {}
    for regime in REGIMES:
        regime_exs = [ex for ex in examples if ex["cardinality_regime"] == regime]
        if len(regime_exs) > 2:
            svds = np.array([ex["erank_svd"] for ex in regime_exs])
            prxs = np.array([ex["erank_proxy"] for ex in regime_exs])
            sp = scipy_stats.spearmanr(svds, prxs)
            per_card_corr[regime] = {"spearman": float(sp.statistic), "pvalue": float(sp.pvalue)}
            logger.info(f"  {regime}: Spearman={sp.statistic:.4f} (n={len(regime_exs)})")
        else:
            per_card_corr[regime] = {"spearman": 0.0, "pvalue": 1.0}

    # --- A2 & A3: Per-condition rank statistics ---
    per_cond_stats = []
    for cond in CONDITIONS:
        cexs = by_condition.get(cond, [])
        if not cexs:
            continue
        eranks_svd = np.array([e["erank_svd"] for e in cexs])
        eranks_proxy = np.array([e["erank_proxy"] for e in cexs])
        card = cexs[0]["cardinality"]
        proj_rank = cexs[0]["projection_rank"]
        max_rank = min(card, CHILD_DIM)
        collapse_ratios = eranks_svd / max_rank

        # Effective rank of mean vectors (measures aggregation diversity)
        mean_vecs = np.stack([e["child_mean"] for e in cexs])
        erank_of_means = _compute_erank_of_matrix(mean_vecs)

        stat = {
            "condition": cond, "cardinality": card, "projection_rank": proj_rank,
            "num_examples": len(cexs), "max_possible_rank": max_rank,
            "mean_erank_svd": float(np.mean(eranks_svd)),
            "std_erank_svd": float(np.std(eranks_svd)),
            "mean_erank_proxy": float(np.mean(eranks_proxy)),
            "std_erank_proxy": float(np.std(eranks_proxy)),
            "erank_of_means": erank_of_means,
            "mean_collapse_ratio": float(np.mean(collapse_ratios)),
            "std_collapse_ratio": float(np.std(collapse_ratios)),
        }
        per_cond_stats.append(stat)
        logger.info(f"  {cond}: erank={stat['mean_erank_svd']:.2f}+/-{stat['std_erank_svd']:.2f} "
                     f"collapse={stat['mean_collapse_ratio']:.3f} erank_means={erank_of_means:.2f}")

    # --- A3: Verify SVD recomputation (sample) ---
    recompute_errs = []
    for cond in CONDITIONS:
        for ex in by_condition.get(cond, [])[:20]:
            C = ex["child_features"]
            recomp = _compute_erank_from_child_matrix(C)
            recompute_errs.append(abs(recomp - ex["erank_svd"]))
    mean_recomp_err = float(np.mean(recompute_errs)) if recompute_errs else 0.0
    logger.info(f"SVD recompute verification: mean abs error = {mean_recomp_err:.6f}")

    # --- Plots ---
    _plot_proxy_vs_svd(examples)
    _plot_heatmap(per_cond_stats, "mean_erank_svd", "Mean SVD Effective Rank", "YlOrRd", "plot_rank_heatmap.png", fmt=".2f")
    _plot_heatmap(per_cond_stats, "mean_collapse_ratio", "Mean Collapse Ratio", "RdYlGn_r", "plot_collapse_heatmap.png", fmt=".3f")

    return {
        "proxy_vs_svd_correlation": {
            "spearman": sp_rho, "spearman_pvalue": sp_p,
            "pearson": pe_r, "pearson_pvalue": pe_p,
        },
        "per_cardinality_correlation": per_card_corr,
        "per_condition_rank_stats": per_cond_stats,
        "svd_recompute_mean_error": mean_recomp_err,
    }


def _compute_erank_of_matrix(M: np.ndarray) -> float:
    """Compute effective rank of a matrix via SVD entropy."""
    if M.shape[0] < 2:
        return 1.0
    Mc = M - M.mean(axis=0)
    cov = Mc.T @ Mc / M.shape[0]
    sigmas = np.linalg.svd(cov, compute_uv=False)
    sigmas = sigmas[sigmas > EPS]
    if len(sigmas) == 0:
        return 1.0
    p = sigmas / sigmas.sum()
    H = -np.sum(p * np.log(p + EPS))
    return float(np.exp(H))


def _compute_erank_from_child_matrix(C: np.ndarray) -> float:
    """Recompute SVD effective rank from raw child features."""
    if C.shape[0] < 2:
        return 1.0
    Cc = C - C.mean(axis=0)
    cov = Cc.T @ Cc / C.shape[0]
    sigmas = np.linalg.svd(cov, compute_uv=False)
    sigmas = sigmas[sigmas > EPS]
    if len(sigmas) == 0:
        return 1.0
    p = sigmas / sigmas.sum()
    H = -np.sum(p * np.log(p + EPS))
    return float(np.exp(H))


def _plot_proxy_vs_svd(examples: list):
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        conds = sorted(set(ex["condition"] for ex in examples))
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(conds), 1)))
        cond2c = dict(zip(conds, colors))
        for c in conds:
            exs = [e for e in examples if e["condition"] == c]
            ax.scatter([e["erank_svd"] for e in exs], [e["erank_proxy"] for e in exs],
                       c=[cond2c[c]], label=c, alpha=0.5, s=10)
        ax.set_xlabel("SVD Effective Rank")
        ax.set_ylabel("Variance Entropy Proxy")
        ax.set_title("SVD Effective Rank vs Variance Entropy Proxy")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.tight_layout()
        plt.savefig(str(PLOT_DIR / "plot_proxy_vs_svd.png"), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved plot_proxy_vs_svd.png")
    except Exception:
        logger.exception("Failed to create proxy_vs_svd plot")


def _plot_heatmap(per_cond_stats: list, key: str, title: str, cmap: str, fname: str, fmt: str = ".2f"):
    try:
        data = np.full((len(REGIMES), len(RANKS)), np.nan)
        for stat in per_cond_stats:
            parts = stat["condition"].split("_rank")
            if len(parts) == 2:
                ri = REGIMES.index(parts[0]) if parts[0] in REGIMES else -1
                rj = RANKS.index(int(parts[1])) if int(parts[1]) in RANKS else -1
                if ri >= 0 and rj >= 0:
                    data[ri, rj] = stat[key]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(data, annot=True, fmt=fmt, xticklabels=[f"rank{r}" for r in RANKS],
                    yticklabels=REGIMES, cmap=cmap, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(str(PLOT_DIR / fname), dpi=150)
        plt.close()
        logger.info(f"Saved {fname}")
    except Exception:
        logger.exception(f"Failed to create {fname}")


# ============================================================
# PHASE B: MODEL COMPARISON
# ============================================================
def phase_b_model_comparison(data: dict) -> tuple:
    """Train and compare 5 aggregation methods. Returns (results_dict, rama_states)."""
    logger.info("=" * 60)
    logger.info("PHASE B: Model Comparison")
    logger.info("=" * 60)

    by_fold = data["by_fold"]
    examples = data["examples"]
    train_ex = by_fold.get("train", [])
    val_ex = by_fold.get("val", [])
    test_ex = by_fold.get("test", [])

    if not train_ex or not val_ex:
        logger.warning("Insufficient data for training (empty train or val set)")
        return {"results_summary": {}, "all_predictions": {}, "training_time_seconds": 0}, {}

    train_degrees = [ex["cardinality"] for ex in train_ex]

    all_results = {}       # (method, task, seed) -> metrics dict
    all_predictions = {}   # (method, task, seed) -> {idx: pred}
    rama_reg_states = {}   # seed -> state_dict (for Phase C reuse)

    total_runs = len(METHOD_NAMES) * len(TASK_NAMES) * NUM_SEEDS
    run_count = 0
    t_start = time.time()

    for method in METHOD_NAMES:
        for task in TASK_NAMES:
            for seed in range(NUM_SEEDS):
                run_count += 1
                t_run = time.time()
                logger.info(f"[{run_count}/{total_runs}] {method}/{task}/seed{seed}")

                try:
                    best_state, best_val, epochs = train_one_model(
                        method, task, train_ex, val_ex, seed, train_degrees,
                    )

                    # Build model, load weights
                    model = MODEL_CLASSES[method]()
                    if method == "pna":
                        model.set_avg_degree(train_degrees)
                    model.load_state_dict(best_state)

                    # Save RAMA regression states for Phase C
                    if method == "rama" and task == "regression":
                        rama_reg_states[seed] = {k: v.clone() for k, v in best_state.items()}

                    # Predict on all examples (for output) and evaluate test
                    preds_all = predict_all(model, examples, task)
                    all_predictions[(method, task, seed)] = preds_all

                    overall = evaluate_predictions(preds_all, test_ex, task)
                    per_cond = {}
                    for cond in CONDITIONS:
                        cond_test = [e for e in test_ex if e["condition"] == cond]
                        if cond_test:
                            per_cond[cond] = evaluate_predictions(preds_all, cond_test, task)

                    all_results[(method, task, seed)] = {
                        "overall": overall, "per_condition": per_cond,
                        "val_loss": best_val, "epochs": epochs,
                    }

                    dt = time.time() - t_run
                    m_str = (f"MSE={overall.get('mse',0):.4f} R2={overall.get('r2',0):.3f}"
                             if task == "regression"
                             else f"Acc={overall.get('accuracy',0):.4f} AUC={overall.get('auroc',0):.3f}")
                    logger.info(f"  {dt:.1f}s, {epochs}ep, val={best_val:.4f}, {m_str}")

                    del model, best_state
                    gc.collect()

                except Exception:
                    logger.exception(f"FAILED: {method}/{task}/seed{seed}")
                    all_results[(method, task, seed)] = None

    total_time = time.time() - t_start
    logger.info(f"Phase B done in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Aggregate results
    summary = _aggregate_results(all_results)
    _plot_comparison(summary, "regression")
    _plot_comparison(summary, "classification")

    return {
        "results_summary": summary,
        "all_predictions": all_predictions,
        "training_time_seconds": total_time,
    }, rama_reg_states


def _aggregate_results(all_results: dict) -> dict:
    """Aggregate per-seed results into mean +/- std, compute Cohen's d."""
    summary = {}
    for task in TASK_NAMES:
        task_sum = {"overall": [], "per_condition": []}

        for method in METHOD_NAMES:
            # Overall
            seeds = [all_results.get((method, task, s)) for s in range(NUM_SEEDS)]
            seeds = [s for s in seeds if s is not None]
            if seeds:
                if task == "regression":
                    mses = [s["overall"]["mse"] for s in seeds]
                    r2s = [s["overall"]["r2"] for s in seeds]
                    task_sum["overall"].append({
                        "method": method,
                        "mean_mse": float(np.mean(mses)), "std_mse": float(np.std(mses)),
                        "mean_r2": float(np.mean(r2s)), "std_r2": float(np.std(r2s)),
                    })
                else:
                    accs = [s["overall"]["accuracy"] for s in seeds]
                    aucs = [s["overall"]["auroc"] for s in seeds]
                    task_sum["overall"].append({
                        "method": method,
                        "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
                        "mean_auroc": float(np.mean(aucs)), "std_auroc": float(np.std(aucs)),
                    })

            # Per condition
            for cond in CONDITIONS:
                sc = [s for s in seeds if s and cond in s["per_condition"]]
                if sc:
                    if task == "regression":
                        ms = [s["per_condition"][cond]["mse"] for s in sc]
                        rs = [s["per_condition"][cond]["r2"] for s in sc]
                        task_sum["per_condition"].append({
                            "condition": cond, "method": method,
                            "mean_mse": float(np.mean(ms)), "std_mse": float(np.std(ms)),
                            "mean_r2": float(np.mean(rs)), "std_r2": float(np.std(rs)),
                        })
                    else:
                        acs = [s["per_condition"][cond]["accuracy"] for s in sc]
                        aus = [s["per_condition"][cond]["auroc"] for s in sc]
                        task_sum["per_condition"].append({
                            "condition": cond, "method": method,
                            "mean_accuracy": float(np.mean(acs)), "std_accuracy": float(np.std(acs)),
                            "mean_auroc": float(np.mean(aus)), "std_auroc": float(np.std(aus)),
                        })

        task_sum["cohen_d_rama_vs_mean"] = _compute_cohen_d(all_results, task)
        summary[task] = task_sum
    return summary


def _compute_cohen_d(all_results: dict, task: str) -> dict:
    """Cohen's d for RAMA vs Mean, per condition and DerSimonian-Laird pooled."""
    per_cond_d = []
    for cond in CONDITIONS:
        rama_vals, mean_vals = [], []
        for seed in range(NUM_SEEDS):
            r_res = all_results.get(("rama", task, seed))
            m_res = all_results.get(("mean", task, seed))
            if r_res and m_res and cond in r_res["per_condition"] and cond in m_res["per_condition"]:
                metric = "mse" if task == "regression" else "accuracy"
                rama_vals.append(r_res["per_condition"][cond][metric])
                mean_vals.append(m_res["per_condition"][cond][metric])

        if len(rama_vals) >= 2:
            ra, ma = np.array(rama_vals), np.array(mean_vals)
            # positive d = RAMA better (lower MSE or higher accuracy)
            diff = (ma - ra) if task == "regression" else (ra - ma)
            pooled_std = np.sqrt((np.var(ra, ddof=1) + np.var(ma, ddof=1)) / 2)
            d = float(np.mean(diff) / max(pooled_std, EPS))
            _, pv = scipy_stats.ttest_rel(ra, ma)
            per_cond_d.append({"condition": cond, "cohen_d": d, "p_value": float(pv), "n_seeds": len(rama_vals)})

    pooled_d, pooled_p = _dersimonian_laird(per_cond_d)
    return {"per_condition": per_cond_d, "pooled_d": pooled_d, "pooled_p": pooled_p}


def _dersimonian_laird(items: list) -> tuple:
    """Random-effects meta-analysis for pooled Cohen's d."""
    if not items:
        return 0.0, 1.0
    ds = np.array([x["cohen_d"] for x in items])
    ns = np.array([x["n_seeds"] for x in items], dtype=float)
    vs = 2.0 / ns + ds**2 / (2.0 * ns)
    ws = 1.0 / (vs + EPS)
    d_fe = np.sum(ws * ds) / np.sum(ws)
    Q = np.sum(ws * (ds - d_fe)**2)
    k = len(ds)
    C = np.sum(ws) - np.sum(ws**2) / np.sum(ws)
    tau2 = max(0.0, (Q - k + 1) / max(C, EPS))
    ws_re = 1.0 / (vs + tau2 + EPS)
    d_pooled = float(np.sum(ws_re * ds) / np.sum(ws_re))
    var_pooled = float(1.0 / np.sum(ws_re))
    z = d_pooled / max(np.sqrt(var_pooled), EPS)
    p = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
    return d_pooled, p


def _plot_comparison(summary: dict, task: str):
    try:
        task_data = summary.get(task, {})
        per_cond = task_data.get("per_condition", [])
        if not per_cond:
            return
        metric = "mean_mse" if task == "regression" else "mean_accuracy"
        std_key = "std_mse" if task == "regression" else "std_accuracy"
        ylabel = "MSE (lower=better)" if task == "regression" else "Accuracy (higher=better)"

        avail_conds = sorted(set(x["condition"] for x in per_cond))
        ncols = min(4, len(avail_conds))
        nrows = math.ceil(len(avail_conds) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
        if nrows * ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for idx_c, cond in enumerate(avail_conds):
            ax = axes[idx_c]
            cd = [x for x in per_cond if x["condition"] == cond]
            methods, vals, errs = [], [], []
            for mi, m in enumerate(METHOD_NAMES):
                md = [x for x in cd if x["method"] == m]
                if md:
                    methods.append(m)
                    vals.append(md[0][metric])
                    errs.append(md[0][std_key])
            x = np.arange(len(methods))
            ax.bar(x, vals, yerr=errs, color=colors[:len(methods)], capsize=3, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=6)
            ax.set_title(cond, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=6)

        for idx_c in range(len(avail_conds), len(axes)):
            axes[idx_c].set_visible(False)

        plt.suptitle(f"{task.capitalize()}: Method Comparison by Condition", fontsize=12)
        plt.tight_layout()
        plt.savefig(str(PLOT_DIR / f"plot_{task}_comparison.png"), dpi=150)
        plt.close()
        logger.info(f"Saved plot_{task}_comparison.png")
    except Exception:
        logger.exception(f"Failed to create {task} comparison plot")


# ============================================================
# PHASE C: GATE ANALYSIS
# ============================================================
def phase_c_gate_analysis(data: dict, rama_reg_states: dict) -> dict:
    """Analyze RAMA gate values using saved regression models."""
    logger.info("=" * 60)
    logger.info("PHASE C: Gate Analysis")
    logger.info("=" * 60)

    test_ex = data["by_fold"].get("test", [])
    if not test_ex or not rama_reg_states:
        logger.warning("No test data or RAMA states for gate analysis")
        return {}

    test_lookup = {ex["idx"]: ex for ex in test_ex}
    all_gate_data = []

    for seed, state in rama_reg_states.items():
        logger.info(f"Extracting gates from RAMA seed {seed}...")
        model = RAMAModel()
        model.load_state_dict(state)
        model.eval()

        loader = make_loader(test_ex, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                g, rho, N = model.get_gate_values(
                    batch["x_children"], batch["parent_idx"],
                    batch["x_parents"], batch["x_parents"].size(0),
                )
                for i, idx in enumerate(batch["indices"]):
                    ex = test_lookup[idx]
                    all_gate_data.append({
                        "seed": seed, "idx": idx,
                        "condition": ex["condition"],
                        "cardinality": ex["cardinality"],
                        "projection_rank": ex["projection_rank"],
                        "erank_svd": ex["erank_svd"],
                        "erank_proxy": ex["erank_proxy"],
                        "gate_mean": float(g[i].mean().item()),
                        "gate_std": float(g[i].std().item()),
                        "rho": float(rho[i].item()),
                        "N": float(N[i].item()),
                    })
        del model
        gc.collect()

    if not all_gate_data:
        return {}

    # Per-condition gate stats
    gate_by_cond = defaultdict(lambda: {"gates": [], "rhos": []})
    for gd in all_gate_data:
        gate_by_cond[gd["condition"]]["gates"].append(gd["gate_mean"])
        gate_by_cond[gd["condition"]]["rhos"].append(gd["rho"])

    per_cond_gate = []
    gate_heatmap = {}
    for cond in CONDITIONS:
        if cond in gate_by_cond:
            gs = np.array(gate_by_cond[cond]["gates"])
            rs = np.array(gate_by_cond[cond]["rhos"])
            stat = {
                "condition": cond,
                "mean_gate": float(np.mean(gs)), "std_gate": float(np.std(gs)),
                "mean_rho": float(np.mean(rs)), "std_rho": float(np.std(rs)),
            }
            per_cond_gate.append(stat)
            gate_heatmap[cond] = stat["mean_gate"]
            logger.info(f"  {cond}: gate={stat['mean_gate']:.4f}+/-{stat['std_gate']:.4f} rho={stat['mean_rho']:.4f}")

    # Correlations
    gm = [gd["gate_mean"] for gd in all_gate_data]
    er = [gd["erank_svd"] for gd in all_gate_data]
    lc = [np.log(gd["cardinality"] + 1) for gd in all_gate_data]

    if len(gm) > 2 and len(set(er)) > 1:
        ge_sp = scipy_stats.spearmanr(gm, er)
        ge_pe = scipy_stats.pearsonr(gm, er)
        gate_erank_corr = {
            "spearman": float(0.0 if np.isnan(ge_sp.statistic) else ge_sp.statistic),
            "pearson": float(0.0 if np.isnan(ge_pe.statistic) else ge_pe.statistic),
        }
        logger.info(f"Gate vs erank: Sp={gate_erank_corr['spearman']:.4f} Pe={gate_erank_corr['pearson']:.4f}")
    else:
        gate_erank_corr = {"spearman": 0.0, "pearson": 0.0}
        logger.info("Gate vs erank: insufficient variation in data")

    if len(gm) > 2 and len(set(lc)) > 1:
        gc_sp = scipy_stats.spearmanr(gm, lc)
        gc_pe = scipy_stats.pearsonr(gm, lc)
        gate_card_corr = {
            "spearman": float(0.0 if np.isnan(gc_sp.statistic) else gc_sp.statistic),
            "pearson": float(0.0 if np.isnan(gc_pe.statistic) else gc_pe.statistic),
        }
        logger.info(f"Gate vs log(card): Sp={gate_card_corr['spearman']:.4f} Pe={gate_card_corr['pearson']:.4f}")
    else:
        gate_card_corr = {"spearman": 0.0, "pearson": 0.0}
        logger.info("Gate vs log(card): insufficient variation in data")

    # Plots
    _plot_gate_heatmap(per_cond_gate)
    _plot_gate_scatter(all_gate_data)

    return {
        "gate_heatmap": gate_heatmap,
        "gate_vs_erank_correlation": gate_erank_corr,
        "gate_vs_cardinality_correlation": gate_card_corr,
        "per_condition_gate_stats": per_cond_gate,
    }


def _plot_gate_heatmap(per_cond_gate: list):
    try:
        data = np.full((len(REGIMES), len(RANKS)), np.nan)
        for stat in per_cond_gate:
            parts = stat["condition"].split("_rank")
            if len(parts) == 2 and parts[0] in REGIMES and int(parts[1]) in RANKS:
                data[REGIMES.index(parts[0]), RANKS.index(int(parts[1]))] = stat["mean_gate"]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(data, annot=True, fmt=".4f", xticklabels=[f"rank{r}" for r in RANKS],
                    yticklabels=REGIMES, cmap="viridis", ax=ax)
        ax.set_title("RAMA Mean Gate Value by Condition")
        plt.tight_layout()
        plt.savefig(str(PLOT_DIR / "plot_gate_heatmap.png"), dpi=150)
        plt.close()
        logger.info("Saved plot_gate_heatmap.png")
    except Exception:
        logger.exception("Failed gate heatmap plot")


def _plot_gate_scatter(all_gate_data: list):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        gm = [gd["gate_mean"] for gd in all_gate_data]
        er = [gd["erank_svd"] for gd in all_gate_data]
        lc = [np.log(gd["cardinality"] + 1) for gd in all_gate_data]
        ax1.scatter(er, gm, alpha=0.3, s=5)
        ax1.set_xlabel("SVD Effective Rank")
        ax1.set_ylabel("Mean Gate Value")
        ax1.set_title("Gate vs Effective Rank")
        ax2.scatter(lc, gm, alpha=0.3, s=5)
        ax2.set_xlabel("log(Cardinality + 1)")
        ax2.set_ylabel("Mean Gate Value")
        ax2.set_title("Gate vs log(Cardinality)")
        plt.tight_layout()
        plt.savefig(str(PLOT_DIR / "plot_gate_vs_erank.png"), dpi=150)
        plt.close()
        logger.info("Saved plot_gate_vs_erank.png")
    except Exception:
        logger.exception("Failed gate scatter plot")


# ============================================================
# COLLAPSE-BENEFIT CORRELATION
# ============================================================
def compute_collapse_benefit(phase_a: dict, phase_b: dict) -> dict:
    """Correlate collapse severity with RAMA's benefit over Mean."""
    per_cond_stats = phase_a.get("per_condition_rank_stats", [])
    reg_summary = phase_b.get("results_summary", {}).get("regression", {})
    cohen_items = reg_summary.get("cohen_d_rama_vs_mean", {}).get("per_condition", [])

    collapse_map = {s["condition"]: s["mean_collapse_ratio"] for s in per_cond_stats}
    cohen_map = {d["condition"]: d["cohen_d"] for d in cohen_items}

    shared = sorted(set(collapse_map) & set(cohen_map))
    if len(shared) < 3:
        return {"spearman_rho": 0.0, "p_value": 1.0, "interpretation": "Insufficient data"}

    cr = [collapse_map[c] for c in shared]
    cd = [cohen_map[c] for c in shared]
    sp = scipy_stats.spearmanr(cr, cd)

    # Plot
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(cr, cd)
        for i, c in enumerate(shared):
            ax.annotate(c, (cr[i], cd[i]), fontsize=5, alpha=0.7)
        z = np.polyfit(cr, cd, 1)
        xr = np.linspace(min(cr), max(cr), 100)
        ax.plot(xr, np.poly1d(z)(xr), "r--", alpha=0.5)
        ax.set_xlabel("Mean Collapse Ratio")
        ax.set_ylabel("Cohen's d (RAMA vs Mean)")
        ax.set_title(f"Collapse vs RAMA Benefit (Spearman={sp.statistic:.3f}, p={sp.pvalue:.3f})")
        plt.tight_layout()
        plt.savefig(str(PLOT_DIR / "plot_collapse_vs_benefit.png"), dpi=150)
        plt.close()
        logger.info("Saved plot_collapse_vs_benefit.png")
    except Exception:
        logger.exception("Failed collapse-benefit plot")

    if sp.statistic < -0.3:
        interp = "Negative correlation: RAMA helps more where collapse ratio is lower (more severe collapse)"
    elif sp.statistic > 0.3:
        interp = "Positive correlation: RAMA helps more where collapse ratio is higher"
    else:
        interp = "Weak or no relationship between collapse severity and RAMA benefit"

    return {"spearman_rho": float(sp.statistic), "p_value": float(sp.pvalue), "interpretation": interp}


# ============================================================
# OUTPUT GENERATION
# ============================================================
def generate_output(
    data: dict, phase_a: dict, phase_b: dict, phase_c: dict, collapse_benefit: dict,
) -> dict:
    """Generate method_out.json conforming to exp_gen_sol_out schema."""
    logger.info("Generating output JSON...")

    examples = data["examples"]
    all_preds = phase_b.get("all_predictions", {})

    # Average predictions across seeds per method
    avg_preds = {}  # method -> {idx: {regression_pred, classification_prob, classification_pred}}
    for method in METHOD_NAMES:
        mp = {}
        for ex in examples:
            idx = ex["idx"]
            rp, cp = [], []
            for seed in range(NUM_SEEDS):
                rd = all_preds.get((method, "regression", seed), {})
                cd = all_preds.get((method, "classification", seed), {})
                if idx in rd:
                    rp.append(rd[idx])
                if idx in cd:
                    cp.append(cd[idx])
            if rp or cp:
                mp[idx] = {
                    "regression_pred": round(float(np.mean(rp)), 4) if rp else None,
                    "classification_prob": round(float(np.mean(cp)), 4) if cp else None,
                    "classification_pred": int(np.mean(cp) >= 0.5) if cp else None,
                }
        avg_preds[method] = mp

    # Build output examples
    output_examples = []
    for ex in examples:
        out_ex = {"input": ex["original_input"], "output": ex["original_output"]}
        # Copy original metadata
        for k, v in ex["original_metadata"].items():
            out_ex[k] = v
        # Add predictions
        for method in METHOD_NAMES:
            avg = avg_preds[method].get(ex["idx"])
            if avg is not None:
                out_ex[f"predict_{method}"] = json.dumps(avg)
        output_examples.append(out_ex)

    # Clean phase_b for serialization (remove non-serializable all_predictions)
    phase_b_meta = {
        "results_summary": phase_b.get("results_summary", {}),
        "training_time_seconds": phase_b.get("training_time_seconds", 0),
    }

    output = {
        "metadata": {
            "experiment": "RAMA Synthetic Testbed Validation",
            "description": (
                "Three-phase experiment: (A) rank collapse measurement and proxy validation, "
                "(B) 5 aggregation methods comparison (Mean, UngatedMeanVar, RAMA, PNA, VPA), "
                "(C) RAMA gate behavior analysis"
            ),
            "num_examples": len(examples),
            "num_conditions": len(set(ex["condition"] for ex in examples)),
            "num_seeds": NUM_SEEDS,
            "num_epochs_max": NUM_EPOCHS,
            "methods": METHOD_NAMES,
            "phase_a_rank_analysis": phase_a,
            "phase_b_model_comparison": phase_b_meta,
            "phase_c_gate_analysis": phase_c,
            "collapse_benefit_correlation": collapse_benefit,
        },
        "datasets": [{
            "dataset": "synthetic_relational_rank_collapse_testbed",
            "examples": output_examples,
        }],
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved {out_path} ({size_mb:.1f} MB)")
    return output


# ============================================================
# MAIN
# ============================================================
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("RAMA Synthetic Testbed Validation Experiment")
    logger.info("=" * 60)
    t_total = time.time()

    max_examples = int(os.environ.get("MAX_EXAMPLES", "0")) or None

    # Choose data file
    if max_examples and max_examples <= 3:
        data_path = DATA_DIR / "mini_data_out.json"
        max_examples = None
    else:
        data_path = DATA_DIR / "full_data_out.json"

    # Load data
    data = load_data(data_path, max_examples)

    # Phase A: Rank collapse analysis
    phase_a = phase_a_rank_analysis(data)

    # Phase B: Model comparison (returns results + saved RAMA states)
    phase_b, rama_states = phase_b_model_comparison(data)

    # Phase C: Gate analysis using saved RAMA states
    phase_c = phase_c_gate_analysis(data, rama_states)

    # Collapse-benefit correlation
    collapse_benefit = compute_collapse_benefit(phase_a, phase_b)

    # Generate output
    generate_output(data, phase_a, phase_b, phase_c, collapse_benefit)

    dt = time.time() - t_total
    logger.info(f"TOTAL TIME: {dt:.0f}s ({dt/60:.1f} min)")


if __name__ == "__main__":
    main()
