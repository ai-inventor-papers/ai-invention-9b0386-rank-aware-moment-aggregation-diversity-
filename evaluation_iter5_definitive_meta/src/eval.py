#!/usr/bin/env python3
"""Definitive Meta-Analysis of RAMA/CAMA Across 12 Experiments (Iterations 2-4).

Phases:
  A - Evidence Registry & Quality Grading (GREEN/YELLOW/RED per task x experiment)
  B - Per-Task Cohen's d with 10k Bootstrap CIs
  C - DerSimonian-Laird Random-Effects Meta-Analysis (3 pools)
  D - Critical Diagnostic Analyses (6 diagnostics)
  E - Publication Readiness Scorecard (7 dimensions, 0-3 each)
"""

import gc
import json
import math
import os
import resource
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# Setup
# ============================================================
WORKSPACE = Path(__file__).parent
(WORKSPACE / "logs").mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(WORKSPACE / "logs" / "run.log", rotation="30 MB", level="DEBUG")

# Container: 29 GB RAM, 4 CPUs
RAM_BUDGET = 24 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# Paths
BASE = Path(
    "/ai-inventor/aii_pipeline/runs/"
    "leskovec-predictive-residual-message-passing-v2_sti/3_invention_loop"
)
EXPERIMENT_PATHS = {
    "exp_id1_it2": BASE / "iter_2/gen_art/exp_id1_it2__opus",
    "exp_id2_it2": BASE / "iter_2/gen_art/exp_id2_it2__opus",
    "exp_id3_it2": BASE / "iter_2/gen_art/exp_id3_it2__opus",
    "exp_id4_it2": BASE / "iter_2/gen_art/exp_id4_it2__opus",
    "exp_id5_it2": BASE / "iter_2/gen_art/exp_id5_it2__opus",
    "exp_id2_it3": BASE / "iter_3/gen_art/exp_id2_it3__opus",
    "exp_id3_it3": BASE / "iter_3/gen_art/exp_id3_it3__opus",
    "exp_id4_it3": BASE / "iter_3/gen_art/exp_id4_it3__opus",
    "exp_id5_it3": BASE / "iter_3/gen_art/exp_id5_it3__opus",
    "exp_id1_it4": BASE / "iter_4/gen_art/exp_id1_it4__opus",
    "exp_id2_it4": BASE / "iter_4/gen_art/exp_id2_it4__opus",
    "exp_id3_it4": BASE / "iter_4/gen_art/exp_id3_it4__opus",
}

N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(42)


# ============================================================
# Data Structures
# ============================================================
@dataclass
class TaskResult:
    """Per-task per-experiment comparison result."""

    exp_id: str
    iteration: int
    dataset: str
    task: str
    task_type: str  # "regression" or "classification"
    metric_name: str
    higher_is_better: bool
    baseline_seeds: list
    method_seeds: list
    method_name: str  # "RAMA" or "CAMA"
    architecture: str  # "HeteroSAGE" or "RelGNN"
    epochs_planned: int
    epochs_actual: int
    eval_split: str  # "test", "val", or "val=test"
    val_test_identical: bool = False
    gate_stats: Optional[dict] = None
    grade: str = ""
    grade_flags: list = field(default_factory=list)
    # Computed in Phase B
    cohens_d: float = 0.0
    cohens_d_ci: tuple = (0.0, 0.0)
    p_ttest: float = 1.0
    p_wilcoxon: float = 1.0
    raw_effect: float = 0.0
    raw_effect_ci: tuple = (0.0, 0.0)
    se_d: float = 0.0

    @property
    def n_seeds(self) -> int:
        return min(len(self.baseline_seeds), len(self.method_seeds))

    @property
    def unique_task_key(self) -> str:
        return f"{self.dataset}/{self.task}"


# ============================================================
# Loading & Extraction
# ============================================================
def load_metadata(exp_id: str) -> dict:
    """Load metadata from experiment's full JSON file."""
    exp_dir = EXPERIMENT_PATHS[exp_id]
    full_path = exp_dir / "full_method_out.json"
    if not full_path.exists():
        logger.warning(f"Missing full_method_out.json for {exp_id}, trying preview")
        full_path = exp_dir / "preview_method_out.json"
    if not full_path.exists():
        logger.error(f"No JSON files found for {exp_id}")
        return {}
    logger.info(f"Loading metadata from {exp_id} ({full_path.name})")
    try:
        data = json.loads(full_path.read_text())
        meta = data.get("metadata", {})
        del data
        gc.collect()
        return meta
    except Exception:
        logger.exception(f"Failed to load {exp_id}")
        return {}


def _sort_seed_dict(d: dict) -> list:
    """Extract values from a seed-keyed dict, sorted by seed number."""
    return [d[k] for k in sorted(d.keys(), key=lambda x: int(x))]


# ----- Per-experiment extractors -----

def extract_exp_id1_it2(meta: dict) -> list[TaskResult]:
    """Synthetic testbed: RAMA vs mean (regression + classification)."""
    results = []
    phase_b = meta.get("phase_b_model_comparison", {})
    summary = phase_b.get("results_summary", {})

    for task_type_key, task_label, metric, hib in [
        ("regression", "synthetic-regression", "mse", False),
        ("classification", "synthetic-classification", "auroc", True),
    ]:
        section = summary.get(task_type_key, {})
        overall = section.get("overall", [])
        mean_data = next((m for m in overall if m["method"] == "mean"), None)
        rama_data = next((m for m in overall if m["method"] == "rama"), None)
        if not mean_data or not rama_data:
            continue
        # Synthetic has pooled d from cohen_d_rama_vs_mean
        cd_section = section.get("cohen_d_rama_vs_mean", {})
        # We don't have individual seed values for the pooled stats -
        # use per-condition data instead. For meta-analysis we need per-seed.
        # The synthetic experiment aggregates across conditions.
        # We'll represent it as a single entry using the pooled stats.
        results.append(TaskResult(
            exp_id="exp_id1_it2", iteration=2,
            dataset="synthetic", task=task_label,
            task_type="classification" if task_type_key == "classification" else "regression",
            metric_name=metric, higher_is_better=hib,
            baseline_seeds=[mean_data.get(f"mean_{'r2' if metric == 'mse' else 'auroc'}", 0.0)],
            method_seeds=[rama_data.get(f"mean_{'r2' if metric == 'mse' else 'auroc'}", 0.0)],
            method_name="RAMA", architecture="Synthetic-MLP",
            epochs_planned=100, epochs_actual=100,
            eval_split="test",
            grade="YELLOW",
            grade_flags=["synthetic_data", "no_individual_seeds_for_pooled"],
        ))
    return results


def extract_exp_id2_it2(meta: dict) -> list[TaskResult]:
    """rel-f1 dual-task: RAMA vs mean."""
    results = []
    tasks_data = meta.get("tasks", {})
    for task_name, td in tasks_data.items():
        baseline = td.get("mean_baseline", {}).get("per_seed", [])
        method = td.get("rama", {}).get("per_seed", [])
        if not baseline or not method:
            continue
        task_type_raw = td.get("task_type", "")
        is_cls = "classification" in task_type_raw
        metric = td.get("metric", "mae")
        hib = metric in ("average_precision", "roc_auc", "accuracy", "f1")
        results.append(TaskResult(
            exp_id="exp_id2_it2", iteration=2,
            dataset="rel-f1", task=task_name,
            task_type="classification" if is_cls else "regression",
            metric_name=metric, higher_is_better=hib,
            baseline_seeds=baseline, method_seeds=method,
            method_name="RAMA", architecture="HeteroSAGE",
            epochs_planned=10, epochs_actual=10,
            eval_split="test",
        ))
    return results


def extract_exp_id3_it2(meta: dict) -> list[TaskResult]:
    """rel-stack: RAMA vs mean on post-votes and user-engagement."""
    results = []
    analysis = meta.get("analysis", {})
    tasks_data = analysis.get("tasks", {})
    gate_analysis = meta.get("gate_analysis", {})

    for task_name, td in tasks_data.items():
        metric = td.get("metric", "mae")
        hib = metric in ("roc_auc", "average_precision")
        task_type_raw = td.get("task_type", "")
        is_cls = "classification" in task_type_raw

        # Try all_metrics first (has all 5 seeds as dict keys)
        all_m = td.get("all_metrics", {})
        if all_m:
            baseline_seeds = []
            method_seeds = []
            for seed_key in sorted(all_m.keys(), key=lambda x: int(x)):
                sd = all_m[seed_key]
                baseline_seeds.append(sd.get(f"mean_{metric}", 0.0))
                method_seeds.append(sd.get(f"rama_{metric}", 0.0))
        else:
            baseline_seeds = td.get("baseline_mean_results", {}).get("seeds", [])
            method_seeds = td.get("rama_results", {}).get("seeds", [])

        results.append(TaskResult(
            exp_id="exp_id3_it2", iteration=2,
            dataset="rel-stack", task=task_name,
            task_type="classification" if is_cls else "regression",
            metric_name=metric, higher_is_better=hib,
            baseline_seeds=baseline_seeds, method_seeds=method_seeds,
            method_name="RAMA", architecture="HeteroSAGE",
            epochs_planned=10, epochs_actual=5,
            eval_split="test",
            gate_stats=gate_analysis.get("per_edge_type"),
        ))
    return results


def extract_exp_id4_it2(meta: dict) -> list[TaskResult]:
    """rel-amazon item-ltv: RAMA vs mean (val-only, no test targets)."""
    sa = meta.get("statistical_analysis", {})
    baseline = sa.get("baseline_mean_maes", [])
    method = sa.get("rama_maes", [])
    if not baseline or not method:
        return []
    gate_analysis = meta.get("gate_analysis", {})
    return [TaskResult(
        exp_id="exp_id4_it2", iteration=2,
        dataset="rel-amazon", task="item-ltv",
        task_type="regression", metric_name="mae", higher_is_better=False,
        baseline_seeds=baseline, method_seeds=method,
        method_name="RAMA", architecture="HeteroSAGE",
        epochs_planned=10, epochs_actual=10,
        eval_split="val",
        gate_stats=gate_analysis,
    )]


def extract_exp_id5_it2(meta: dict) -> list[TaskResult]:
    """rel-f1 ablation: rama_full vs standard_mean on driver-position."""
    sa = meta.get("statistical_analysis", {})
    pmr = sa.get("per_method_results", {})
    baseline_data = pmr.get("standard_mean", {})
    method_data = pmr.get("rama_full", {})
    baseline = baseline_data.get("test_mae_per_seed", [])
    method = method_data.get("test_mae_per_seed", [])
    if not baseline or not method:
        return []
    return [TaskResult(
        exp_id="exp_id5_it2", iteration=2,
        dataset="rel-f1", task="driver-position",
        task_type="regression", metric_name="mae", higher_is_better=False,
        baseline_seeds=baseline, method_seeds=method,
        method_name="RAMA", architecture="HeteroSAGE",
        epochs_planned=10, epochs_actual=10,
        eval_split="test",
    )]


def extract_exp_id2_it3(meta: dict) -> list[TaskResult]:
    """rel-avito ad-ctr stress test: RAMA vs mean."""
    res = meta.get("results", {})
    bl_per = res.get("baseline_mean", {}).get("per_seed", {})
    rm_per = res.get("rama", {}).get("per_seed", {})
    if not bl_per or not rm_per:
        return []
    baseline = [bl_per[k]["mae"] for k in sorted(bl_per.keys(), key=lambda x: int(x))]
    method = [rm_per[k]["mae"] for k in sorted(rm_per.keys(), key=lambda x: int(x))]
    gate_analysis = meta.get("gate_analysis", {}).get("per_edge_type", {})
    return [TaskResult(
        exp_id="exp_id2_it3", iteration=3,
        dataset="rel-avito", task="ad-ctr",
        task_type="regression", metric_name="mae", higher_is_better=False,
        baseline_seeds=baseline, method_seeds=method,
        method_name="RAMA", architecture="HeteroSAGE",
        epochs_planned=10, epochs_actual=10,
        eval_split="val",
        gate_stats=gate_analysis,
    )]


def extract_exp_id3_it3(meta: dict) -> list[TaskResult]:
    """RelGNN integration: RAMA vs sum on rel-f1 and rel-amazon."""
    results = []
    sa = meta.get("statistical_analysis", {})
    gate_analysis = meta.get("gate_analysis", {})

    task_map = {
        "rel-f1/driver-position": ("rel-f1", "driver-position", "regression", "mae", False),
        "rel-amazon/item-ltv": ("rel-amazon", "item-ltv", "regression", "mae", False),
    }
    for task_key, (dataset, task, ttype, metric, hib) in task_map.items():
        td = sa.get(task_key, {})
        baseline = td.get("baseline_maes", [])
        method = td.get("rama_maes", [])
        if not baseline or not method:
            continue
        results.append(TaskResult(
            exp_id="exp_id3_it3", iteration=3,
            dataset=dataset, task=task,
            task_type=ttype, metric_name=metric, higher_is_better=hib,
            baseline_seeds=baseline, method_seeds=method,
            method_name="RAMA", architecture="RelGNN",
            epochs_planned=10, epochs_actual=10,
            eval_split="test",
            gate_stats=gate_analysis.get(task_key),
        ))
    return results


def extract_exp_id4_it3(meta: dict) -> list[TaskResult]:
    """Controlled RAMA vs mean on rel-f1 (dual-task)."""
    results = []
    sa = meta.get("statistical_analysis", {})
    per_task = sa.get("per_task", {})
    gate_analysis = meta.get("gate_analysis", {})

    for task_key, td in per_task.items():
        raw = td.get("raw_values", {})
        baseline = raw.get("baseline", [])
        method = raw.get("rama_full", [])
        if not baseline or not method:
            continue
        metric_key = td.get("metric_key", "mae")
        hib = td.get("higher_is_better", False)
        parts = task_key.split("/")
        dataset = parts[0] if len(parts) > 1 else "rel-f1"
        task = parts[1] if len(parts) > 1 else task_key

        results.append(TaskResult(
            exp_id="exp_id4_it3", iteration=3,
            dataset=dataset, task=task,
            task_type="classification" if hib else "regression",
            metric_name=metric_key, higher_is_better=hib,
            baseline_seeds=baseline, method_seeds=method,
            method_name="RAMA", architecture="HeteroSAGE",
            epochs_planned=10, epochs_actual=10,
            eval_split="test",
            gate_stats=gate_analysis,
        ))
    return results


def extract_exp_id5_it3(meta: dict) -> list[TaskResult]:
    """rel-trial: RAMA vs mean on study-outcome and study-adverse."""
    results = []
    sa = meta.get("statistical_analysis", {})
    per_task = sa.get("per_task_results", {})

    for task_name, td in per_task.items():
        baseline = td.get("per_seed_baseline", [])
        method = td.get("per_seed_rama", [])
        if not baseline or not method:
            continue
        metric = td.get("metric", "mae")
        direction = td.get("direction", "lower_better")
        hib = "higher" in direction
        task_type_raw = td.get("task_type", "")
        is_cls = "classification" in task_type_raw

        results.append(TaskResult(
            exp_id="exp_id5_it3", iteration=3,
            dataset="rel-trial", task=task_name,
            task_type="classification" if is_cls else "regression",
            metric_name=metric, higher_is_better=hib,
            baseline_seeds=baseline, method_seeds=method,
            method_name="RAMA", architecture="HeteroSAGE",
            epochs_planned=20, epochs_actual=20,
            eval_split="test",
        ))
    return results


def extract_exp_id1_it4(meta: dict) -> list[TaskResult]:
    """CAMA vs mean on rel-f1 (driver-position + driver-dnf). val=test anomaly."""
    results = []
    summary = meta.get("summary_table", {})
    per_run = meta.get("per_run_results", [])

    # Check val=test anomaly from per_run_results
    val_test_same = False
    for run in per_run:
        vm = run.get("val_metrics", {})
        tm = run.get("test_metrics", {})
        if vm and tm and vm == tm:
            val_test_same = True
            break

    task_configs = {
        "driver-position": ("mae", False, "regression"),
        "driver-dnf": ("average_precision" if "test_ap_mean" in summary.get("driver-dnf", {}).get("mean", {}) else "average_precision", True, "classification"),
    }

    for task_name, (_, _, _) in task_configs.items():
        td = summary.get(task_name, {})
        bl_data = td.get("mean", {})
        method_data = td.get("mean_cama", {})

        baseline = bl_data.get("per_seed", [])
        method = method_data.get("per_seed", [])
        if not baseline or not method:
            continue

        # Determine metric from key names
        if "test_mae_mean" in bl_data:
            metric = "mae"
            hib = False
            ttype = "regression"
        else:
            metric = "average_precision"
            hib = True
            ttype = "classification"

        results.append(TaskResult(
            exp_id="exp_id1_it4", iteration=4,
            dataset="rel-f1", task=task_name,
            task_type=ttype, metric_name=metric, higher_is_better=hib,
            baseline_seeds=baseline, method_seeds=method,
            method_name="CAMA", architecture="HeteroSAGE",
            epochs_planned=10, epochs_actual=10,
            eval_split="val=test",
            val_test_identical=val_test_same,
        ))
    return results


def extract_exp_id2_it4(meta: dict) -> list[TaskResult]:
    """CAMA on rel-trial + rel-stack (3 tasks, 5 seeds)."""
    results = []
    tla = meta.get("task_level_analysis", {})
    gate_analysis = meta.get("gate_analysis", {})

    for task_name, td in tla.items():
        baseline = td.get("baseline_per_seed", [])
        method = td.get("cama_per_seed", [])
        if not baseline or not method:
            continue
        metric = td.get("metric", "mae")
        hib = td.get("higher_is_better", metric in ("roc_auc", "average_precision"))
        task_type_raw = td.get("task_type", "") if "task_type" in td else ""

        # Infer dataset and task_type
        if "study" in task_name:
            dataset = "rel-trial"
        elif "engagement" in task_name:
            dataset = "rel-stack"
        else:
            dataset = "unknown"

        if metric in ("roc_auc", "average_precision"):
            ttype = "classification"
        else:
            ttype = "regression"

        results.append(TaskResult(
            exp_id="exp_id2_it4", iteration=4,
            dataset=dataset, task=task_name,
            task_type=ttype, metric_name=metric, higher_is_better=hib,
            baseline_seeds=baseline, method_seeds=method,
            method_name="CAMA", architecture="HeteroSAGE",
            epochs_planned=20, epochs_actual=20,
            eval_split="test",
            gate_stats=gate_analysis.get(task_name),
        ))
    return results


def extract_exp_id3_it4(meta: dict) -> list[TaskResult]:
    """CAMA on Amazon + Avito (with negative gate init variant)."""
    results_list = []
    res = meta.get("results", {})

    # Part A: Amazon item-ltv
    part_a = res.get("part_a_amazon", {})
    if part_a:
        pm = part_a.get("per_method", {})
        for bl_key, method_key, method_label in [
            ("mean_baseline", "cama_default", "CAMA"),
        ]:
            bl_data = pm.get(bl_key, {}).get("per_seed", {})
            m_data = pm.get(method_key, {}).get("per_seed", {})
            if not bl_data or not m_data:
                continue
            seeds_sorted = sorted(bl_data.keys(), key=lambda x: int(x))
            baseline = [bl_data[s]["val_mae"] for s in seeds_sorted if bl_data[s].get("val_mae") is not None]
            method = [m_data[s]["val_mae"] for s in sorted(m_data.keys(), key=lambda x: int(x))
                       if m_data[s].get("val_mae") is not None]
            if baseline and method:
                results_list.append(TaskResult(
                    exp_id="exp_id3_it4", iteration=4,
                    dataset="rel-amazon", task="item-ltv",
                    task_type="regression", metric_name="mae", higher_is_better=False,
                    baseline_seeds=baseline, method_seeds=method,
                    method_name=method_label, architecture="HeteroSAGE",
                    epochs_planned=10, epochs_actual=5,
                    eval_split="val",
                ))

    # Part B: Avito ad-ctr (default and neg_init)
    part_b = res.get("part_b_avito", {})
    if part_b:
        pm = part_b.get("per_method", {})
        bl_data = pm.get("mean_baseline", {}).get("per_seed", {})
        for method_key, method_label in [
            ("cama_default", "CAMA"),
            ("cama_neg_init", "CAMA-neg-init"),
        ]:
            m_data = pm.get(method_key, {}).get("per_seed", {})
            if not bl_data or not m_data:
                continue
            seeds_sorted = sorted(bl_data.keys(), key=lambda x: int(x))
            baseline = [bl_data[s]["val_mae"] for s in seeds_sorted if bl_data[s].get("val_mae") is not None]
            m_seeds_sorted = sorted(m_data.keys(), key=lambda x: int(x))
            method = [m_data[s]["val_mae"] for s in m_seeds_sorted if m_data[s].get("val_mae") is not None]
            if baseline and method:
                task_suffix = "ad-ctr" if method_label == "CAMA" else "ad-ctr-neg-init"
                results_list.append(TaskResult(
                    exp_id="exp_id3_it4", iteration=4,
                    dataset="rel-avito", task=task_suffix,
                    task_type="regression", metric_name="mae", higher_is_better=False,
                    baseline_seeds=baseline, method_seeds=method,
                    method_name=method_label, architecture="HeteroSAGE",
                    epochs_planned=10, epochs_actual=5,
                    eval_split="val",
                ))
    return results_list


EXTRACTORS = {
    "exp_id1_it2": extract_exp_id1_it2,
    "exp_id2_it2": extract_exp_id2_it2,
    "exp_id3_it2": extract_exp_id3_it2,
    "exp_id4_it2": extract_exp_id4_it2,
    "exp_id5_it2": extract_exp_id5_it2,
    "exp_id2_it3": extract_exp_id2_it3,
    "exp_id3_it3": extract_exp_id3_it3,
    "exp_id4_it3": extract_exp_id4_it3,
    "exp_id5_it3": extract_exp_id5_it3,
    "exp_id1_it4": extract_exp_id1_it4,
    "exp_id2_it4": extract_exp_id2_it4,
    "exp_id3_it4": extract_exp_id3_it4,
}


def extract_all_task_results() -> list[TaskResult]:
    """Load all experiments and extract unified TaskResult objects."""
    all_results = []
    for exp_id in EXPERIMENT_PATHS:
        try:
            meta = load_metadata(exp_id)
            if not meta:
                logger.warning(f"Empty metadata for {exp_id}")
                continue
            extractor = EXTRACTORS.get(exp_id)
            if not extractor:
                logger.warning(f"No extractor for {exp_id}")
                continue
            results = extractor(meta)
            logger.info(f"  {exp_id}: extracted {len(results)} task comparisons")
            all_results.extend(results)
        except Exception:
            logger.exception(f"Failed to extract {exp_id}")
    logger.info(f"Total task comparisons extracted: {len(all_results)}")
    return all_results


# ============================================================
# Phase A: Evidence Registry & Quality Grading
# ============================================================
def grade_evidence(results: list[TaskResult]) -> list[TaskResult]:
    """Assign GREEN/YELLOW/RED grades to each task x experiment."""
    for r in results:
        flags = list(r.grade_flags)  # keep any pre-set flags

        # 1. val_test_identical
        if r.val_test_identical:
            flags.append("val_test_identical")

        # 2. degenerate_baseline: CV < threshold (stricter for regression)
        bl = np.array(r.baseline_seeds, dtype=float)
        bl_mean = np.mean(bl) if len(bl) > 0 else 0.0
        bl_std = np.std(bl, ddof=1) if len(bl) > 1 else 0.0
        cv_threshold = 0.001  # very low variance relative to mean
        if len(bl) > 1 and abs(bl_mean) > 1e-12 and bl_std / abs(bl_mean) < cv_threshold:
            flags.append("degenerate_baseline")
        if len(bl) > 1 and bl_std == 0.0:
            flags.append("zero_variance_baseline")

        # 3. seed_count
        if r.n_seeds < 5:
            flags.append(f"only_{r.n_seeds}_seeds")
        if r.n_seeds < 3:
            flags.append("fewer_than_3_seeds")

        # 4. epoch_deviation
        if r.epochs_actual < r.epochs_planned:
            flags.append(f"epoch_deviation_{r.epochs_actual}_vs_{r.epochs_planned}")

        # 5. gate_stasis: check if gate_stats available and all gates near 0.5
        if r.gate_stats and isinstance(r.gate_stats, dict):
            stasis = True
            n_edges_checked = 0
            for edge_key, edge_data in r.gate_stats.items():
                if isinstance(edge_data, dict):
                    gate_val = edge_data.get("mean_gate", edge_data.get("gate_bias_sigmoid_mean"))
                    if gate_val is not None and not math.isnan(gate_val):
                        n_edges_checked += 1
                        if abs(gate_val - 0.5) > 0.02:
                            stasis = False
                            break
            if stasis and n_edges_checked > 0:
                flags.append("gate_stasis")

        # 6. val-only evaluation
        if r.eval_split == "val":
            flags.append("val_only_evaluation")

        # Assign grade
        critical_flags = {"val_test_identical", "degenerate_baseline",
                          "zero_variance_baseline", "fewer_than_3_seeds", "synthetic_data"}
        minor_flags = {"val_only_evaluation", "gate_stasis",
                       "no_individual_seeds_for_pooled"}

        has_critical = bool(critical_flags & set(flags))
        has_minor = bool(set(f for f in flags if f.startswith("epoch_deviation") or f.startswith("only_")) | (minor_flags & set(flags)))

        if has_critical:
            r.grade = "RED"
        elif has_minor:
            r.grade = "YELLOW"
        else:
            r.grade = "GREEN"

        r.grade_flags = flags

    # Log summary
    grades = [r.grade for r in results]
    logger.info(f"Phase A grades (initial): GREEN={grades.count('GREEN')}, "
                f"YELLOW={grades.count('YELLOW')}, RED={grades.count('RED')}")
    return results


def regrade_after_effect_sizes(results: list[TaskResult]) -> list[TaskResult]:
    """Re-grade after Phase B to catch inflated effect sizes (|d| > 5)."""
    for r in results:
        if abs(r.cohens_d) > 5.0 and "inflated_d" not in r.grade_flags:
            r.grade_flags.append(f"inflated_d_{abs(r.cohens_d):.1f}")
            # Downgrade to at least YELLOW
            if r.grade == "GREEN":
                r.grade = "YELLOW"
            # If d > 20, downgrade to RED (clearly degenerate)
            if abs(r.cohens_d) > 20:
                r.grade = "RED"

    grades = [r.grade for r in results]
    logger.info(f"Phase A grades (after re-grade): GREEN={grades.count('GREEN')}, "
                f"YELLOW={grades.count('YELLOW')}, RED={grades.count('RED')}")
    return results


# ============================================================
# Phase B: Per-Task Cohen's d with Bootstrap CIs
# ============================================================
def compute_cohens_d_signed(baseline: np.ndarray, method: np.ndarray,
                            higher_is_better: bool) -> float:
    """Compute Cohen's d with sign such that positive = method is better."""
    n = len(baseline)
    if n < 2:
        return 0.0
    diff = method - baseline
    if higher_is_better:
        # positive diff = method is better
        pass
    else:
        # negative diff = method is better (lower is better)
        diff = -diff  # flip so positive = method is better

    pooled_sd = np.sqrt((np.var(baseline, ddof=1) + np.var(method, ddof=1)) / 2.0)
    if pooled_sd < 1e-15:
        return 0.0
    return float(np.mean(diff) / pooled_sd)


def bootstrap_cohens_d_ci(baseline: np.ndarray, method: np.ndarray,
                          higher_is_better: bool, n_boot: int = N_BOOTSTRAP) -> tuple:
    """Bootstrap 95% CI for Cohen's d."""
    n = len(baseline)
    if n < 2:
        return (0.0, 0.0)
    boot_ds = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boot_ds[i] = compute_cohens_d_signed(baseline[idx], method[idx], higher_is_better)
    lo = float(np.nanpercentile(boot_ds, 2.5))
    hi = float(np.nanpercentile(boot_ds, 97.5))
    return (lo, hi)


def compute_all_effect_sizes(results: list[TaskResult]) -> list[TaskResult]:
    """Compute Cohen's d, bootstrap CI, p-values for each task result."""
    logger.info("Phase B: Computing effect sizes for all task comparisons...")
    for r in results:
        bl = np.array(r.baseline_seeds, dtype=float)
        mt = np.array(r.method_seeds, dtype=float)
        n = min(len(bl), len(mt))
        if n < 2:
            r.cohens_d = 0.0
            r.cohens_d_ci = (0.0, 0.0)
            r.p_ttest = 1.0
            r.p_wilcoxon = 1.0
            r.raw_effect = 0.0
            r.raw_effect_ci = (0.0, 0.0)
            r.se_d = 0.0
            continue

        bl = bl[:n]
        mt = mt[:n]

        # Cohen's d (positive = method is better)
        r.cohens_d = compute_cohens_d_signed(bl, mt, r.higher_is_better)

        # Bootstrap CI
        r.cohens_d_ci = bootstrap_cohens_d_ci(bl, mt, r.higher_is_better)

        # SE of d (approximation)
        r.se_d = math.sqrt(2.0 / n + r.cohens_d ** 2 / (2.0 * n))

        # Paired t-test
        try:
            _, p_two = stats.ttest_rel(bl, mt)
            r.p_ttest = float(p_two) if not (math.isnan(p_two) or math.isinf(p_two)) else 1.0
        except Exception:
            r.p_ttest = 1.0

        # Wilcoxon signed-rank test
        try:
            diff_raw = mt - bl if r.higher_is_better else bl - mt
            if np.all(diff_raw == 0):
                r.p_wilcoxon = 1.0
            else:
                _, p_w = stats.wilcoxon(diff_raw, alternative="two-sided")
                r.p_wilcoxon = float(p_w)
        except Exception:
            r.p_wilcoxon = 1.0

        # Raw effect (in metric units)
        if r.higher_is_better:
            raw_diff = mt - bl
        else:
            raw_diff = bl - mt
        r.raw_effect = float(np.mean(raw_diff))
        se_raw = float(np.std(raw_diff, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        r.raw_effect_ci = (
            r.raw_effect - 1.96 * se_raw,
            r.raw_effect + 1.96 * se_raw,
        )

        logger.debug(
            f"  {r.exp_id}/{r.task}: d={r.cohens_d:.3f} "
            f"CI=[{r.cohens_d_ci[0]:.3f}, {r.cohens_d_ci[1]:.3f}] "
            f"p_t={r.p_ttest:.4f} grade={r.grade}"
        )
    logger.info("Phase B complete.")
    return results


# ============================================================
# Phase C: DerSimonian-Laird Random-Effects Meta-Analysis
# ============================================================
def dersimonian_laird(ds: list[float], ses: list[float]) -> dict:
    """DerSimonian-Laird random-effects meta-analysis.

    Args:
        ds: list of effect sizes (Cohen's d)
        ses: list of standard errors

    Returns:
        dict with pooled_d, ci, tau2, I2, Q, Q_p, prediction_interval
    """
    k = len(ds)
    if k == 0:
        return {
            "pooled_d": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
            "tau2": 0.0, "I2": 0.0, "Q": 0.0, "Q_p": 1.0,
            "pred_lo": 0.0, "pred_hi": 0.0, "p_value": 1.0, "k": 0,
        }
    if k == 1:
        d, se = ds[0], ses[0]
        return {
            "pooled_d": d, "se": se,
            "ci_lo": d - 1.96 * se, "ci_hi": d + 1.96 * se,
            "tau2": 0.0, "I2": 0.0, "Q": 0.0, "Q_p": 1.0,
            "pred_lo": d - 1.96 * se, "pred_hi": d + 1.96 * se,
            "p_value": 2 * (1 - stats.norm.cdf(abs(d / se))) if se > 0 else 1.0,
            "k": 1,
        }

    d_arr = np.array(ds, dtype=float)
    se_arr = np.array(ses, dtype=float)
    var_arr = se_arr ** 2

    # Fixed-effect weights
    w = 1.0 / var_arr
    sum_w = np.sum(w)
    d_fe = np.sum(w * d_arr) / sum_w

    # Cochran's Q
    Q = float(np.sum(w * (d_arr - d_fe) ** 2))
    Q_p = float(1 - stats.chi2.cdf(Q, k - 1)) if k > 1 else 1.0

    # Between-study variance (tau^2)
    C = sum_w - np.sum(w ** 2) / sum_w
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0

    # Random-effects weights
    w_star = 1.0 / (var_arr + tau2)
    sum_w_star = np.sum(w_star)
    d_re = float(np.sum(w_star * d_arr) / sum_w_star)
    se_re = float(1.0 / np.sqrt(sum_w_star))

    # I^2
    I2 = max(0.0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0.0

    # p-value
    z = d_re / se_re if se_re > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Prediction interval (accounts for tau^2)
    if k >= 3:
        t_crit = stats.t.ppf(0.975, k - 2)
        pred_se = math.sqrt(se_re ** 2 + tau2)
        pred_lo = d_re - t_crit * pred_se
        pred_hi = d_re + t_crit * pred_se
    else:
        pred_lo = d_re - 1.96 * se_re
        pred_hi = d_re + 1.96 * se_re

    return {
        "pooled_d": d_re, "se": se_re,
        "ci_lo": d_re - 1.96 * se_re, "ci_hi": d_re + 1.96 * se_re,
        "tau2": float(tau2), "I2": float(I2),
        "Q": Q, "Q_p": Q_p,
        "pred_lo": float(pred_lo), "pred_hi": float(pred_hi),
        "p_value": p_value, "k": k,
    }


def run_meta_analyses(results: list[TaskResult]) -> dict:
    """Run 3 pooled meta-analyses + classification subgroup."""
    # Filter out synthetic
    real = [r for r in results if r.dataset != "synthetic"]

    green = [r for r in real if r.grade == "GREEN"]
    green_yellow = [r for r in real if r.grade in ("GREEN", "YELLOW")]
    classification = [r for r in green_yellow if r.task_type == "classification"]
    regression = [r for r in green_yellow if r.task_type == "regression"]

    def pool(subset, label):
        ds = [r.cohens_d for r in subset]
        ses = [r.se_d for r in subset]
        result = dersimonian_laird(ds, ses)
        logger.info(
            f"  {label}: k={result['k']}, pooled_d={result['pooled_d']:.3f}, "
            f"CI=[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}], "
            f"p={result['p_value']:.4f}, I2={result['I2']:.1f}%"
        )
        return result

    logger.info("Phase C: Running meta-analyses...")
    meta_results = {
        "green_only": pool(green, "GREEN-only"),
        "green_yellow": pool(green_yellow, "GREEN+YELLOW"),
        "classification_only": pool(classification, "Classification-only"),
        "regression_only": pool(regression, "Regression-only"),
    }
    logger.info("Phase C complete.")
    return meta_results


# ============================================================
# Phase D: Critical Diagnostic Analyses
# ============================================================
def run_all_diagnostics(results: list[TaskResult]) -> dict:
    """Run all 6 diagnostic analyses."""
    logger.info("Phase D: Running diagnostic analyses...")
    diagnostics = {}

    # D1: Val-Test Inflation Factor
    diagnostics["val_test_inflation"] = _diag_val_test_inflation(results)

    # D2: Classification vs Regression Subgroup
    diagnostics["class_vs_reg"] = _diag_class_vs_reg(results)

    # D3: Amazon Stability
    diagnostics["amazon_stability"] = _diag_amazon_stability(results)

    # D4: Gate Learning
    diagnostics["gate_learning"] = _diag_gate_learning(results)

    # D5: Degenerate Baselines
    diagnostics["degenerate_baselines"] = _diag_degenerate_baselines(results)

    # D6: Avito Safety
    diagnostics["avito_safety"] = _diag_avito_safety(results)

    logger.info("Phase D complete.")
    return diagnostics


def _diag_val_test_inflation(results: list[TaskResult]) -> dict:
    """D1: Compare val-only vs test-evaluated effect sizes for same tasks."""
    # Find tasks that appear with both val and test evaluation
    task_groups: dict[str, dict] = {}
    for r in results:
        key = r.unique_task_key
        if key not in task_groups:
            task_groups[key] = {"val": [], "test": []}
        if r.eval_split in ("val", "val=test"):
            task_groups[key]["val"].append(r)
        else:
            task_groups[key]["test"].append(r)

    inflation_entries = []
    for task_key, groups in task_groups.items():
        if groups["val"] and groups["test"]:
            for vr in groups["val"]:
                for tr in groups["test"]:
                    if tr.cohens_d != 0:
                        ratio = vr.cohens_d / tr.cohens_d
                    else:
                        ratio = float("inf") if vr.cohens_d != 0 else 1.0
                    inflation_entries.append({
                        "task": task_key,
                        "val_exp": vr.exp_id,
                        "test_exp": tr.exp_id,
                        "d_val": vr.cohens_d,
                        "d_test": tr.cohens_d,
                        "inflation_ratio": ratio,
                    })

    mean_inflation = (
        np.mean([e["inflation_ratio"] for e in inflation_entries
                 if abs(e["inflation_ratio"]) < 100])
        if inflation_entries else float("nan")
    )
    return {
        "entries": inflation_entries,
        "mean_inflation_ratio": float(mean_inflation) if not math.isnan(mean_inflation) else 0.0,
        "n_comparisons": len(inflation_entries),
    }


def _diag_class_vs_reg(results: list[TaskResult]) -> dict:
    """D2: Compare effect sizes between classification and regression tasks."""
    real = [r for r in results if r.dataset != "synthetic" and r.grade != "RED"]
    cls_ds = [r.cohens_d for r in real if r.task_type == "classification"]
    reg_ds = [r.cohens_d for r in real if r.task_type == "regression"]

    cls_mean = float(np.mean(cls_ds)) if cls_ds else 0.0
    reg_mean = float(np.mean(reg_ds)) if reg_ds else 0.0

    # Q_between from meta-regression (simplified)
    if cls_ds and reg_ds:
        try:
            t_stat, p_val = stats.ttest_ind(cls_ds, reg_ds, equal_var=False)
            p_subgroup = float(p_val)
        except Exception:
            p_subgroup = 1.0
    else:
        p_subgroup = 1.0

    return {
        "classification_n": len(cls_ds),
        "classification_mean_d": cls_mean,
        "regression_n": len(reg_ds),
        "regression_mean_d": reg_mean,
        "p_subgroup_difference": p_subgroup,
    }


def _diag_amazon_stability(results: list[TaskResult]) -> dict:
    """D3: Compare Amazon item-ltv across iterations."""
    amazon = [r for r in results if r.task == "item-ltv" and r.dataset == "rel-amazon"]
    entries = []
    for r in amazon:
        bl_mean = float(np.mean(r.baseline_seeds))
        mt_mean = float(np.mean(r.method_seeds))
        entries.append({
            "exp_id": r.exp_id,
            "iteration": r.iteration,
            "architecture": r.architecture,
            "epochs": r.epochs_actual,
            "baseline_mean_mae": bl_mean,
            "method_mean_mae": mt_mean,
            "cohens_d": r.cohens_d,
            "eval_split": r.eval_split,
            "n_seeds": r.n_seeds,
            "baseline_std": float(np.std(r.baseline_seeds, ddof=1)) if r.n_seeds > 1 else 0.0,
            "method_std": float(np.std(r.method_seeds, ddof=1)) if r.n_seeds > 1 else 0.0,
        })

    d_values = [e["cohens_d"] for e in entries]
    return {
        "experiments": entries,
        "d_range": (min(d_values) if d_values else 0.0, max(d_values) if d_values else 0.0),
        "d_sign_consistent": all(d > 0 for d in d_values) or all(d < 0 for d in d_values) if d_values else False,
        "n_experiments": len(entries),
    }


def _diag_gate_learning(results: list[TaskResult]) -> dict:
    """D4: Analyze gate learning across experiments with gate_stats."""
    gate_entries = []
    for r in results:
        if not r.gate_stats or not isinstance(r.gate_stats, dict):
            continue
        max_deviation = 0.0
        n_edges = 0
        edge_details = []
        for edge_key, edge_data in r.gate_stats.items():
            if not isinstance(edge_data, dict):
                continue
            gate_val = edge_data.get("mean_gate", edge_data.get("gate_bias_sigmoid_mean"))
            if gate_val is None or (isinstance(gate_val, float) and math.isnan(gate_val)):
                continue
            dev = abs(gate_val - 0.5)
            max_deviation = max(max_deviation, dev)
            n_edges += 1
            edge_details.append({
                "edge": edge_key[:80],
                "mean_gate": gate_val,
                "deviation_from_half": dev,
            })

        gate_entries.append({
            "exp_id": r.exp_id,
            "task": r.unique_task_key,
            "max_deviation": max_deviation,
            "n_edges": n_edges,
            "stasis": max_deviation <= 0.02,
            "task_d": r.cohens_d,
        })

    # Correlation between gate deviation and effect size
    deviations = [e["max_deviation"] for e in gate_entries if not e["stasis"]]
    d_vals = [e["task_d"] for e in gate_entries if not e["stasis"]]
    if len(deviations) >= 3:
        corr, p_corr = stats.spearmanr(deviations, d_vals)
    else:
        corr, p_corr = 0.0, 1.0

    stasis_count = sum(1 for e in gate_entries if e["stasis"])
    return {
        "entries": gate_entries,
        "n_with_gate_stats": len(gate_entries),
        "n_stasis": stasis_count,
        "correlation_deviation_vs_d": float(corr) if not math.isnan(corr) else 0.0,
        "correlation_p": float(p_corr) if not math.isnan(p_corr) else 1.0,
    }


def _diag_degenerate_baselines(results: list[TaskResult]) -> dict:
    """D5: Identify degenerate baselines (near-zero variance)."""
    entries = []
    for r in results:
        bl = np.array(r.baseline_seeds, dtype=float)
        if len(bl) < 2:
            continue
        bl_std = float(np.std(bl, ddof=1))
        bl_mean = float(np.mean(bl))
        ratio = bl_std / abs(bl_mean) if abs(bl_mean) > 1e-12 else float("inf")
        if ratio < 0.01 or bl_std < 1e-6:
            entries.append({
                "exp_id": r.exp_id,
                "task": r.unique_task_key,
                "baseline_mean": bl_mean,
                "baseline_std": bl_std,
                "cv_ratio": ratio,
                "cohens_d": r.cohens_d,
                "inflated_d": abs(r.cohens_d) > 10,
                "recommendation": "exclude" if abs(r.cohens_d) > 10 else "downweight",
            })
    return {"entries": entries, "n_degenerate": len(entries)}


def _diag_avito_safety(results: list[TaskResult]) -> dict:
    """D6: Avito safety assessment across iterations."""
    avito = [r for r in results if "avito" in r.dataset.lower() or "ad-ctr" in r.task]
    entries = []
    for r in avito:
        bl_mean = float(np.mean(r.baseline_seeds))
        mt_mean = float(np.mean(r.method_seeds))
        pct_diff = (mt_mean - bl_mean) / bl_mean * 100 if bl_mean != 0 else 0.0
        # For MAE, positive pct_diff = method is worse
        worst_seed_pct = 0.0
        for i in range(r.n_seeds):
            if i < len(r.baseline_seeds) and i < len(r.method_seeds):
                s_pct = (r.method_seeds[i] - r.baseline_seeds[i]) / r.baseline_seeds[i] * 100 if r.baseline_seeds[i] != 0 else 0.0
                worst_seed_pct = max(worst_seed_pct, s_pct)

        entries.append({
            "exp_id": r.exp_id,
            "iteration": r.iteration,
            "method_name": r.method_name,
            "task": r.task,
            "baseline_mae": bl_mean,
            "method_mae": mt_mean,
            "pct_diff": pct_diff,
            "worst_seed_pct_diff": worst_seed_pct,
            "within_5pct": abs(pct_diff) <= 5.0,
            "cohens_d": r.cohens_d,
        })

    within_5 = sum(1 for e in entries if e["within_5pct"])
    return {
        "entries": entries,
        "n_experiments": len(entries),
        "n_within_5pct": within_5,
        "safety_claim_supported": within_5 == len(entries) if entries else False,
    }


# ============================================================
# Phase E: Publication Readiness Scorecard
# ============================================================
def publication_readiness_scorecard(
    results: list[TaskResult],
    meta_results: dict,
    diagnostics: dict,
) -> dict:
    """7-dimension scorecard, each 0-3. Total max = 21."""
    scores = {}

    # 1. Pooled effect size (GREEN+YELLOW pool)
    gy = meta_results.get("green_yellow", {})
    d_gy = gy.get("pooled_d", 0.0)
    ci_lo = gy.get("ci_lo", 0.0)
    if d_gy <= 0:
        scores["effect_size"] = 0
    elif d_gy < 0.3:
        scores["effect_size"] = 1
    elif d_gy < 0.8:
        scores["effect_size"] = 2
    else:
        scores["effect_size"] = 3 if ci_lo > 0 else 2

    # 2. Statistical significance
    p_gy = gy.get("p_value", 1.0)
    if p_gy >= 0.10:
        scores["significance"] = 0
    elif p_gy >= 0.05:
        scores["significance"] = 1
    elif p_gy >= 0.01:
        scores["significance"] = 2
    else:
        scores["significance"] = 3

    # 3. Classification evidence
    cls = meta_results.get("classification_only", {})
    d_cls = cls.get("pooled_d", 0.0)
    p_cls = cls.get("p_value", 1.0)
    if d_cls <= 0:
        scores["classification_evidence"] = 0
    elif p_cls > 0.10:
        scores["classification_evidence"] = 1 if d_cls > 0 else 0
    elif p_cls > 0.05:
        scores["classification_evidence"] = 2
    else:
        scores["classification_evidence"] = 3

    # 4. Safety (no catastrophic failures)
    avito_diag = diagnostics.get("avito_safety", {})
    entries = avito_diag.get("entries", [])
    if not entries:
        scores["safety"] = 2  # no data to assess
    else:
        worst_pct = max(abs(e.get("pct_diff", 0)) for e in entries)
        if worst_pct > 20:
            scores["safety"] = 0
        elif worst_pct > 10:
            scores["safety"] = 1
        elif worst_pct > 5:
            scores["safety"] = 2
        else:
            scores["safety"] = 3

    # 5. Mechanism evidence (gate learning)
    gate_diag = diagnostics.get("gate_learning", {})
    n_stasis = gate_diag.get("n_stasis", 0)
    n_total = gate_diag.get("n_with_gate_stats", 0)
    corr = gate_diag.get("correlation_deviation_vs_d", 0)
    if n_total == 0:
        scores["mechanism"] = 0
    elif n_stasis == n_total:
        scores["mechanism"] = 0
    elif n_stasis > n_total * 0.5:
        scores["mechanism"] = 1
    elif abs(corr) > 0.5:
        scores["mechanism"] = 3
    else:
        scores["mechanism"] = 2

    # 6. Reproducibility (cross-experiment consistency)
    # Check if same task across experiments gives consistent sign
    task_ds: dict[str, list] = {}
    for r in results:
        if r.dataset == "synthetic" or r.grade == "RED":
            continue
        key = r.unique_task_key
        if key not in task_ds:
            task_ds[key] = []
        task_ds[key].append(r.cohens_d)

    consistent_count = 0
    total_multi = 0
    for key, ds in task_ds.items():
        if len(ds) >= 2:
            total_multi += 1
            if all(d > 0 for d in ds) or all(d < 0 for d in ds) or all(d == 0 for d in ds):
                consistent_count += 1

    if total_multi == 0:
        scores["reproducibility"] = 1
    else:
        ratio = consistent_count / total_multi
        if ratio < 0.5:
            scores["reproducibility"] = 0
        elif ratio < 0.7:
            scores["reproducibility"] = 1
        elif ratio < 0.9:
            scores["reproducibility"] = 2
        else:
            scores["reproducibility"] = 3

    # 7. SOTA integration benefit (RelGNN results)
    relgnn = [r for r in results if r.architecture == "RelGNN"]
    if not relgnn:
        scores["sota_integration"] = 0
    else:
        relgnn_ds = [r.cohens_d for r in relgnn]
        mean_d = np.mean(relgnn_ds)
        if mean_d < 0:
            scores["sota_integration"] = 0
        elif mean_d < 0.2:
            scores["sota_integration"] = 1
        elif mean_d < 0.5:
            scores["sota_integration"] = 2
        else:
            scores["sota_integration"] = 3

    total = sum(scores.values())
    # Recommendation: READY (>=15), CONDITIONAL (8-14), NOT_READY (<8)
    if total >= 15:
        recommendation = 3  # READY
        rec_label = "READY"
    elif total >= 8:
        recommendation = 2  # CONDITIONAL
        rec_label = "CONDITIONAL"
    else:
        recommendation = 1  # NOT_READY
        rec_label = "NOT_READY"

    scorecard = {
        "scores": scores,
        "total": total,
        "max_possible": 21,
        "recommendation_code": recommendation,
        "recommendation_label": rec_label,
    }
    logger.info(
        f"Phase E Scorecard: total={total}/21, recommendation={rec_label}"
    )
    for dim, score in scores.items():
        logger.info(f"  {dim}: {score}/3")
    return scorecard


# ============================================================
# Output Formatting (exp_eval_sol_out schema)
# ============================================================
def format_output(
    results: list[TaskResult],
    meta_results: dict,
    diagnostics: dict,
    scorecard: dict,
) -> dict:
    """Format everything into exp_eval_sol_out.json schema."""

    # --- metrics_agg ---
    gy = meta_results.get("green_yellow", {})
    go = meta_results.get("green_only", {})
    cls_pool = meta_results.get("classification_only", {})
    reg_pool = meta_results.get("regression_only", {})

    real = [r for r in results if r.dataset != "synthetic"]
    grades = [r.grade for r in real]

    metrics_agg = {
        "n_experiments": len(EXPERIMENT_PATHS),
        "n_task_comparisons": len(real),
        "n_green": grades.count("GREEN"),
        "n_yellow": grades.count("YELLOW"),
        "n_red": grades.count("RED"),
        # GREEN-only pool
        "pooled_d_green_only": go.get("pooled_d", 0.0),
        "pooled_d_green_only_ci_lo": go.get("ci_lo", 0.0),
        "pooled_d_green_only_ci_hi": go.get("ci_hi", 0.0),
        "pooled_d_green_only_p": go.get("p_value", 1.0),
        "pooled_d_green_only_k": go.get("k", 0),
        "pooled_d_green_only_tau2": go.get("tau2", 0.0),
        "pooled_d_green_only_I2": go.get("I2", 0.0),
        # GREEN+YELLOW pool (primary)
        "pooled_d_primary": gy.get("pooled_d", 0.0),
        "pooled_d_primary_ci_lo": gy.get("ci_lo", 0.0),
        "pooled_d_primary_ci_hi": gy.get("ci_hi", 0.0),
        "pooled_d_primary_p": gy.get("p_value", 1.0),
        "pooled_d_primary_k": gy.get("k", 0),
        "pooled_d_primary_tau2": gy.get("tau2", 0.0),
        "pooled_d_primary_I2": gy.get("I2", 0.0),
        "pooled_d_primary_Q": gy.get("Q", 0.0),
        "pooled_d_primary_Q_p": gy.get("Q_p", 1.0),
        "pooled_d_primary_pred_lo": gy.get("pred_lo", 0.0),
        "pooled_d_primary_pred_hi": gy.get("pred_hi", 0.0),
        # Classification subgroup
        "pooled_d_classification": cls_pool.get("pooled_d", 0.0),
        "pooled_d_classification_ci_lo": cls_pool.get("ci_lo", 0.0),
        "pooled_d_classification_ci_hi": cls_pool.get("ci_hi", 0.0),
        "pooled_d_classification_p": cls_pool.get("p_value", 1.0),
        "pooled_d_classification_k": cls_pool.get("k", 0),
        # Regression subgroup
        "pooled_d_regression": reg_pool.get("pooled_d", 0.0),
        "pooled_d_regression_ci_lo": reg_pool.get("ci_lo", 0.0),
        "pooled_d_regression_ci_hi": reg_pool.get("ci_hi", 0.0),
        "pooled_d_regression_p": reg_pool.get("p_value", 1.0),
        "pooled_d_regression_k": reg_pool.get("k", 0),
        # Diagnostics summary
        "diag_val_test_mean_inflation": diagnostics.get("val_test_inflation", {}).get("mean_inflation_ratio", 0.0),
        "diag_class_vs_reg_p": diagnostics.get("class_vs_reg", {}).get("p_subgroup_difference", 1.0),
        "diag_amazon_sign_consistent": 1.0 if diagnostics.get("amazon_stability", {}).get("d_sign_consistent", False) else 0.0,
        "diag_gate_n_stasis": float(diagnostics.get("gate_learning", {}).get("n_stasis", 0)),
        "diag_gate_corr_vs_d": diagnostics.get("gate_learning", {}).get("correlation_deviation_vs_d", 0.0),
        "diag_n_degenerate_baselines": float(diagnostics.get("degenerate_baselines", {}).get("n_degenerate", 0)),
        "diag_avito_safety_supported": 1.0 if diagnostics.get("avito_safety", {}).get("safety_claim_supported", False) else 0.0,
        # Scorecard
        "scorecard_effect_size": float(scorecard["scores"]["effect_size"]),
        "scorecard_significance": float(scorecard["scores"]["significance"]),
        "scorecard_classification": float(scorecard["scores"]["classification_evidence"]),
        "scorecard_safety": float(scorecard["scores"]["safety"]),
        "scorecard_mechanism": float(scorecard["scores"]["mechanism"]),
        "scorecard_reproducibility": float(scorecard["scores"]["reproducibility"]),
        "scorecard_sota_integration": float(scorecard["scores"]["sota_integration"]),
        "scorecard_total": float(scorecard["total"]),
        "scorecard_max": 21.0,
        "scorecard_recommendation": float(scorecard["recommendation_code"]),
    }

    # Ensure no NaN/Inf in metrics_agg
    for k, v in list(metrics_agg.items()):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            metrics_agg[k] = 0.0

    # --- datasets ---
    datasets = []

    # Dataset 1: Evidence Registry (per-task per-experiment entries)
    evidence_examples = []
    for r in real:
        input_str = (
            f"Experiment: {r.exp_id} (iter {r.iteration})\n"
            f"Task: {r.dataset}/{r.task} ({r.task_type})\n"
            f"Method: {r.method_name} vs baseline ({r.architecture})\n"
            f"Metric: {r.metric_name} ({'higher' if r.higher_is_better else 'lower'} is better)\n"
            f"Seeds: {r.n_seeds}, Epochs: {r.epochs_actual}/{r.epochs_planned}\n"
            f"Eval split: {r.eval_split}\n"
            f"Grade: {r.grade} (flags: {', '.join(r.grade_flags) if r.grade_flags else 'none'})"
        )
        bl_mean = float(np.mean(r.baseline_seeds)) if r.baseline_seeds else 0.0
        mt_mean = float(np.mean(r.method_seeds)) if r.method_seeds else 0.0
        output_str = (
            f"Baseline {r.metric_name}: {bl_mean:.4f}, "
            f"Method {r.metric_name}: {mt_mean:.4f}, "
            f"Cohen's d: {r.cohens_d:.3f} [{r.cohens_d_ci[0]:.3f}, {r.cohens_d_ci[1]:.3f}], "
            f"p_ttest: {r.p_ttest:.4f}, p_wilcoxon: {r.p_wilcoxon:.4f}"
        )
        example = {
            "input": input_str,
            "output": output_str,
            "predict_baseline": f"{bl_mean:.6f}",
            "predict_our_method": f"{mt_mean:.6f}",
            "eval_cohens_d": _safe_float(r.cohens_d),
            "eval_cohens_d_ci_lo": _safe_float(r.cohens_d_ci[0]),
            "eval_cohens_d_ci_hi": _safe_float(r.cohens_d_ci[1]),
            "eval_se_d": _safe_float(r.se_d),
            "eval_p_ttest": _safe_float(r.p_ttest),
            "eval_p_wilcoxon": _safe_float(r.p_wilcoxon),
            "eval_raw_effect": _safe_float(r.raw_effect),
            "eval_raw_effect_ci_lo": _safe_float(r.raw_effect_ci[0]),
            "eval_raw_effect_ci_hi": _safe_float(r.raw_effect_ci[1]),
            "eval_n_seeds": float(r.n_seeds),
            "eval_grade_code": {"GREEN": 3.0, "YELLOW": 2.0, "RED": 1.0}.get(r.grade, 0.0),
            "eval_is_classification": 1.0 if r.task_type == "classification" else 0.0,
            "eval_val_test_identical": 1.0 if r.val_test_identical else 0.0,
            "eval_iteration": float(r.iteration),
        }
        evidence_examples.append(example)

    datasets.append({
        "dataset": "evidence_registry",
        "examples": evidence_examples,
    })

    # Dataset 2: Meta-Analysis Pools
    pool_examples = []
    for pool_name, pool_data in meta_results.items():
        input_str = f"Meta-analysis pool: {pool_name}"
        output_str = (
            f"k={pool_data['k']}, pooled_d={pool_data['pooled_d']:.4f}, "
            f"CI=[{pool_data['ci_lo']:.4f}, {pool_data['ci_hi']:.4f}], "
            f"p={pool_data['p_value']:.6f}, tau2={pool_data['tau2']:.4f}, "
            f"I2={pool_data['I2']:.1f}%"
        )
        pool_examples.append({
            "input": input_str,
            "output": output_str,
            "predict_baseline": "0.0",
            "predict_our_method": f"{pool_data['pooled_d']:.6f}",
            "eval_pooled_d": _safe_float(pool_data["pooled_d"]),
            "eval_ci_lo": _safe_float(pool_data["ci_lo"]),
            "eval_ci_hi": _safe_float(pool_data["ci_hi"]),
            "eval_p_value": _safe_float(pool_data["p_value"]),
            "eval_tau2": _safe_float(pool_data["tau2"]),
            "eval_I2": _safe_float(pool_data["I2"]),
            "eval_Q": _safe_float(pool_data["Q"]),
            "eval_k": float(pool_data["k"]),
        })
    datasets.append({
        "dataset": "meta_analysis_pools",
        "examples": pool_examples,
    })

    # Dataset 3: Diagnostics
    diag_examples = []

    # D1
    d1 = diagnostics.get("val_test_inflation", {})
    diag_examples.append({
        "input": "Diagnostic: Val-Test Inflation Factor",
        "output": f"Mean inflation ratio: {d1.get('mean_inflation_ratio', 0):.3f} across {d1.get('n_comparisons', 0)} comparisons",
        "predict_baseline": "1.0",
        "predict_our_method": f"{d1.get('mean_inflation_ratio', 0):.6f}",
        "eval_mean_inflation_ratio": _safe_float(d1.get("mean_inflation_ratio", 0)),
        "eval_n_comparisons": float(d1.get("n_comparisons", 0)),
    })

    # D2
    d2 = diagnostics.get("class_vs_reg", {})
    diag_examples.append({
        "input": "Diagnostic: Classification vs Regression Subgroup",
        "output": (
            f"Classification mean d={d2.get('classification_mean_d', 0):.3f} (n={d2.get('classification_n', 0)}), "
            f"Regression mean d={d2.get('regression_mean_d', 0):.3f} (n={d2.get('regression_n', 0)}), "
            f"p_difference={d2.get('p_subgroup_difference', 1):.4f}"
        ),
        "predict_baseline": f"{d2.get('regression_mean_d', 0):.6f}",
        "predict_our_method": f"{d2.get('classification_mean_d', 0):.6f}",
        "eval_cls_mean_d": _safe_float(d2.get("classification_mean_d", 0)),
        "eval_reg_mean_d": _safe_float(d2.get("regression_mean_d", 0)),
        "eval_p_subgroup": _safe_float(d2.get("p_subgroup_difference", 1)),
    })

    # D3
    d3 = diagnostics.get("amazon_stability", {})
    diag_examples.append({
        "input": "Diagnostic: Amazon Stability",
        "output": (
            f"n_experiments={d3.get('n_experiments', 0)}, "
            f"d_range=[{d3.get('d_range', (0, 0))[0]:.3f}, {d3.get('d_range', (0, 0))[1]:.3f}], "
            f"sign_consistent={d3.get('d_sign_consistent', False)}"
        ),
        "predict_baseline": "0.0",
        "predict_our_method": f"{1.0 if d3.get('d_sign_consistent', False) else 0.0}",
        "eval_n_amazon_experiments": float(d3.get("n_experiments", 0)),
        "eval_amazon_sign_consistent": 1.0 if d3.get("d_sign_consistent", False) else 0.0,
    })

    # D4
    d4 = diagnostics.get("gate_learning", {})
    diag_examples.append({
        "input": "Diagnostic: Gate Learning",
        "output": (
            f"n_with_stats={d4.get('n_with_gate_stats', 0)}, "
            f"n_stasis={d4.get('n_stasis', 0)}, "
            f"corr(deviation, d)={d4.get('correlation_deviation_vs_d', 0):.3f}"
        ),
        "predict_baseline": "0.5",
        "predict_our_method": f"{d4.get('correlation_deviation_vs_d', 0):.6f}",
        "eval_n_gate_stats": float(d4.get("n_with_gate_stats", 0)),
        "eval_n_stasis": float(d4.get("n_stasis", 0)),
        "eval_gate_corr": _safe_float(d4.get("correlation_deviation_vs_d", 0)),
    })

    # D5
    d5 = diagnostics.get("degenerate_baselines", {})
    diag_examples.append({
        "input": "Diagnostic: Degenerate Baselines",
        "output": f"n_degenerate={d5.get('n_degenerate', 0)}",
        "predict_baseline": "0.0",
        "predict_our_method": f"{d5.get('n_degenerate', 0):.1f}",
        "eval_n_degenerate": float(d5.get("n_degenerate", 0)),
    })

    # D6
    d6 = diagnostics.get("avito_safety", {})
    diag_examples.append({
        "input": "Diagnostic: Avito Safety Assessment",
        "output": (
            f"n_experiments={d6.get('n_experiments', 0)}, "
            f"n_within_5pct={d6.get('n_within_5pct', 0)}, "
            f"safety_supported={d6.get('safety_claim_supported', False)}"
        ),
        "predict_baseline": "1.0",
        "predict_our_method": f"{1.0 if d6.get('safety_claim_supported', False) else 0.0}",
        "eval_n_avito_experiments": float(d6.get("n_experiments", 0)),
        "eval_n_within_5pct": float(d6.get("n_within_5pct", 0)),
        "eval_safety_supported": 1.0 if d6.get("safety_claim_supported", False) else 0.0,
    })

    datasets.append({
        "dataset": "diagnostics",
        "examples": diag_examples,
    })

    # Dataset 4: Publication Scorecard
    sc_scores = scorecard["scores"]
    sc_examples = []
    for dim, score in sc_scores.items():
        sc_examples.append({
            "input": f"Scorecard dimension: {dim}",
            "output": f"Score: {score}/3",
            "predict_baseline": "3.0",
            "predict_our_method": f"{score:.1f}",
            "eval_score": float(score),
        })
    sc_examples.append({
        "input": f"Scorecard: Overall (recommendation={scorecard['recommendation_label']})",
        "output": f"Total: {scorecard['total']}/{scorecard['max_possible']}",
        "predict_baseline": f"{scorecard['max_possible']:.1f}",
        "predict_our_method": f"{scorecard['total']:.1f}",
        "eval_score": float(scorecard["total"]),
    })
    datasets.append({
        "dataset": "publication_scorecard",
        "examples": sc_examples,
    })

    return {
        "metadata": {
            "evaluation_name": "Definitive Meta-Analysis of RAMA/CAMA (Iterations 2-4)",
            "description": (
                "Rigorous meta-analysis integrating 12 experiments across 8 unique tasks "
                "and 6 RelBench datasets. Includes evidence grading, per-task Cohen's d "
                "with bootstrap CIs, DerSimonian-Laird random-effects meta-analysis, "
                "critical diagnostics, and publication readiness scorecard."
            ),
            "n_bootstrap": N_BOOTSTRAP,
            "random_seed": 42,
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }


def _safe_float(v) -> float:
    """Convert to float, replacing NaN/Inf with 0.0."""
    if v is None:
        return 0.0
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


# ============================================================
# Main
# ============================================================
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Definitive Meta-Analysis of RAMA/CAMA (Iterations 2-4)")
    logger.info("=" * 60)

    # Phase A: Extract and grade
    logger.info("Phase A: Extracting evidence registry...")
    results = extract_all_task_results()
    results = grade_evidence(results)

    # Phase B: Compute effect sizes
    results = compute_all_effect_sizes(results)

    # Re-grade after effect sizes to catch inflated d
    results = regrade_after_effect_sizes(results)

    # Log summary table
    logger.info("=" * 60)
    logger.info("Evidence Registry Summary:")
    logger.info(f"{'Exp':<15} {'Task':<40} {'d':>8} {'p':>8} {'Grade':<6}")
    logger.info("-" * 80)
    for r in results:
        if r.dataset == "synthetic":
            continue
        logger.info(
            f"{r.exp_id:<15} {r.unique_task_key:<40} "
            f"{r.cohens_d:>8.3f} {r.p_ttest:>8.4f} {r.grade:<6}"
        )
    logger.info("=" * 60)

    # Phase C: Meta-analyses
    meta_results = run_meta_analyses(results)

    # Phase D: Diagnostics
    diagnostics = run_all_diagnostics(results)

    # Phase E: Scorecard
    scorecard = publication_readiness_scorecard(results, meta_results, diagnostics)

    # Format and save output
    output = format_output(results, meta_results, diagnostics, scorecard)
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output saved to {out_path}")
    logger.info(f"Output size: {out_path.stat().st_size / 1024:.1f} KB")

    # Print key results
    logger.info("=" * 60)
    logger.info("KEY RESULTS:")
    gy = meta_results["green_yellow"]
    logger.info(
        f"Primary pooled d (GREEN+YELLOW): {gy['pooled_d']:.3f} "
        f"[{gy['ci_lo']:.3f}, {gy['ci_hi']:.3f}], p={gy['p_value']:.4f}"
    )
    logger.info(f"I2={gy['I2']:.1f}%, tau2={gy['tau2']:.4f}")
    logger.info(
        f"Scorecard: {scorecard['total']}/{scorecard['max_possible']} "
        f"-> {scorecard['recommendation_label']}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
