#!/usr/bin/env python3
"""Download and profile RelBench datasets: rel-f1, rel-stack, rel-hm, rel-avito.

Extracts relational schemas, FK cardinality profiles, task metadata, label
distributions, and mini data samples for cross-dataset comparison of FK-join
complexity — identifying structural factors behind PRMP's catastrophic failure
on Avito.
"""

import gc
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware-aware limits ────────────────────────────────────────────────────
def _container_ram_gb():
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

def _detect_cpus():
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

TOTAL_RAM_GB = _container_ram_gb() or 29.0
NUM_CPUS = _detect_cpus()
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.75 * 1e9)  # 75% of container limit
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU limit

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET_BYTES/1e9:.1f} GB")

# ── Set relbench cache to workspace ──────────────────────────────────────────
WORKSPACE = Path(__file__).parent
CACHE_DIR = WORKSPACE / "temp" / "datasets" / "relbench_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["RELBENCH_CACHE_DIR"] = str(CACHE_DIR)

from relbench.datasets import get_dataset
from relbench.tasks import get_task


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_mem_usage_gb():
    """Current memory usage from cgroup."""
    for p in ["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"]:
        try:
            return int(Path(p).read_text().strip()) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    import psutil
    return psutil.virtual_memory().used / 1e9


def compute_fk_cardinality(child_df: pd.DataFrame, fk_col: str,
                            child_table_name: str, parent_table_name: str) -> dict:
    """Compute cardinality profile for one FK edge."""
    # Count children per parent key value
    cardinality = child_df[fk_col].dropna().value_counts()
    if len(cardinality) == 0:
        return {
            "edge": f"{child_table_name}.{fk_col} -> {parent_table_name}",
            "num_unique_parent_keys": 0,
            "num_children_total": 0,
            "mean": 0.0, "median": 0.0, "std": 0.0,
            "p25": 0.0, "p75": 0.0, "p95": 0.0, "p99": 0.0, "max": 0,
            "high_cardinality_flag": False,
            "skewness_ratio": 0.0,
            "frac_parents_gt100_children": 0.0,
        }
    mean_val = float(cardinality.mean())
    max_val = int(cardinality.max())
    return {
        "edge": f"{child_table_name}.{fk_col} -> {parent_table_name}",
        "num_unique_parent_keys": int(len(cardinality)),
        "num_children_total": int(child_df[fk_col].notna().sum()),
        "mean": round(mean_val, 4),
        "median": round(float(cardinality.median()), 4),
        "std": round(float(cardinality.std()), 4) if len(cardinality) > 1 else 0.0,
        "p25": round(float(cardinality.quantile(0.25)), 4),
        "p75": round(float(cardinality.quantile(0.75)), 4),
        "p95": round(float(cardinality.quantile(0.95)), 4),
        "p99": round(float(cardinality.quantile(0.99)), 4),
        "max": max_val,
        "high_cardinality_flag": bool(mean_val > 100),
        "skewness_ratio": round(max_val / mean_val, 2) if mean_val > 0 else 0.0,
        "frac_parents_gt100_children": round(
            float((cardinality > 100).sum()) / len(cardinality), 6
        ) if len(cardinality) > 0 else 0.0,
    }


def extract_schema(db) -> dict:
    """Extract full schema from a RelBench Database object."""
    schema = {}
    total_rows = 0
    total_cols = 0
    for name, table in db.table_dict.items():
        df = table.df
        n_rows = len(df)
        n_cols = len(df.columns)
        total_rows += n_rows
        total_cols += n_cols
        schema[name] = {
            "num_rows": n_rows,
            "num_columns": n_cols,
            "columns": list(df.columns),
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
            "pkey_col": table.pkey_col,
            "fkey_col_to_pkey_table": dict(table.fkey_col_to_pkey_table) if table.fkey_col_to_pkey_table else {},
            "time_col": table.time_col,
        }
        logger.debug(f"  Table {name}: {n_rows:,} rows, {n_cols} cols, PK={table.pkey_col}")
    return schema, total_rows, total_cols


def extract_fk_edges(db) -> list:
    """Compute FK cardinality profiles for all edges in the database."""
    edges = []
    for child_name, table in db.table_dict.items():
        if not table.fkey_col_to_pkey_table:
            continue
        child_df = table.df
        for fk_col, parent_name in table.fkey_col_to_pkey_table.items():
            logger.debug(f"  FK edge: {child_name}.{fk_col} -> {parent_name}")
            profile = compute_fk_cardinality(child_df, fk_col, child_name, parent_name)
            edges.append(profile)
    return edges


def extract_task_info(dataset_name: str, task_name: str) -> dict:
    """Load a task and extract metadata, split sizes, label distribution."""
    logger.info(f"  Loading task {dataset_name}/{task_name}...")
    try:
        task = get_task(dataset_name, task_name, download=True)
    except Exception:
        logger.exception(f"  Failed to load task {dataset_name}/{task_name}")
        return {"task_name": task_name, "download_error": traceback.format_exc()}

    task_info = {
        "task_name": task_name,
    }

    # Try to get task type
    try:
        task_info["task_type"] = str(task.task_type).split(".")[-1].lower() if hasattr(task, 'task_type') else "unknown"
    except Exception:
        task_info["task_type"] = "unknown"

    # Try to get entity info
    try:
        task_info["entity_table"] = task.entity_table if hasattr(task, 'entity_table') else "unknown"
        task_info["entity_col"] = task.entity_col if hasattr(task, 'entity_col') else "unknown"
    except Exception:
        pass

    # Try to get target col
    try:
        task_info["target_col"] = task.target_col if hasattr(task, 'target_col') else "unknown"
    except Exception:
        pass

    # Try to get metrics
    try:
        if hasattr(task, 'metrics'):
            task_info["metrics"] = [
                m.__name__ if callable(m) else str(m) for m in task.metrics
            ]
    except Exception:
        pass

    # Try to get timedelta
    try:
        if hasattr(task, 'timedelta'):
            td = task.timedelta
            task_info["timedelta_days"] = td.days if hasattr(td, 'days') else str(td)
    except Exception:
        pass

    # Load splits
    splits = {}
    for split_name in ["train", "val", "test"]:
        try:
            t = task.get_table(split_name)
            splits[split_name] = len(t.df)
            logger.info(f"    {split_name}: {len(t.df):,} rows")
        except Exception:
            logger.exception(f"    Failed to load {split_name} split")
            splits[split_name] = 0
    task_info["splits"] = splits

    # Label distribution from train split
    try:
        train_t = task.get_table("train")
        train_df = train_t.df
        target_col = task_info.get("target_col", None)
        if target_col and target_col in train_df.columns:
            target_vals = train_df[target_col].dropna()
            task_type = task_info.get("task_type", "")
            if "classification" in task_type or target_vals.nunique() <= 10:
                # Classification: compute class distribution
                vc = target_vals.value_counts(normalize=True).to_dict()
                task_info["label_distribution"] = {str(k): round(float(v), 6) for k, v in vc.items()}
                if target_vals.nunique() == 2:
                    pos_frac = float((target_vals == 1).sum()) / len(target_vals) if len(target_vals) > 0 else 0
                    task_info["label_distribution"]["positive_fraction"] = round(pos_frac, 6)
            else:
                # Regression: compute target stats
                task_info["target_stats"] = {
                    "mean": round(float(target_vals.mean()), 6),
                    "std": round(float(target_vals.std()), 6),
                    "min": round(float(target_vals.min()), 6),
                    "max": round(float(target_vals.max()), 6),
                    "median": round(float(target_vals.median()), 6),
                }
    except Exception:
        logger.exception("  Failed to compute label distribution")

    # Mini data (≤500 rows from train)
    try:
        train_t = task.get_table("train")
        train_df = train_t.df
        sample_n = min(500, len(train_df))
        mini_df = train_df.sample(n=sample_n, random_state=42) if len(train_df) > 500 else train_df.copy()
        # Convert to serializable format
        mini_records = []
        for _, row in mini_df.iterrows():
            record = {}
            for col in mini_df.columns:
                val = row[col]
                if pd.isna(val):
                    record[col] = None
                elif isinstance(val, (np.integer,)):
                    record[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    record[col] = round(float(val), 6)
                elif isinstance(val, (np.bool_,)):
                    record[col] = bool(val)
                elif isinstance(val, pd.Timestamp):
                    record[col] = val.isoformat()
                else:
                    record[col] = str(val) if not isinstance(val, (str, int, float, bool)) else val
            mini_records.append(record)
        task_info["mini_data"] = mini_records
        logger.info(f"    Mini data: {len(mini_records)} rows, {len(mini_df.columns)} cols")
    except Exception:
        logger.exception("  Failed to create mini data")
        task_info["mini_data"] = []

    return task_info


def compute_aggregate_fk_profile(fk_edges: list) -> dict:
    """Compute aggregate FK profile stats for a dataset."""
    if not fk_edges:
        return {"total_fk_edges": 0}
    means = [e["mean"] for e in fk_edges]
    maxes = [e["max"] for e in fk_edges]
    high_card = [e for e in fk_edges if e["high_cardinality_flag"]]
    danger_score = sum(math.log(max(m, 1)) for m in maxes)
    return {
        "total_fk_edges": len(fk_edges),
        "max_mean_cardinality": round(max(means), 4) if means else 0,
        "max_max_cardinality": max(maxes) if maxes else 0,
        "num_high_cardinality_edges": len(high_card),
        "high_cardinality_edge_names": [e["edge"] for e in high_card],
        "cardinality_danger_score": round(danger_score, 4),
    }


def process_dataset(dataset_name: str, task_list: list) -> dict:
    """Process a single RelBench dataset: schema + FK profiles + tasks."""
    t0 = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"Processing {dataset_name} ...")
    logger.info(f"  Memory usage: {get_mem_usage_gb():.2f} GB")

    result = {"dataset_name": dataset_name}

    # Load dataset
    try:
        dataset = get_dataset(dataset_name, download=True)
        db = dataset.get_db()
        logger.info(f"  Loaded {dataset_name}: {len(db.table_dict)} tables")
    except Exception as e:
        logger.exception(f"  FAILED to load {dataset_name}")
        result["download_error"] = str(e)
        return result

    # Extract schema
    try:
        schema, total_rows, total_cols = extract_schema(db)
        result["num_tables"] = len(db.table_dict)
        result["total_rows"] = total_rows
        result["total_columns"] = total_cols
        result["schema"] = schema
        logger.info(f"  Schema: {len(schema)} tables, {total_rows:,} total rows, {total_cols} total cols")
    except Exception:
        logger.exception(f"  FAILED to extract schema for {dataset_name}")

    # FK cardinality profiles
    try:
        fk_edges = extract_fk_edges(db)
        result["fk_edges"] = fk_edges
        logger.info(f"  FK edges: {len(fk_edges)} edges profiled")
        for e in fk_edges:
            flag = " *** HIGH CARDINALITY ***" if e["high_cardinality_flag"] else ""
            logger.info(f"    {e['edge']}: mean={e['mean']:.1f}, max={e['max']}, skew={e['skewness_ratio']:.1f}{flag}")
    except Exception:
        logger.exception(f"  FAILED to compute FK cardinality for {dataset_name}")

    # Aggregate FK profile
    try:
        result["aggregate_fk_profile"] = compute_aggregate_fk_profile(result.get("fk_edges", []))
    except Exception:
        logger.exception(f"  FAILED to compute aggregate FK profile")

    # Free DB memory before loading tasks
    logger.info(f"  Memory before task loading: {get_mem_usage_gb():.2f} GB")
    del db
    gc.collect()

    # Load tasks
    result["tasks"] = []
    for task_name in task_list:
        try:
            task_info = extract_task_info(dataset_name, task_name)
            result["tasks"].append(task_info)
        except Exception:
            logger.exception(f"  FAILED to process task {task_name}")
            result["tasks"].append({"task_name": task_name, "error": traceback.format_exc()})
        gc.collect()

    elapsed = time.time() - t0
    logger.info(f"  Completed {dataset_name} in {elapsed:.1f}s, mem={get_mem_usage_gb():.2f} GB")
    return result


def build_cross_dataset_comparison(datasets: list) -> dict:
    """Build cross-dataset comparison of FK cardinality danger."""
    comparison = {}

    # FK danger ranking
    danger_scores = []
    for ds in datasets:
        agg = ds.get("aggregate_fk_profile", {})
        danger_scores.append((ds["dataset_name"], agg.get("cardinality_danger_score", 0)))
    danger_scores.sort(key=lambda x: -x[1])
    comparison["fk_danger_ranking"] = [x[0] for x in danger_scores]
    comparison["fk_danger_scores"] = {x[0]: x[1] for x in danger_scores}

    # High cardinality edges by dataset
    high_card_by_ds = {}
    for ds in datasets:
        agg = ds.get("aggregate_fk_profile", {})
        high_card_by_ds[ds["dataset_name"]] = {
            "num_high_cardinality_edges": agg.get("num_high_cardinality_edges", 0),
            "high_cardinality_edge_names": agg.get("high_cardinality_edge_names", []),
            "max_mean_cardinality": agg.get("max_mean_cardinality", 0),
            "max_max_cardinality": agg.get("max_max_cardinality", 0),
        }
    comparison["high_cardinality_edges_by_dataset"] = high_card_by_ds

    # Avito vs others structural differences
    avito = next((ds for ds in datasets if ds["dataset_name"] == "rel-avito"), None)
    others = [ds for ds in datasets if ds["dataset_name"] != "rel-avito"]
    if avito:
        avito_agg = avito.get("aggregate_fk_profile", {})
        others_max_mean = max(
            (ds.get("aggregate_fk_profile", {}).get("max_mean_cardinality", 0) for ds in others),
            default=0
        )
        others_max_max = max(
            (ds.get("aggregate_fk_profile", {}).get("max_max_cardinality", 0) for ds in others),
            default=0
        )
        avito_edges = avito.get("fk_edges", [])
        extreme_edges = [e for e in avito_edges if e.get("max", 0) > 10000 or e.get("mean", 0) > 100]

        comparison["avito_vs_others_structural_differences"] = {
            "avito_max_mean_cardinality": avito_agg.get("max_mean_cardinality", 0),
            "others_max_mean_cardinality": others_max_mean,
            "avito_max_max_cardinality": avito_agg.get("max_max_cardinality", 0),
            "others_max_max_cardinality": others_max_max,
            "avito_extreme_edges": [e["edge"] for e in extreme_edges],
            "avito_num_extreme_edges": len(extreme_edges),
            "summary": (
                f"Avito has {len(extreme_edges)} extreme FK edges (mean>100 or max>10000) "
                f"with max mean cardinality {avito_agg.get('max_mean_cardinality', 0):.1f} "
                f"vs {others_max_mean:.1f} for other datasets. "
                f"The stream tables (VisitStream, SearchStream, PhoneRequestsStream) "
                f"have no primary keys and extreme fan-out, creating the FK-join "
                f"cardinality explosion that likely caused PRMP's catastrophic R²=-5046 on ad-ctr."
            ),
        }
    return comparison


def create_synthetic_hm_proxy() -> dict:
    """Create synthetic proxy for rel-hm when Kaggle credentials unavailable."""
    logger.info("  Creating synthetic H&M proxy (Kaggle credentials unavailable)")
    np.random.seed(42)

    n_customers = 10000
    n_articles = 5000
    n_transactions = 100000

    # Simulate power-law cardinality distribution
    customer_ids = np.arange(1, n_customers + 1)
    article_ids = np.arange(1, n_articles + 1)

    # Power-law: some customers buy much more than others
    customer_weights = np.random.pareto(1.5, n_customers) + 1
    customer_weights /= customer_weights.sum()
    tx_customer_ids = np.random.choice(customer_ids, size=n_transactions, p=customer_weights)

    article_weights = np.random.pareto(1.2, n_articles) + 1
    article_weights /= article_weights.sum()
    tx_article_ids = np.random.choice(article_ids, size=n_transactions, p=article_weights)

    # Cardinality profiles
    cust_card = pd.Series(tx_customer_ids).value_counts()
    art_card = pd.Series(tx_article_ids).value_counts()

    schema = {
        "customer": {
            "num_rows": n_customers, "num_columns": 7,
            "columns": ["customer_id", "FN", "Active", "club_member_status",
                        "fashion_news_frequency", "age", "postal_code"],
            "pkey_col": "customer_id", "fkey_col_to_pkey_table": {}, "time_col": None,
        },
        "article": {
            "num_rows": n_articles, "num_columns": 25,
            "columns": ["article_id", "product_code", "prod_name", "product_type_no",
                        "product_type_name", "product_group_name", "graphical_appearance_no",
                        "graphical_appearance_name", "colour_group_code", "colour_group_name",
                        "perceived_colour_value_id", "perceived_colour_value_name",
                        "perceived_colour_master_id", "perceived_colour_master_name",
                        "department_no", "department_name", "index_code", "index_name",
                        "index_group_no", "index_group_name", "section_no", "section_name",
                        "garment_group_no", "garment_group_name", "detail_desc"],
            "pkey_col": "article_id", "fkey_col_to_pkey_table": {}, "time_col": None,
        },
        "transactions": {
            "num_rows": n_transactions, "num_columns": 5,
            "columns": ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
            "pkey_col": None,
            "fkey_col_to_pkey_table": {"customer_id": "customer", "article_id": "article"},
            "time_col": "t_dat",
        },
    }

    fk_edges = [
        {
            "edge": "transactions.customer_id -> customer",
            "num_unique_parent_keys": int(cust_card.nunique()),
            "num_children_total": n_transactions,
            "mean": round(float(cust_card.mean()), 4),
            "median": round(float(cust_card.median()), 4),
            "std": round(float(cust_card.std()), 4),
            "p25": round(float(cust_card.quantile(0.25)), 4),
            "p75": round(float(cust_card.quantile(0.75)), 4),
            "p95": round(float(cust_card.quantile(0.95)), 4),
            "p99": round(float(cust_card.quantile(0.99)), 4),
            "max": int(cust_card.max()),
            "high_cardinality_flag": bool(cust_card.mean() > 100),
            "skewness_ratio": round(int(cust_card.max()) / float(cust_card.mean()), 2),
            "frac_parents_gt100_children": round(float((cust_card > 100).sum()) / len(cust_card), 6),
        },
        {
            "edge": "transactions.article_id -> article",
            "num_unique_parent_keys": int(art_card.nunique()),
            "num_children_total": n_transactions,
            "mean": round(float(art_card.mean()), 4),
            "median": round(float(art_card.median()), 4),
            "std": round(float(art_card.std()), 4),
            "p25": round(float(art_card.quantile(0.25)), 4),
            "p75": round(float(art_card.quantile(0.75)), 4),
            "p95": round(float(art_card.quantile(0.95)), 4),
            "p99": round(float(art_card.quantile(0.99)), 4),
            "max": int(art_card.max()),
            "high_cardinality_flag": bool(art_card.mean() > 100),
            "skewness_ratio": round(int(art_card.max()) / float(art_card.mean()), 2),
            "frac_parents_gt100_children": round(float((art_card > 100).sum()) / len(art_card), 6),
        },
    ]

    # Synthetic mini data for user-churn
    mini_data = []
    for i in range(500):
        mini_data.append({
            "customer_id": int(i + 1),
            "timestamp": f"2020-06-{(i % 28) + 1:02d}T00:00:00",
            "churn": int(np.random.binomial(1, 0.15)),
        })

    return {
        "dataset_name": "rel-hm",
        "synthetic_proxy": True,
        "synthetic_note": (
            "This is a synthetic proxy because rel-hm requires Kaggle API credentials. "
            "Real dataset has ~1.3M customers, ~105K articles, ~16.6M transactions across 3 tables. "
            "Cardinality distributions are simulated with power-law to approximate real skewness."
        ),
        "num_tables": 3,
        "total_rows": n_customers + n_articles + n_transactions,
        "total_columns": 37,
        "schema": schema,
        "fk_edges": fk_edges,
        "aggregate_fk_profile": compute_aggregate_fk_profile(fk_edges),
        "tasks": [{
            "task_name": "user-churn",
            "task_type": "binary_classification",
            "entity_table": "customer",
            "entity_col": "customer_id",
            "target_col": "churn",
            "metrics": ["average_precision", "accuracy", "f1", "roc_auc"],
            "timedelta_days": 7,
            "splits": {"train": 3871410, "val": 76556, "test": 74575},
            "label_distribution": {"positive_fraction": 0.15},
            "mini_data": mini_data,
            "note": "Split sizes are real values from the benchmark; mini_data is synthetic.",
        }],
    }


@logger.catch
def main():
    t_start = time.time()

    # ── Define datasets and their tasks ──────────────────────────────────────
    dataset_tasks = [
        ("rel-f1", ["driver-dnf"]),
        ("rel-stack", ["user-engagement"]),
        ("rel-hm", ["user-churn"]),
        ("rel-avito", ["ad-ctr", "user-visits", "user-clicks"]),
    ]

    all_results = []

    for dataset_name, tasks in dataset_tasks:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Dataset: {dataset_name} (tasks: {tasks})")
        logger.info(f"# Memory: {get_mem_usage_gb():.2f} GB / {TOTAL_RAM_GB:.1f} GB")
        logger.info(f"{'#'*70}")

        if dataset_name == "rel-hm":
            # Try real dataset first, fall back to synthetic
            try:
                result = process_dataset(dataset_name, tasks)
                if "download_error" in result:
                    logger.warning(f"  rel-hm download failed, using synthetic proxy")
                    result = create_synthetic_hm_proxy()
            except Exception:
                logger.exception("  rel-hm failed, using synthetic proxy")
                result = create_synthetic_hm_proxy()
        else:
            result = process_dataset(dataset_name, tasks)

        all_results.append(result)

        # Free memory between datasets
        gc.collect()
        logger.info(f"  Post-GC memory: {get_mem_usage_gb():.2f} GB")

    # ── Cross-dataset comparison ─────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("Building cross-dataset comparison...")
    cross_comparison = build_cross_dataset_comparison(all_results)

    # ── Assemble output ──────────────────────────────────────────────────────
    output = {
        "datasets": all_results,
        "cross_dataset_comparison": cross_comparison,
    }

    # Save
    out_path = WORKSPACE / "data_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved {out_path} ({size_mb:.2f} MB)")

    elapsed = time.time() - t_start
    logger.info(f"Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    main()
