#!/usr/bin/env python3
"""Process cached RelBench parquet datasets into standardized JSON format.

Reads cached parquet files directly — no relbench dependency needed.
Memory-safe: processes ONE dataset at a time, loads only needed columns for large tables.
Previous run cached rel-f1, rel-stack, rel-amazon. rel-hm blocked (Kaggle auth).
"""

import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WS = Path(
    "/ai-inventor/aii_pipeline/runs/leskovec-predictive-residual-message-passing-v2_sti"
    "/3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
)
LOG_DIR = WS / "logs"
CACHE_DIR = WS / "relbench_cache"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware / Memory budget
# ---------------------------------------------------------------------------
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


TOTAL_RAM_GB = _container_ram_gb() or 29.0
logger.info(f"Hardware: {TOTAL_RAM_GB:.1f} GB RAM")

# ---------------------------------------------------------------------------
# Dataset definitions (from RelBench documentation + parquet inspection)
# ---------------------------------------------------------------------------

# FK relationships: {ds_name: {child_table: {fk_col: parent_table}}}
FK_MAP: dict[str, dict[str, dict[str, str]]] = {
    "rel-f1": {
        "races": {"circuitId": "circuits"},
        "results": {
            "raceId": "races",
            "driverId": "drivers",
            "constructorId": "constructors",
        },
        "standings": {"raceId": "races", "driverId": "drivers"},
        "qualifying": {
            "raceId": "races",
            "driverId": "drivers",
            "constructorId": "constructors",
        },
        "constructor_results": {"raceId": "races", "constructorId": "constructors"},
        "constructor_standings": {"raceId": "races", "constructorId": "constructors"},
    },
    "rel-stack": {
        "posts": {"OwnerUserId": "users", "ParentId": "posts"},
        "comments": {"PostId": "posts", "UserId": "users"},
        "votes": {"PostId": "posts", "UserId": "users"},
        "badges": {"UserId": "users"},
        "postLinks": {"PostId": "posts", "RelatedPostId": "posts"},
        "postHistory": {"PostId": "posts", "UserId": "users"},
    },
    "rel-amazon": {
        "review": {"customer_id": "customer", "product_id": "product"},
    },
}

# Primary keys per table
PKEY_MAP: dict[str, dict[str, str | None]] = {
    "rel-f1": {
        "circuits": "circuitId",
        "constructors": "constructorId",
        "drivers": "driverId",
        "races": "raceId",
        "results": "resultId",
        "standings": "driverStandingsId",
        "qualifying": "qualifyId",
        "constructor_results": "constructorResultsId",
        "constructor_standings": "constructorStandingsId",
    },
    "rel-stack": {
        "users": "Id",
        "posts": "Id",
        "comments": "Id",
        "votes": "Id",
        "badges": "Id",
        "postLinks": "Id",
        "postHistory": "Id",
    },
    "rel-amazon": {
        "customer": "customer_id",
        "product": "product_id",
        "review": None,
    },
}

# Task metadata (from RelBench docs)
TASK_META: dict[tuple[str, str], dict] = {
    ("rel-f1", "driver-position"): {
        "entity_col": "driverId",
        "target_col": "position",
        "entity_table": "drivers",
        "timedelta_days": 60,
    },
    ("rel-stack", "post-votes"): {
        "entity_col": "PostId",
        "target_col": "popularity",
        "entity_table": "posts",
        "timedelta_days": 90,
    },
    ("rel-amazon", "item-ltv"): {
        "entity_col": "product_id",
        "target_col": "ltv",
        "entity_table": "product",
        "timedelta_days": 90,
    },
}

# Columns to SKIP when loading entity tables (too large for examples)
SKIP_COLS: dict[str, set[str]] = {
    "posts": {"Body"},
    "product": {"description"},
    "users": {"AboutMe", "ProfileImageUrl", "WebsiteUrl"},
    "postHistory": {"Text", "Comment", "RevisionGUID", "ContentLicense"},
    "comments": {"Text", "ContentLicense", "UserDisplayName"},
}

# Datasets to process (ordered smallest → largest)
DATASETS: list[tuple[str, str]] = [
    ("rel-f1", "driver-position"),
    ("rel-stack", "post-votes"),
    ("rel-amazon", "item-ltv"),
]


# ---------------------------------------------------------------------------
# Schema extraction (from parquet metadata — NO data loading)
# ---------------------------------------------------------------------------
def extract_schema(ds_name: str, db_dir: Path) -> dict:
    """Extract relational schema from parquet file metadata."""
    schema: dict = {"dataset": ds_name, "tables": {}}
    fk_map = FK_MAP.get(ds_name, {})
    pkey_map = PKEY_MAP.get(ds_name, {})

    for f in sorted(db_dir.glob("*.parquet")):
        table_name = f.stem
        pq_schema = pq.read_schema(str(f))
        pq_meta = pq.read_metadata(str(f))

        columns = {
            name: str(pq_schema.field(name).type) for name in pq_schema.names
        }

        schema["tables"][table_name] = {
            "num_rows": pq_meta.num_rows,
            "num_columns": len(pq_schema.names),
            "columns": columns,
            "pkey_col": pkey_map.get(table_name),
            "fkey_col_to_pkey_table": fk_map.get(table_name, {}),
        }

    return schema


# ---------------------------------------------------------------------------
# FK cardinality profiles — MEMORY SAFE
# For large tables (>1M rows): loads ONLY the FK column needed.
# ---------------------------------------------------------------------------
def compute_fk_cardinality(ds_name: str, db_dir: Path) -> list[dict]:
    """Compute FK-join cardinality profiles."""
    profiles: list[dict] = []
    fk_map = FK_MAP.get(ds_name, {})
    pkey_map = PKEY_MAP.get(ds_name, {})

    for child_table_name, fk_relations in fk_map.items():
        child_path = db_dir / f"{child_table_name}.parquet"
        if not child_path.exists():
            logger.warning(f"  Child table not found: {child_path}")
            continue

        for fk_col, parent_table_name in fk_relations.items():
            parent_path = db_dir / f"{parent_table_name}.parquet"
            parent_pkey = pkey_map.get(parent_table_name)

            logger.info(
                f"  FK edge: {child_table_name}.{fk_col} -> {parent_table_name}"
            )

            try:
                # Load ONLY the FK column from child table
                child_fk_series = pd.read_parquet(
                    str(child_path), columns=[fk_col]
                )[fk_col]

                null_count = int(child_fk_series.isna().sum())
                valid_fks = child_fk_series.dropna()
                del child_fk_series
                gc.collect()

                if len(valid_fks) == 0:
                    logger.warning(f"    No valid FK values")
                    del valid_fks
                    continue

                counts = valid_fks.value_counts()
                total_refs = int(len(valid_fks))
                del valid_fks
                gc.collect()

                # Count parents without children
                num_zero_children = 0
                if parent_pkey and parent_path.exists():
                    parent_ids = set(
                        pd.read_parquet(str(parent_path), columns=[parent_pkey])[
                            parent_pkey
                        ].unique()
                    )
                    referenced_ids = set(counts.index)
                    num_zero_children = len(parent_ids - referenced_ids)
                    del parent_ids, referenced_ids
                    gc.collect()

                vals = counts.values
                profile = {
                    "edge_type": f"{child_table_name}.{fk_col} -> {parent_table_name}",
                    "child_table": child_table_name,
                    "fk_column": fk_col,
                    "parent_table": parent_table_name,
                    "num_parents_with_children": int(len(counts)),
                    "num_parents_without_children": int(num_zero_children),
                    "total_fk_references": total_refs,
                    "null_fk_count": null_count,
                    "cardinality_stats": {
                        "min": int(vals.min()),
                        "p25": int(np.percentile(vals, 25)),
                        "median": int(np.median(vals)),
                        "mean": round(float(np.mean(vals)), 4),
                        "p75": int(np.percentile(vals, 75)),
                        "p90": int(np.percentile(vals, 90)),
                        "p95": int(np.percentile(vals, 95)),
                        "p99": int(np.percentile(vals, 99)),
                        "max": int(vals.max()),
                        "std": round(float(np.std(vals)), 4),
                    },
                }
                profiles.append(profile)
                cs = profile["cardinality_stats"]
                logger.info(
                    f"    median={cs['median']}, p99={cs['p99']}, max={cs['max']}"
                )

                del counts, vals
                gc.collect()

            except Exception:
                logger.exception(
                    f"    Failed: {child_table_name}.{fk_col} -> {parent_table_name}"
                )
                continue

    return profiles


# ---------------------------------------------------------------------------
# Task split statistics
# ---------------------------------------------------------------------------
def extract_task_stats(
    ds_name: str, task_name: str, task_dir: Path, task_meta: dict
) -> dict:
    """Extract train/val/test split statistics from task parquet files."""
    stats: dict = {"dataset": ds_name, "task": task_name, "splits": {}}
    target_col = task_meta["target_col"]
    entity_col = task_meta["entity_col"]

    for split_name in ["train", "val", "test"]:
        split_path = task_dir / f"{split_name}.parquet"
        if not split_path.exists():
            stats["splits"][split_name] = {"error": f"Not found: {split_path}"}
            continue

        try:
            df = pd.read_parquet(str(split_path))
            split_info: dict = {
                "num_rows": len(df),
                "num_unique_entities": (
                    int(df[entity_col].nunique()) if entity_col in df.columns else 0
                ),
                "columns": list(df.columns),
            }

            if target_col in df.columns and not df[target_col].isna().all():
                targets = df[target_col].dropna()
                if len(targets) > 0:
                    split_info["target_stats"] = {
                        "count": int(len(targets)),
                        "mean": round(float(targets.mean()), 6),
                        "std": round(float(targets.std()), 6) if len(targets) > 1 else 0.0,
                        "min": float(targets.min()),
                        "p25": float(targets.quantile(0.25)),
                        "median": float(targets.median()),
                        "p75": float(targets.quantile(0.75)),
                        "max": float(targets.max()),
                    }
                else:
                    split_info["target_stats"] = "no_valid_targets"
            else:
                split_info["target_stats"] = "masked_or_unavailable"

            stats["splits"][split_name] = split_info
            del df
            gc.collect()
        except Exception:
            logger.exception(f"  Failed loading {split_path}")
            stats["splits"][split_name] = {"error": str(split_path)}

    stats["task_type"] = "regression"
    stats["target_col"] = target_col
    stats["entity_col"] = entity_col
    stats["entity_table"] = task_meta["entity_table"]
    stats["timedelta_days"] = task_meta["timedelta_days"]
    stats["metrics"] = ["mae", "rmse", "r2"]

    return stats


# ---------------------------------------------------------------------------
# Mini dataset statistics (estimate without creating full mini)
# ---------------------------------------------------------------------------
def compute_mini_stats(ds_name: str, db_dir: Path, sample_frac: float = 0.05) -> dict:
    """Estimate mini dataset row counts for each table."""
    mini_stats: dict = {}
    pkey_map = PKEY_MAP.get(ds_name, {})

    for f in sorted(db_dir.glob("*.parquet")):
        table_name = f.stem
        num_rows = pq.read_metadata(str(f)).num_rows
        pkey = pkey_map.get(table_name)

        if pkey:
            n_sample = max(int(num_rows * sample_frac), min(100, num_rows))
        else:
            n_sample = max(int(num_rows * sample_frac), 100)

        mini_stats[table_name] = {
            "original_rows": num_rows,
            "estimated_mini_rows": n_sample,
            "sample_frac": sample_frac,
        }

    return mini_stats


# ---------------------------------------------------------------------------
# JSON-safe value conversion
# ---------------------------------------------------------------------------
def _safe_val(val):
    """Convert a value to JSON-serializable type."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    try:
        if pd.isna(val):
            return None
    except (ValueError, TypeError):
        pass
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        v = float(val)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_safe_val(v) for v in val]
    if isinstance(val, (int, float, str, bool)):
        return val
    return str(val)


# ---------------------------------------------------------------------------
# Extract examples — join entity features with task split targets
# ---------------------------------------------------------------------------
def extract_examples(
    ds_name: str,
    task_name: str,
    task_dir: Path,
    db_dir: Path,
    task_meta: dict,
    max_per_split: int = 5000,
) -> list[dict]:
    """Each example = one task data point: entity features → target value."""
    examples: list[dict] = []
    entity_col = task_meta["entity_col"]
    target_col = task_meta["target_col"]
    entity_table = task_meta["entity_table"]

    # --- Load entity table (skip large text columns) ---
    entity_path = db_dir / f"{entity_table}.parquet"
    skip = SKIP_COLS.get(entity_table, set())
    all_cols = pq.read_schema(str(entity_path)).names
    cols_to_load = [c for c in all_cols if c not in skip]

    logger.info(
        f"  Loading entity table '{entity_table}' cols={cols_to_load} (skipping {skip})"
    )
    entity_df = pd.read_parquet(str(entity_path), columns=cols_to_load)
    entity_pkey = PKEY_MAP[ds_name][entity_table]
    feature_cols = [c for c in cols_to_load if c != entity_pkey]
    logger.info(f"  Entity table loaded: {len(entity_df):,} rows, features={feature_cols}")

    # --- Process each split ---
    fold_map = {"train": 0, "val": 1, "test": 2}

    for split_name in ["train", "val", "test"]:
        split_path = task_dir / f"{split_name}.parquet"
        if not split_path.exists():
            logger.warning(f"  Split not found: {split_path}")
            continue

        task_df = pd.read_parquet(str(split_path))
        n_rows = len(task_df)

        if n_rows > max_per_split:
            task_df = task_df.sample(n=max_per_split, random_state=42)
            logger.info(f"  Sampled {max_per_split:,}/{n_rows:,} for {split_name}")

        # Merge entity features into task rows
        merged = task_df.merge(
            entity_df,
            left_on=entity_col,
            right_on=entity_pkey,
            how="left",
            suffixes=("", "_entity"),
        )

        # Determine input feature column names (everything except target)
        # Include entity features + task timestamp columns
        task_feature_cols = [c for c in task_df.columns if c != target_col]
        all_feature_cols = task_feature_cols + feature_cols
        # De-duplicate (entity_col may appear in both)
        seen = set()
        unique_feature_cols = []
        for c in all_feature_cols:
            if c not in seen and c in merged.columns:
                seen.add(c)
                unique_feature_cols.append(c)

        # Convert to list of dicts (much faster than iterrows)
        records = merged.to_dict("records")

        for record in records:
            features = {}
            for col in unique_feature_cols:
                features[col] = _safe_val(record.get(col))

            input_str = json.dumps(features, default=str)

            target_val = record.get(target_col)
            if target_val is None or (isinstance(target_val, float) and math.isnan(target_val)):
                output_str = "masked"
            else:
                output_str = str(round(float(target_val), 6))

            example = {
                "input": input_str,
                "output": output_str,
                "metadata_fold": fold_map[split_name],
                "metadata_task_type": "regression",
                "metadata_entity_col": entity_col,
                "metadata_target_col": target_col,
                "metadata_row_index": int(record.get(entity_col, 0)),
                "metadata_feature_names": unique_feature_cols,
            }
            examples.append(example)

        del task_df, merged, records
        gc.collect()

    del entity_df
    gc.collect()

    logger.info(f"  Extracted {len(examples):,} examples for {ds_name}/{task_name}")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@logger.catch
def main():
    start_time = time.time()

    output_datasets: list[dict] = []
    all_task_metadata: list[dict] = []

    for ds_name, task_name in DATASETS:
        t0 = time.time()
        logger.info(f"{'=' * 60}")
        logger.info(f"Processing {ds_name}/{task_name}")
        logger.info(f"{'=' * 60}")

        db_dir = CACHE_DIR / ds_name / "db"
        task_dir = CACHE_DIR / ds_name / "tasks" / task_name
        tmeta = TASK_META[(ds_name, task_name)]

        if not db_dir.exists():
            logger.error(f"  DB directory not found: {db_dir}")
            continue
        if not task_dir.exists():
            logger.error(f"  Task directory not found: {task_dir}")
            continue

        # --- Memory check ---
        mem_used_gb = 0.0
        try:
            v = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip())
            mem_used_gb = v / 1e9
        except (FileNotFoundError, ValueError):
            pass
        logger.info(f"  Memory used: {mem_used_gb:.1f} GB / {TOTAL_RAM_GB:.1f} GB")

        # Step 1: Schema
        logger.info("  Step 1: Extracting schema (metadata only)...")
        schema = extract_schema(ds_name, db_dir)
        total_rows = sum(t["num_rows"] for t in schema["tables"].values())
        logger.info(
            f"  Schema: {len(schema['tables'])} tables, {total_rows:,} total rows"
        )

        # Step 2: FK cardinality
        logger.info("  Step 2: Computing FK cardinality profiles...")
        fk_profiles = compute_fk_cardinality(ds_name, db_dir)
        logger.info(f"  FK profiles: {len(fk_profiles)} edges computed")

        # Step 3: Task split stats
        logger.info("  Step 3: Extracting task split statistics...")
        task_stats = extract_task_stats(ds_name, task_name, task_dir, tmeta)
        for split, info in task_stats["splits"].items():
            if isinstance(info, dict) and "num_rows" in info:
                logger.info(
                    f"    {split}: {info['num_rows']:,} rows, "
                    f"{info.get('num_unique_entities', '?')} unique entities"
                )

        # Step 4: Mini stats
        logger.info("  Step 4: Computing mini dataset statistics...")
        mini_stats = compute_mini_stats(ds_name, db_dir)

        # Step 5: Examples
        logger.info("  Step 5: Extracting examples...")
        examples = extract_examples(ds_name, task_name, task_dir, db_dir, tmeta)

        # Collect metadata
        task_metadata = {
            "dataset": ds_name,
            "task": task_name,
            "schema": schema,
            "fk_cardinality_profiles": fk_profiles,
            "task_stats": task_stats,
            "mini_dataset_stats": mini_stats,
        }
        all_task_metadata.append(task_metadata)

        output_datasets.append(
            {"dataset": f"{ds_name}/{task_name}", "examples": examples}
        )

        elapsed = time.time() - t0
        logger.info(
            f"  DONE {ds_name}/{task_name} in {elapsed:.1f}s — {len(examples):,} examples"
        )

        # Free memory aggressively
        del schema, fk_profiles, task_stats, mini_stats, examples, task_metadata
        gc.collect()

    # --- Assemble output ---
    logger.info("=" * 60)
    logger.info("Assembling final output...")

    output = {
        "metadata": {
            "description": "RelBench regression task benchmarks with FK-join cardinality profiles",
            "source": "relbench (Stanford SNAP)",
            "paper": "RelBench: A Benchmark for Deep Learning on Relational Databases (NeurIPS 2024)",
            "paper_url": "https://arxiv.org/abs/2407.20060",
            "cache_dir": str(CACHE_DIR),
            "num_tasks_requested": 4,
            "num_tasks_downloaded": len(output_datasets),
            "blocked_tasks": [
                {
                    "dataset": "rel-hm",
                    "task": "item-sales",
                    "reason": "Requires Kaggle credentials for H&M competition data",
                }
            ],
            "tasks_metadata": all_task_metadata,
        },
        "datasets": output_datasets,
    }

    # Save full_data_out.json
    out_path = WS / "full_data_out.json"
    out_text = json.dumps(output, indent=2, default=str)
    out_path.write_text(out_text)
    file_size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved {out_path.name} ({file_size_mb:.1f} MB)")

    total_examples = sum(len(d["examples"]) for d in output_datasets)
    elapsed_total = time.time() - start_time
    logger.info(
        f"DONE: {len(output_datasets)} datasets, "
        f"{total_examples:,} examples in {elapsed_total:.0f}s"
    )


if __name__ == "__main__":
    main()
