#!/usr/bin/env python3
"""Transform raw RelBench data into per-row examples for exp_sel_data_out.json schema.

Reads raw_data_out.json (collected by background task) and expands each mini_data
row into a separate example with structured input (relational schema context +
entity query) and output (target label/value). Groups by dataset-task pair.
"""

import json
import math
import os
import resource
import sys
from pathlib import Path

from loguru import logger

# --- Logging ---
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "data.log"), rotation="30 MB", level="DEBUG")

# --- Hardware detection ---
def _container_ram_gb():
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.5, 14) * 1e9)  # 50% of container, max 14GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET/1e9:.1f} GB (container: {TOTAL_RAM_GB:.1f} GB)")

WORKSPACE = Path(__file__).parent
RAW_DATA_PATH = WORKSPACE / "raw_data_out.json"
OUTPUT_PATH = WORKSPACE / "full_data_out.json"

# The 4 chosen dataset-task pairs (best for FK cardinality analysis):
# 1. rel-f1/driver-dnf: small baseline, moderate cardinality (9 tables, 13 FK edges)
# 2. rel-stack/user-engagement: medium, self-referential FKs (7 tables, 11 FK edges)
# 3. rel-hm/user-churn: large (16.6M rows), high cardinality (3 tables, 2 FK edges)
# 4. rel-avito/ad-ctr: extreme cardinality (mean=114,625), catastrophic PRMP failure
SELECTED_TASKS = {
    "rel-f1": ["driver-dnf"],
    "rel-stack": ["user-engagement"],
    "rel-hm": ["user-churn"],
    "rel-avito": ["ad-ctr"],
}


def build_schema_summary(schema: dict) -> str:
    """Build a concise text summary of the relational schema."""
    lines = []
    for tname, tinfo in schema.items():
        pk = tinfo.get("pkey_col", "none")
        fks = tinfo.get("fkey_col_to_pkey_table", {})
        fk_strs = [f"{col}->{tbl}" for col, tbl in fks.items()] if fks else ["(root)"]
        lines.append(f"  {tname} ({tinfo['num_rows']} rows, {tinfo['num_columns']} cols) "
                     f"PK={pk} FKs=[{', '.join(fk_strs)}]")
    return "\n".join(lines)


def build_fk_summary(fk_edges: list) -> str:
    """Build a concise text summary of FK cardinality profiles."""
    lines = []
    for edge in fk_edges:
        flag = " [HIGH-CARDINALITY]" if edge.get("high_cardinality_flag") else ""
        lines.append(
            f"  {edge['edge']}: mean={edge['mean']:.1f} median={edge['median']:.1f} "
            f"p95={edge['p95']:.1f} p99={edge['p99']:.1f} max={edge['max']}{flag}"
        )
    return "\n".join(lines)


def build_aggregate_summary(ds: dict) -> str:
    """Build aggregate FK profile text."""
    agg = ds.get("aggregate_fk_profile", {})
    if not agg:
        return "No aggregate profile available."
    return (
        f"Total FK edges: {agg.get('total_fk_edges', '?')}, "
        f"Max mean cardinality: {agg.get('max_mean_cardinality', '?')}, "
        f"Max max cardinality: {agg.get('max_max_cardinality', '?')}, "
        f"High-cardinality edges: {agg.get('num_high_cardinality_edges', '?')}, "
        f"Danger score: {agg.get('danger_score', '?')}"
    )


def format_row_value(val) -> str:
    """Format a single value for display (handle None, bool, float, etc.)."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return "null"
        # Keep reasonable precision
        if val == int(val):
            return str(int(val))
        return f"{val:.6g}"
    return str(val)


def build_example_input(
    ds_name: str,
    task_info: dict,
    row: dict,
    schema_summary: str,
    fk_summary: str,
    agg_summary: str,
    ds_meta: dict,
) -> str:
    """Build structured input string for one prediction instance."""

    entity_col = task_info["entity_col"]
    target_col = task_info["target_col"]
    timestamp_col = None
    for k in row:
        if k not in (entity_col, target_col):
            timestamp_col = k
            break

    # Build entity query section
    entity_id = format_row_value(row.get(entity_col))
    timestamp = format_row_value(row.get(timestamp_col)) if timestamp_col else "N/A"

    # Build the structured input
    input_text = (
        f"=== Relational Prediction Task ===\n"
        f"Dataset: {ds_name}\n"
        f"Task: {task_info['task_name']} ({task_info['task_type']})\n"
        f"Predict: {target_col} for entity {entity_col}\n"
        f"Metrics: {', '.join(task_info.get('metrics', []))}\n"
        f"Timedelta: {task_info.get('timedelta_days', '?')} days\n"
        f"\n"
        f"=== Query Instance ===\n"
        f"{entity_col}: {entity_id}\n"
        f"Timestamp: {timestamp}\n"
        f"\n"
        f"=== Relational Schema ({ds_meta.get('num_tables', '?')} tables, "
        f"{ds_meta.get('total_rows', '?')} total rows) ===\n"
        f"{schema_summary}\n"
        f"\n"
        f"=== FK Cardinality Profiles ===\n"
        f"{fk_summary}\n"
        f"\n"
        f"=== Aggregate FK Profile ===\n"
        f"{agg_summary}"
    )
    return input_text


@logger.catch
def main():
    logger.info(f"Loading raw data from {RAW_DATA_PATH}")
    raw = json.loads(RAW_DATA_PATH.read_text())
    datasets_raw = raw["datasets"]
    cross_comparison = raw.get("cross_dataset_comparison", {})
    logger.info(f"Found {len(datasets_raw)} datasets in raw data")

    output_datasets = []
    total_examples = 0

    for ds in datasets_raw:
        ds_name = ds["dataset_name"]
        schema = ds.get("schema", {})
        fk_edges = ds.get("fk_edges", [])
        tasks = ds.get("tasks", [])
        agg = ds.get("aggregate_fk_profile", {})

        schema_summary = build_schema_summary(schema)
        fk_summary = build_fk_summary(fk_edges)
        agg_summary = build_aggregate_summary(ds)

        ds_meta = {
            "num_tables": ds.get("num_tables"),
            "total_rows": ds.get("total_rows"),
            "total_columns": ds.get("total_columns"),
        }

        # Filter to selected datasets/tasks only
        if ds_name not in SELECTED_TASKS:
            logger.info(f"Skipping dataset: {ds_name} (not in selected)")
            continue

        logger.info(f"Processing dataset: {ds_name} ({len(tasks)} tasks)")

        for task_info in tasks:
            task_name = task_info["task_name"]
            if task_name not in SELECTED_TASKS[ds_name]:
                logger.info(f"  Skipping task: {task_name} (not in selected)")
                continue
            task_type = task_info.get("task_type", "unknown")
            mini_data = task_info.get("mini_data", [])
            splits = task_info.get("splits", {})
            label_dist = task_info.get("label_distribution", {})

            if not mini_data:
                logger.warning(f"  No mini_data for {ds_name}/{task_name}, skipping")
                continue

            # Dataset group name: combine dataset + task for unique identification
            group_name = f"{ds_name}__{task_name}"

            logger.info(f"  Task: {task_name} ({task_type}), {len(mini_data)} rows")

            examples = []
            entity_col = task_info.get("entity_col", "")
            target_col = task_info.get("target_col", "")

            for row_idx, row in enumerate(mini_data):
                try:
                    # Build input
                    input_text = build_example_input(
                        ds_name=ds_name,
                        task_info=task_info,
                        row=row,
                        schema_summary=schema_summary,
                        fk_summary=fk_summary,
                        agg_summary=agg_summary,
                        ds_meta=ds_meta,
                    )

                    # Build output: the target value as string
                    target_val = row.get(target_col)
                    output_text = format_row_value(target_val)

                    # Build metadata
                    example = {
                        "input": input_text,
                        "output": output_text,
                        "metadata_dataset_name": ds_name,
                        "metadata_task_name": task_name,
                        "metadata_task_type": task_type,
                        "metadata_entity_col": entity_col,
                        "metadata_target_col": target_col,
                        "metadata_entity_id": format_row_value(row.get(entity_col)),
                        "metadata_row_index": row_idx,
                        "metadata_num_tables": ds.get("num_tables", 0),
                        "metadata_total_rows": ds.get("total_rows", 0),
                        "metadata_num_fk_edges": len(fk_edges),
                        "metadata_num_high_cardinality_edges": sum(
                            1 for e in fk_edges if e.get("high_cardinality_flag")
                        ),
                        "metadata_danger_score": agg.get("danger_score", 0.0),
                        "metadata_split_train": splits.get("train", 0),
                        "metadata_split_val": splits.get("val", 0),
                        "metadata_split_test": splits.get("test", 0),
                    }

                    # Add label distribution metadata
                    if task_type == "binary_classification":
                        pos_frac = label_dist.get("positive_fraction")
                        if pos_frac is not None:
                            example["metadata_positive_fraction"] = round(pos_frac, 4)
                        example["metadata_n_classes"] = 2
                    elif task_type == "regression":
                        for stat in ["mean", "std", "min", "max"]:
                            val = label_dist.get(stat)
                            if val is not None:
                                example[f"metadata_target_{stat}"] = round(val, 6) if isinstance(val, float) else val

                    # Add metrics list
                    metrics = task_info.get("metrics", [])
                    if metrics:
                        example["metadata_metrics"] = metrics

                    # Add timedelta
                    td = task_info.get("timedelta_days")
                    if td is not None:
                        example["metadata_timedelta_days"] = td

                    examples.append(example)

                except Exception:
                    logger.exception(f"  Failed on row {row_idx} of {group_name}")
                    continue

            output_datasets.append({
                "dataset": group_name,
                "examples": examples,
            })
            total_examples += len(examples)
            logger.info(f"  -> {len(examples)} examples for {group_name}")

    # Build final output
    output = {
        "datasets": output_datasets,
        "metadata": {
            "description": (
                "RelBench Classification + Avito Stress-Test Tasks with FK-Join Cardinality Profiles. "
                "Each example represents one prediction instance (entity at a timestamp) in a relational "
                "database. Input contains the relational schema, FK cardinality profiles, and entity query. "
                "Output is the target label/value."
            ),
            "source": "relbench (Stanford SNAP)",
            "num_datasets": len(output_datasets),
            "total_examples": total_examples,
            "cross_dataset_comparison": cross_comparison,
        },
    }

    logger.info(f"Writing {total_examples} total examples across {len(output_datasets)} datasets to {OUTPUT_PATH}")
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str))
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.2f} MB")

    # Summary
    logger.info("=== Summary ===")
    for d in output_datasets:
        logger.info(f"  {d['dataset']}: {len(d['examples'])} examples")
    logger.info(f"  Total: {total_examples} examples across {len(output_datasets)} dataset-task pairs")


if __name__ == "__main__":
    main()
