#!/usr/bin/env python3
"""Reformat raw RelBench profiling output into exp_sel_data_out schema.

Schema requires:
{
  "datasets": [
    {
      "dataset": "string",
      "examples": [
        {"input": "string", "output": "string", "metadata_*": ...}
      ]
    }
  ]
}

Each RelBench dataset becomes one dataset entry.
Each task within that dataset becomes one example.
The input contains the task description + schema context.
The output contains the profiled data (FK cardinality, label dist, mini data).
"""

import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path(__file__).parent


def format_schema_summary(ds: dict) -> str:
    """Create a readable schema summary string."""
    lines = []
    lines.append(f"Relational Database: {ds['dataset_name']}")
    lines.append(f"Tables: {ds.get('num_tables', '?')}, Total rows: {ds.get('total_rows', '?'):,}, Total columns: {ds.get('total_columns', '?')}")
    if ds.get("synthetic_proxy"):
        lines.append(f"NOTE: {ds.get('synthetic_note', 'Synthetic proxy data')}")
    lines.append("")
    lines.append("Schema:")
    schema = ds.get("schema", {})
    for tname, tinfo in schema.items():
        pk = tinfo.get("pkey_col", "None")
        fks = tinfo.get("fkey_col_to_pkey_table", {})
        fk_str = ", ".join(f"{k}->{v}" for k, v in fks.items()) if fks else "none"
        lines.append(f"  {tname}: {tinfo.get('num_rows', '?'):,} rows, {tinfo.get('num_columns', '?')} cols, PK={pk}, FKs=[{fk_str}]")
    lines.append("")
    lines.append("FK Cardinality Profiles:")
    for edge in ds.get("fk_edges", []):
        flag = " [HIGH CARDINALITY]" if edge.get("high_cardinality_flag") else ""
        lines.append(f"  {edge['edge']}: mean={edge['mean']:.1f}, median={edge['median']:.1f}, "
                      f"p95={edge['p95']:.1f}, p99={edge['p99']:.1f}, max={edge['max']}, "
                      f"skew_ratio={edge.get('skewness_ratio', 0):.1f}{flag}")
    agg = ds.get("aggregate_fk_profile", {})
    if agg:
        lines.append(f"\nAggregate: {agg.get('total_fk_edges', 0)} FK edges, "
                      f"max_mean={agg.get('max_mean_cardinality', 0):.1f}, "
                      f"max_max={agg.get('max_max_cardinality', 0)}, "
                      f"danger_score={agg.get('cardinality_danger_score', 0):.1f}, "
                      f"high_card_edges={agg.get('num_high_cardinality_edges', 0)}")
    return "\n".join(lines)


def format_task_input(ds: dict, task: dict) -> str:
    """Create input string for a task example."""
    lines = []
    lines.append(f"Dataset: {ds['dataset_name']}")
    lines.append(f"Task: {task.get('task_name', 'unknown')}")
    lines.append(f"Task type: {task.get('task_type', 'unknown')}")
    lines.append(f"Entity table: {task.get('entity_table', 'unknown')}")
    lines.append(f"Entity column: {task.get('entity_col', 'unknown')}")
    lines.append(f"Target column: {task.get('target_col', 'unknown')}")
    if task.get("metrics"):
        lines.append(f"Metrics: {', '.join(task['metrics'])}")
    if task.get("timedelta_days"):
        lines.append(f"Timedelta: {task['timedelta_days']} days")
    splits = task.get("splits", {})
    if splits:
        lines.append(f"Splits: train={splits.get('train', 0):,}, val={splits.get('val', 0):,}, test={splits.get('test', 0):,}")
    lines.append("")
    lines.append(format_schema_summary(ds))
    return "\n".join(lines)


def format_task_output(ds: dict, task: dict) -> str:
    """Create output string with profiled data."""
    output_data = {
        "dataset_name": ds["dataset_name"],
        "task_name": task.get("task_name"),
        "task_type": task.get("task_type"),
        "entity_table": task.get("entity_table"),
        "entity_col": task.get("entity_col"),
        "target_col": task.get("target_col"),
        "metrics": task.get("metrics"),
        "timedelta_days": task.get("timedelta_days"),
        "splits": task.get("splits"),
        "label_distribution": task.get("label_distribution"),
        "target_stats": task.get("target_stats"),
        "schema": ds.get("schema"),
        "fk_edges": ds.get("fk_edges"),
        "aggregate_fk_profile": ds.get("aggregate_fk_profile"),
        "mini_data": task.get("mini_data", []),
    }
    if ds.get("synthetic_proxy"):
        output_data["synthetic_proxy"] = True
        output_data["synthetic_note"] = ds.get("synthetic_note")
    return json.dumps(output_data, default=str)


def main():
    raw_path = WORKSPACE / "data_out.json"
    if not raw_path.exists():
        logger.error(f"Raw output not found: {raw_path}")
        return

    raw = json.loads(raw_path.read_text())
    logger.info(f"Loaded raw output with {len(raw.get('datasets', []))} datasets")

    formatted = {"datasets": []}

    for ds in raw.get("datasets", []):
        ds_entry = {
            "dataset": ds["dataset_name"],
            "examples": [],
        }

        tasks = ds.get("tasks", [])
        if not tasks:
            # No tasks loaded - create a schema-only example
            ds_entry["examples"].append({
                "input": format_schema_summary(ds),
                "output": json.dumps({
                    "dataset_name": ds["dataset_name"],
                    "schema": ds.get("schema"),
                    "fk_edges": ds.get("fk_edges"),
                    "aggregate_fk_profile": ds.get("aggregate_fk_profile"),
                    "error": ds.get("download_error", "No tasks available"),
                }, default=str),
            })
        else:
            for task in tasks:
                example = {
                    "input": format_task_input(ds, task),
                    "output": format_task_output(ds, task),
                }
                # Add metadata fields
                example["metadata_dataset_name"] = ds["dataset_name"]
                example["metadata_task_name"] = task.get("task_name", "unknown")
                example["metadata_task_type"] = task.get("task_type", "unknown")
                example["metadata_num_tables"] = ds.get("num_tables", 0)
                example["metadata_total_rows"] = ds.get("total_rows", 0)
                example["metadata_num_fk_edges"] = ds.get("aggregate_fk_profile", {}).get("total_fk_edges", 0)
                example["metadata_danger_score"] = ds.get("aggregate_fk_profile", {}).get("cardinality_danger_score", 0)
                if ds.get("synthetic_proxy"):
                    example["metadata_synthetic_proxy"] = True
                ds_entry["examples"].append(example)

        formatted["datasets"].append(ds_entry)
        logger.info(f"  {ds['dataset_name']}: {len(ds_entry['examples'])} examples")

    # Add cross-dataset comparison as metadata
    if raw.get("cross_dataset_comparison"):
        formatted["metadata"] = {
            "cross_dataset_comparison": raw["cross_dataset_comparison"],
            "description": (
                "RelBench Classification + Avito Stress-Test Tasks with FK-Join Cardinality Profiles. "
                "Each dataset entry contains relational schema, FK cardinality profiles, task metadata, "
                "label distributions, and mini data samples. Cross-dataset comparison identifies "
                "structural differences explaining Avito's catastrophic PRMP failure."
            ),
            "source": "relbench (Stanford SNAP)",
            "num_datasets": len(formatted["datasets"]),
            "total_examples": sum(len(d["examples"]) for d in formatted["datasets"]),
        }

    # Save
    out_path = WORKSPACE / "data_out.json"
    out_path.write_text(json.dumps(formatted, indent=2, default=str))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved formatted output: {out_path} ({size_mb:.2f} MB)")
    logger.info(f"Datasets: {len(formatted['datasets'])}, "
                f"Total examples: {sum(len(d['examples']) for d in formatted['datasets'])}")


if __name__ == "__main__":
    main()
