#!/usr/bin/env python3
"""Regenerate method_out.json from saved per-run results without re-training."""
import json
import sys
from pathlib import Path

# Add workspace to path so we can import from method.py
WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS))

from method import (
    RESULTS_DIR, WS as METHOD_WS, build_statistical_analysis,
    generate_output, logger
)

def main():
    # Load all saved per-run results
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        logger.error("No result files found in results/")
        sys.exit(1)

    all_results = []
    for rf in result_files:
        try:
            r = json.loads(rf.read_text())
            all_results.append(r)
            logger.info(f"Loaded {rf.name}: {r['method']} seed={r['seed']} MAE={r['test_mae']:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load {rf.name}: {e}")

    logger.info(f"Loaded {len(all_results)} results from {len(result_files)} files")

    # Rebuild stats
    stats = build_statistical_analysis(all_results)

    # Generate output with predict_* fields
    output = generate_output(all_results, stats)

    # Save
    out_path = METHOD_WS / "method_out.json"
    out_text = json.dumps(output, indent=2, default=str)
    out_path.write_text(out_text)
    file_size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved method_out.json ({file_size_mb:.1f} MB)")

    # Quick validation: check predict fields exist
    for ds in output["datasets"][:1]:
        ex = ds["examples"][0]
        predict_keys = [k for k in ex if k.startswith("predict_")]
        logger.info(f"Dataset '{ds['dataset']}' example keys with predict_: {predict_keys}")
        if not predict_keys:
            logger.error("NO predict_* fields found!")
            sys.exit(1)

    logger.info("Regeneration complete.")


if __name__ == "__main__":
    main()
