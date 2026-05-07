#!/usr/bin/env python3
"""Synthetic Relational Database Testbed for Rank Collapse & RAMA Validation.

Generates 12 conditions (4 cardinality x 3 projection rank regimes) with
ground-truth effective rank, regression targets depending on variance,
and classification targets requiring variance.

Output: data_out.json conforming to exp_sel_data_out schema.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.stats import pearsonr, spearmanr

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path(
    "/ai-inventor/aii_pipeline/runs/"
    "leskovec-predictive-residual-message-passing-v2_sti/"
    "3_invention_loop/iter_1/gen_art/data_id5_it1__opus"
)
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "data.log", rotation="30 MB", level="DEBUG")


# ── Hardware detection ───────────────────────────────────────────────────────
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


def _container_ram_gb() -> float | None:
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

# Memory limit: 14 GB budget (well within 29 GB container)
RAM_BUDGET = int(14 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Generation constants ─────────────────────────────────────────────────────
MASTER_SEED = 42
PARENT_DIM = 8
CHILD_DIM = 16
NOISE_STD = 0.1
DATASET_NAME = "synthetic_relational_rank_collapse_testbed"

CARDINALITY_REGIMES = [
    ("low", 5, 500),
    ("medium", 50, 500),
    ("high", 500, 200),
    ("extreme", 2000, 50),
]
PROJECTION_RANKS = [2, 8, 16]

CONDITIONS = []
for ci, (card_name, n_children, n_parents) in enumerate(CARDINALITY_REGIMES):
    for ri, rank in enumerate(PROJECTION_RANKS):
        idx = ci * len(PROJECTION_RANKS) + ri
        CONDITIONS.append({
            "condition_name": f"{card_name}_rank{rank}",
            "cardinality_regime": card_name,
            "num_children": n_children,
            "num_parents": n_parents,
            "projection_rank": rank,
            "condition_index": idx,
        })

# ── Helpers ──────────────────────────────────────────────────────────────────
def round_list(arr: np.ndarray, decimals: int = 4) -> list[float]:
    return [round(float(x), decimals) for x in arr]


def round_nested_list(arr: np.ndarray, decimals: int = 4) -> list[list[float]]:
    return [[round(float(x), decimals) for x in row] for row in arr]


def truncate_str(s: str, max_len: int = 200) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s


# ── Per-condition generator (runs in subprocess) ─────────────────────────────
def generate_condition(cond: dict) -> list[dict]:
    """Generate all parent-child data for one condition."""
    idx = cond["condition_index"]
    rank = cond["projection_rank"]
    n_children = cond["num_children"]
    n_parents = cond["num_parents"]
    card_name = cond["cardinality_regime"]
    cond_name = cond["condition_name"]

    rng = np.random.default_rng(MASTER_SEED + idx)

    # Parent features — shape (num_parents, 8)
    parent_features = rng.standard_normal((n_parents, PARENT_DIM))

    # Projection matrix A — orthonormal columns, shape (16, rank)
    raw = rng.standard_normal((CHILD_DIM, rank))
    A, _ = np.linalg.qr(raw)

    # Parent-to-child bias matrix B — shape (16, 8)
    B = rng.standard_normal((CHILD_DIM, PARENT_DIM)) * 0.5

    # Weight vectors for targets
    w_mean = rng.standard_normal(CHILD_DIM)
    w_var = np.abs(rng.standard_normal(CHILD_DIM)) * 4.0  # all-positive, amplified
    w_cross = rng.standard_normal(CHILD_DIM) * 1.5

    # Parent-dependent variance scaling
    var_scale_weights = rng.standard_normal(PARENT_DIM) * 0.5

    parent_data = []
    for p in range(n_parents):
        scale_p = float(np.exp(np.dot(var_scale_weights, parent_features[p])))
        scale_p = np.clip(scale_p, 0.3, 3.0)

        z = rng.standard_normal((n_children, rank))
        structured = (scale_p * z) @ A.T
        bias = B @ parent_features[p]
        noise = rng.standard_normal((n_children, CHILD_DIM)) * NOISE_STD
        child_feats = structured + bias + noise

        mu = child_feats.mean(axis=0)
        var_per_dim = child_feats.var(axis=0)

        # SVD-based effective rank
        centered = child_feats - mu
        cov = (centered.T @ centered) / n_children
        sigmas = np.linalg.svd(cov, compute_uv=False)
        sigmas_pos = np.maximum(sigmas, 1e-12)
        p_svd = sigmas_pos / sigmas_pos.sum()
        H_svd = -np.sum(p_svd * np.log(p_svd))
        erank_svd = float(np.exp(H_svd))

        # Cheap variance-entropy proxy
        var_pos = np.maximum(var_per_dim, 1e-12)
        p_var = var_pos / var_pos.sum()
        H_var = -np.sum(p_var * np.log(p_var))
        erank_proxy = float(np.exp(H_var)) / CHILD_DIM

        # Regression target
        y_mean_comp = float(np.sum(w_mean * np.sin(mu)))
        y_var_comp = float(np.sum(w_var * np.sqrt(var_per_dim + 0.01)))
        y_cross_comp = float(np.sum(w_cross * mu * var_per_dim))
        y_direct_var = 0.5 * float(np.sum(var_per_dim))
        y = y_mean_comp + y_var_comp + y_cross_comp + y_direct_var + float(rng.normal(0, 0.1))

        total_var = float(np.sum(var_per_dim))

        parent_data.append({
            "parent_features": parent_features[p],
            "child_feats": child_feats,
            "mu": mu,
            "var_per_dim": var_per_dim,
            "erank_svd": erank_svd,
            "erank_proxy": erank_proxy,
            "regression_target": y,
            "total_var": total_var,
            "parent_id": p,
        })

    # Classification target — median split with fuzzy boundary
    total_vars = np.array([pd["total_var"] for pd in parent_data])
    median_var = float(np.median(total_vars))
    distances_to_median = np.abs(total_vars - median_var)
    dist_20th_pct = float(np.percentile(distances_to_median, 20))

    for pd in parent_data:
        tv = pd["total_var"]
        base_label = 1 if tv > median_var else 0
        dist = abs(tv - median_var)
        if dist <= dist_20th_pct and dist_20th_pct > 0:
            mu_sum = float(np.sum(pd["mu"][:4]))
            logit = -2.0 * dist / max(dist_20th_pct, 1e-8) + 0.3 * mu_sum
            flip_prob = 1.0 / (1.0 + np.exp(-logit))
            if rng.random() < flip_prob * 0.3:
                base_label = 1 - base_label
        pd["classification_target"] = base_label

    # Train/val/test split (60/20/20)
    indices = np.arange(n_parents)
    rng.shuffle(indices)
    n_train = int(0.6 * n_parents)
    n_val = int(0.2 * n_parents)
    folds: dict[int, str] = {}
    for i, idx_val in enumerate(indices):
        if i < n_train:
            folds[int(idx_val)] = "train"
        elif i < n_train + n_val:
            folds[int(idx_val)] = "val"
        else:
            folds[int(idx_val)] = "test"

    # Assemble schema-compliant rows
    rows = []
    for pd in parent_data:
        pid = pd["parent_id"]
        input_obj = {
            "parent_features": round_list(pd["parent_features"]),
            "child_features": round_nested_list(pd["child_feats"]),
            "num_children": n_children,
        }
        output_obj = {
            "regression_target": round(pd["regression_target"], 4),
            "classification_target": pd["classification_target"],
        }
        row = {
            "input": json.dumps(input_obj, separators=(",", ":")),
            "output": json.dumps(output_obj, separators=(",", ":")),
            "metadata_fold": folds[pid],
            "metadata_condition": cond_name,
            "metadata_cardinality_regime": card_name,
            "metadata_cardinality": n_children,
            "metadata_projection_rank": rank,
            "metadata_parent_id": pid,
            "metadata_erank_svd": round(pd["erank_svd"], 4),
            "metadata_erank_proxy": round(pd["erank_proxy"], 4),
            "metadata_child_mean": round_list(pd["mu"]),
            "metadata_child_variance": round_list(pd["var_per_dim"]),
        }
        rows.append(row)

    del parent_data
    gc.collect()
    return rows


# ── Validation ───────────────────────────────────────────────────────────────
def run_validations(rows: list[dict]) -> dict:
    """Run all 7 validation checks."""
    results: dict = {}
    all_pass = True
    conditions = sorted(set(r["metadata_condition"] for r in rows))

    # 1. Rank correlation
    logger.info("Check 1: Rank correlation (projection_rank vs erank_svd)")
    for card in ["low", "medium", "high", "extreme"]:
        card_rows = [r for r in rows if r["metadata_cardinality_regime"] == card]
        if len(card_rows) < 3:
            continue
        corr, pval = spearmanr(
            [r["metadata_projection_rank"] for r in card_rows],
            [r["metadata_erank_svd"] for r in card_rows],
        )
        results[f"rank_corr_{card}"] = {"spearman": round(corr, 4), "pval": round(float(pval), 6)}
        ok = corr > 0
        if not ok:
            all_pass = False
        logger.info(f"  {card}: r={corr:.4f} [{'PASS' if ok else 'FAIL'}]")

    # 2. Proxy accuracy
    logger.info("Check 2: Proxy accuracy (erank_svd vs erank_proxy)")
    corr, pval = pearsonr(
        [r["metadata_erank_svd"] for r in rows],
        [r["metadata_erank_proxy"] for r in rows],
    )
    results["proxy_pearson"] = {"pearson": round(corr, 4), "pval": round(float(pval), 6)}
    ok = corr > 0.7
    if not ok:
        all_pass = False
    logger.info(f"  Pearson r={corr:.4f} [{'PASS' if ok else 'FAIL'}]")

    # 3. Target-variance dependency
    logger.info("Check 3: Target-variance dependency")
    for cn in conditions:
        cr = [r for r in rows if r["metadata_condition"] == cn]
        tvars = [sum(r["metadata_child_variance"]) for r in cr]
        tgts = [json.loads(r["output"])["regression_target"] for r in cr]
        corr, pval = pearsonr(tvars, tgts)
        results[f"var_target_corr_{cn}"] = {"pearson": round(corr, 4), "pval": round(float(pval), 6)}
        ok = abs(corr) > 0.2
        if not ok:
            all_pass = False
        logger.info(f"  {cn}: r={corr:.4f} [{'PASS' if ok else 'FAIL'}]")

    # 4. Classification balance
    logger.info("Check 4: Classification balance")
    for cn in conditions:
        cr = [r for r in rows if r["metadata_condition"] == cn]
        labels = [json.loads(r["output"])["classification_target"] for r in cr]
        pf = sum(labels) / len(labels)
        results[f"class_balance_{cn}"] = round(pf, 4)
        ok = 0.4 <= pf <= 0.6
        if not ok:
            all_pass = False
        logger.info(f"  {cn}: {pf:.4f} [{'PASS' if ok else 'FAIL'}]")

    # 5. Condition coverage
    logger.info("Check 5: Condition coverage")
    expected = {f"{c}_rank{r}" for c, _, _ in CARDINALITY_REGIMES for r in PROJECTION_RANKS}
    missing = expected - set(conditions)
    results["condition_coverage"] = {
        "expected": len(expected), "found": len(conditions), "missing": sorted(missing)
    }
    ok = len(missing) == 0
    if not ok:
        all_pass = False
    logger.info(f"  {len(expected)} expected, {len(conditions)} found [{'PASS' if ok else 'FAIL'}]")

    # 6. Split ratios
    logger.info("Check 6: Split ratios")
    for cn in conditions:
        cr = [r for r in rows if r["metadata_condition"] == cn]
        n = len(cr)
        tr = sum(1 for r in cr if r["metadata_fold"] == "train") / n
        va = sum(1 for r in cr if r["metadata_fold"] == "val") / n
        te = sum(1 for r in cr if r["metadata_fold"] == "test") / n
        results[f"split_{cn}"] = [round(tr, 2), round(va, 2), round(te, 2)]
        ok = all(0.15 <= x <= 0.65 for x in [tr, va, te])
        if not ok:
            all_pass = False
        logger.info(f"  {cn}: {tr:.2f}/{va:.2f}/{te:.2f} [{'PASS' if ok else 'FAIL'}]")

    # 7. NaN/Inf check
    logger.info("Check 7: NaN/Inf check")
    bad = 0
    for r in rows:
        for k in ["metadata_erank_svd", "metadata_erank_proxy"]:
            if math.isnan(r[k]) or math.isinf(r[k]):
                bad += 1
        for lst in [r["metadata_child_mean"], r["metadata_child_variance"]]:
            for v in lst:
                if math.isnan(v) or math.isinf(v):
                    bad += 1
        out = json.loads(r["output"])
        if math.isnan(out["regression_target"]) or math.isinf(out["regression_target"]):
            bad += 1
    results["bad_values"] = bad
    ok = bad == 0
    if not ok:
        all_pass = False
    logger.info(f"  Bad values: {bad} [{'PASS' if ok else 'FAIL'}]")

    results["all_pass"] = all_pass
    return results


# ── Main ─────────────────────────────────────────────────────────────────────
@logger.catch
def main() -> None:
    t0 = time.time()
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM")
    logger.info(f"Generating {len(CONDITIONS)} conditions in parallel...")

    # ── Generate all conditions ──
    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        futures = {
            executor.submit(generate_condition, c): c["condition_name"]
            for c in CONDITIONS
        }
        for f in as_completed(futures):
            cond_name = futures[f]
            try:
                rows = f.result()
                all_rows.extend(rows)
                logger.info(f"  {cond_name}: {len(rows)} rows")
            except Exception:
                logger.exception(f"Failed generating {cond_name}")
                raise

    logger.info(f"Total: {len(all_rows)} rows (expected 3750)")

    # ── Validate ──
    logger.info("Running validation checks...")
    validation_results = run_validations(all_rows)
    if validation_results["all_pass"]:
        logger.info("ALL CHECKS PASSED")
    else:
        logger.warning("SOME CHECKS FAILED")

    # ── Build metadata ──
    generation_metadata = {
        "master_seed": MASTER_SEED,
        "noise_std": NOISE_STD,
        "parent_dim": PARENT_DIM,
        "child_dim": CHILD_DIM,
        "conditions": [
            {
                "name": c["condition_name"],
                "cardinality_regime": c["cardinality_regime"],
                "num_children": c["num_children"],
                "num_parents": c["num_parents"],
                "projection_rank": c["projection_rank"],
            }
            for c in CONDITIONS
        ],
        "target_design": (
            "Regression: sum(w_mean*sin(mu)) + sum(w_var*sqrt(var+0.01)) "
            "+ sum(w_cross*mu*var) + 0.5*total_var + noise(0,0.1). "
            "Classification: total_var > median with fuzzy boundary near median."
        ),
        "erank_formula": (
            "SVD: exp(-sum(p_i * log(p_i))) where p_i = sigma_i / sum(sigmas). "
            "Proxy: exp(-sum(q_d * log(q_d))) / D where q_d = var_d / sum(vars)"
        ),
        "validation_results": validation_results,
    }

    # ── Save full_data_out.json ──
    output = {
        "metadata": generation_metadata,
        "datasets": [{"dataset": DATASET_NAME, "examples": all_rows}],
    }
    full_path = WORKSPACE / "full_data_out.json"
    logger.info(f"Writing {full_path}...")
    full_path.write_text(json.dumps(output, separators=(",", ":")))
    size_mb = full_path.stat().st_size / (1024 * 1024)
    logger.info(f"  full_data_out.json: {size_mb:.1f}MB ({len(all_rows)} examples)")

    # ── Save mini_data_out.json (first 3 examples) ──
    mini_output = {
        "metadata": generation_metadata,
        "datasets": [{"dataset": DATASET_NAME, "examples": all_rows[:3]}],
    }
    mini_path = WORKSPACE / "mini_data_out.json"
    mini_path.write_text(json.dumps(mini_output, separators=(",", ":")))
    logger.info(f"  mini_data_out.json: {mini_path.stat().st_size / 1024:.1f}KB (3 examples)")

    # ── Save preview_data_out.json (first 3 examples, strings truncated) ──
    preview_examples = []
    for ex in all_rows[:3]:
        pex = {k: (truncate_str(v) if isinstance(v, str) else v) for k, v in ex.items()}
        preview_examples.append(pex)
    preview_output = {
        "metadata": generation_metadata,
        "datasets": [{"dataset": DATASET_NAME, "examples": preview_examples}],
    }
    preview_path = WORKSPACE / "preview_data_out.json"
    preview_path.write_text(json.dumps(preview_output, indent=2))
    logger.info(f"  preview_data_out.json: {preview_path.stat().st_size / 1024:.1f}KB (3 examples, truncated)")

    logger.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
