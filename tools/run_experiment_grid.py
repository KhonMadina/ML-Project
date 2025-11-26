#!/usr/bin/env python3
"""Run grids of experiments defined in YAML configs.

This tool operationalizes Step 2 (Define and Implement Experimental Program):
- Reads an experiment YAML with a `base` section and a `grid` of parameters.
- Generates the Cartesian product of all grid parameters.
- For each combination, constructs a CLI call to the target training script
  (e.g., modeling/train_baseline.py or modeling/train_transformer.py).
- Ensures tracking options (MLflow/W&B) and experiment names are set.

YAML schema (example):

experiment_name: "norm_ablation_baseline"
tracking: "mlflow"               # none|mlflow|wandb
mlflow_experiment: "khmer_norm_ablation"

base:
  script: "modeling/train_baseline.py"
  input: "annotation/sample_data/final_dataset.csv"
  use_splits: true
  output_root: "runs/norm_ablation_baseline"

  fixed:
    train_ratio: 0.8
    val_ratio: 0.1
    test_ratio: 0.1
    ngram_min: 3
    ngram_max: 5
    min_df: 2
    max_iter: 200
    class_weight: "balanced"
    calibrate: "none"
    resample: "none"
    seed: 123

grid:
  normalize_all: [true]
  norm_khmer_digits: ["keep", "map"]
  norm_khmer_punct: [false, true]
  norm_emoji: ["keep", "map"]

Usage:
  python tools/run_experiment_grid.py --config experiments/normalization_ablation_baseline.yml

This script only orchestrates; individual training scripts remain the source
of truth for behavior and tracking (via modeling/utils/experiment.py).
"""
from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import json
import csv
import math
import hashlib

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover - clear error path
    raise SystemExit("pyyaml is required for run_experiment_grid.py. Install with: pip install pyyaml") from e


@dataclass
class ExperimentConfig:
    experiment_name: str
    tracking: str
    mlflow_experiment: str | None
    base_script: str
    base_input: str
    use_splits: bool
    output_root: str
    fixed: Dict[str, Any]
    grid: Dict[str, List[Any]]


def load_experiment_config(path: Path) -> ExperimentConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    exp_name = str(data.get("experiment_name", path.stem))
    tracking = str(data.get("tracking", "mlflow"))
    mlflow_experiment = data.get("mlflow_experiment")

    base = data.get("base") or {}
    script = base.get("script")
    if not script:
        raise SystemExit("Experiment YAML must define base.script (path to training script)")
    base_input = base.get("input")
    if not base_input:
        raise SystemExit("Experiment YAML must define base.input (dataset path)")
    use_splits = bool(base.get("use_splits", False))
    output_root = base.get("output_root") or f"runs/{exp_name}"

    fixed = base.get("fixed") or {}
    grid = data.get("grid") or {}
    # Normalize grid values to lists
    grid_norm: Dict[str, List[Any]] = {}
    for k, v in grid.items():
        if isinstance(v, list):
            grid_norm[k] = v
        else:
            grid_norm[k] = [v]

    return ExperimentConfig(
        experiment_name=exp_name,
        tracking=tracking,
        mlflow_experiment=str(mlflow_experiment) if mlflow_experiment is not None else None,
        base_script=str(script),
        base_input=str(base_input),
        use_splits=use_splits,
        output_root=str(output_root),
        fixed=fixed,
        grid=grid_norm,
    )


def cartesian_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combos: List[Dict[str, Any]] = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def combo_to_suffix(combo: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in sorted(combo.keys()):
        v = combo[k]
        if isinstance(v, bool):
            vs = "true" if v else "false"
        else:
            vs = str(v).replace("/", "_").replace(" ", "_")
        parts.append(f"{k}_{vs}")
    return "__" + "__".join(parts) if parts else ""


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metrics_json(run_dir: Path) -> Optional[Dict[str, Any]]:
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return None
    try:
        return json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    try:
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur
    except Exception:
        return default


def extract_core_metrics(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "test_f1_macro": None,
        "test_accuracy": None,
        "test_ece": None,
        "test_brier": None,
        "test_throughput_examples_per_sec": None,
    }
    # Try common locations
    test_block = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}
    if isinstance(test_block, dict):
        # HF trainer evaluate usually writes eval_accuracy / eval_f1_macro
        f1m = test_block.get("eval_f1_macro", test_block.get("eval_f1"))
        acc = test_block.get("eval_accuracy", test_block.get("accuracy"))
        out["test_f1_macro"] = float(f1m) if isinstance(f1m, (int, float)) else None
        out["test_accuracy"] = float(acc) if isinstance(acc, (int, float)) else None
    # Calibration block
    calib_block = metrics.get("calibration") if isinstance(metrics.get("calibration"), dict) else {}
    if isinstance(calib_block, dict):
        test_cal = calib_block.get("test") if isinstance(calib_block.get("test"), dict) else {}
        if isinstance(test_cal, dict):
            ece = test_cal.get("ece")
            brier = test_cal.get("brier")
            out["test_ece"] = float(ece) if isinstance(ece, (int, float)) else None
            out["test_brier"] = float(brier) if isinstance(brier, (int, float)) else None
    # Timing/throughput
    thr = _safe_get(metrics, ["timing", "test_throughput_examples_per_sec"], None)
    if isinstance(thr, (int, float)):
        out["test_throughput_examples_per_sec"] = float(thr)
    return out


def mean_std_ci(values: List[float]) -> Dict[str, float]:
    n = len(values)
    mean = sum(values) / n if n else float("nan")
    if n <= 1:
        return {"n": n, "mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    # Normal approx 95% CI
    half = 1.96 * std / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def write_csv_summary(path: Path, rows: List[Dict[str, Any]], field_order: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in field_order})


def build_command(cfg: ExperimentConfig, combo: Dict[str, Any], idx: int, total: int, seed_override: Optional[int] = None) -> Tuple[str, Path]:
    """Construct the command line string and output_dir for one combination.

    The training script is called via `python <script> ...` with:
    - --input, --use_splits as defined in cfg.base
    - --output_dir derived from output_root + combo suffix (+ optional seed suffix)
    - --tracking, --experiment_name, --mlflow_experiment
    - all fixed params as --key value (with optional seed override)
    - all combo params as --key value
    """
    suffix = combo_to_suffix(combo)
    if seed_override is not None:
        suffix = f"{suffix}__seed_{seed_override}"
    output_dir = Path(cfg.output_root) / f"run_{idx:03d}{suffix}"
    # Base command
    cmd: List[str] = [
        "python",
        cfg.base_script,
        "--input",
        cfg.base_input,
        "--output_dir",
        str(output_dir),
        "--tracking",
        cfg.tracking,
        "--experiment_name",
        f"{cfg.experiment_name}_run_{idx:03d}",
    ]
    if cfg.mlflow_experiment:
        cmd.extend(["--mlflow_experiment", cfg.mlflow_experiment])
    if cfg.use_splits:
        cmd.append("--use_splits")

    # Fixed params (respect seed override if provided)
    fixed_local: Dict[str, Any] = dict(cfg.fixed)
    if seed_override is not None:
        fixed_local["seed"] = int(seed_override)
    for k, v in fixed_local.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])

    # Grid combo overrides
    for k, v in combo.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        elif v is None:
            # Skip None: rely on script default
            continue
        else:
            cmd.extend([flag, str(v)])

    return " ".join(shlex.quote(c) for c in cmd), output_dir


def run_command(cmd: str) -> int:
    print(f"[run_experiment_grid] Executing: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[run_experiment_grid] Command failed with code {result.returncode}")
    return result.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a grid of experiments from a YAML config")
    ap.add_argument("--config", required=True, help="Path to experiment YAML config")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    ap.add_argument("--seeds", nargs="+", help="List of seeds to run for each grid combo (e.g., --seeds 123 456 789)")
    ap.add_argument("--aggregate", action="store_true", help="Aggregate metrics across seeds and write summaries under output_root")
    ap.add_argument("--post_ci_pairs", nargs="*", help="Optional pairs A,B of prediction CSVs or run dirs to compute paired bootstrap CIs after runs. Example: --post_ci_pairs runs/a/test_predictions.csv,runs/b/test_predictions.csv ...")
    ap.add_argument("--post_ci_n", type=int, default=10000, help="Number of bootstrap samples for paired CI")
    ap.add_argument("--post_ci_alpha", type=float, default=0.05, help="Alpha for CI (default 0.05 for 95% CI)")
    ap.add_argument("--post_ci_seed", type=int, default=123, help="Seed for bootstrap")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")

    cfg = load_experiment_config(cfg_path)
    combos = cartesian_grid(cfg.grid)
    print(f"Loaded config '{cfg.experiment_name}' with {len(combos)} combinations")

    # Ensure output root exists
    out_root = Path(cfg.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve seeds: CLI overrides YAML fixed seed; default to 42 if none provided
    seeds: List[int]
    if getattr(args, "seeds", None):
        seeds = [int(s) for s in args.seeds]
    else:
        base_seed = int(cfg.fixed.get("seed", 42)) if isinstance(cfg.fixed, dict) else 42
        seeds = [base_seed]

    # Compute input checksum for governance
    try:
        inp_checksum = sha256_of_file(Path(cfg.base_input))
    except Exception:
        inp_checksum = None

    # Track runs for aggregation and checksums
    runs_index: Dict[str, List[Dict[str, Any]]] = {}
    checksums_payload: Dict[str, Any] = {
        "input_path": cfg.base_input,
        "input_sha256": inp_checksum,
        "runs": [],
    }

    for idx, combo in enumerate(combos, start=1):
        base_key = f"run_{idx:03d}{combo_to_suffix(combo)}"
        for sd in seeds:
            cmd, out_dir = build_command(cfg, combo, idx, len(combos), seed_override=sd)
            print(f"\n[run_experiment_grid] ({idx}/{len(combos)}) seed={sd} -> output_dir={out_dir}")
            if args.dry_run:
                print(cmd)
            else:
                rc = run_command(cmd)
                if rc != 0:
                    print(f"[run_experiment_grid] Warning: run {idx} (seed={sd}) exited with code {rc}")
            # Record for later aggregation
            runs_index.setdefault(base_key, []).append({"dir": str(out_dir), "seed": int(sd)})
            checksums_payload["runs"].append({"dir": str(out_dir), "seed": int(sd)})

    # Persist checksums/governance file
    try:
        (out_root / "checksums.json").write_text(json.dumps(checksums_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Aggregation across seeds per combo
    if getattr(args, "aggregate", False):
        print("\n[run_experiment_grid] Aggregating metrics across seeds...")
        aggregate: Dict[str, Any] = {"combos": {}, "seeds_per_combo": {}}
        csv_rows: List[Dict[str, Any]] = []
        for key, runs in runs_index.items():
            # Load metrics for each run
            metrics_list: List[Dict[str, Any]] = []
            for r in runs:
                m = load_metrics_json(Path(r["dir"]))
                if m is not None:
                    metrics_list.append(m)
            # Extract core metrics
            f1s: List[float] = []
            accs: List[float] = []
            eces: List[float] = []
            briers: List[float] = []
            thrus: List[float] = []
            for m in metrics_list:
                core = extract_core_metrics(m)
                if isinstance(core.get("test_f1_macro"), float):
                    f1s.append(float(core["test_f1_macro"]))
                if isinstance(core.get("test_accuracy"), float):
                    accs.append(float(core["test_accuracy"]))
                if isinstance(core.get("test_ece"), float):
                    eces.append(float(core["test_ece"]))
                if isinstance(core.get("test_brier"), float):
                    briers.append(float(core["test_brier"]))
                if isinstance(core.get("test_throughput_examples_per_sec"), float):
                    thrus.append(float(core["test_throughput_examples_per_sec"]))
            agg_entry: Dict[str, Any] = {}
            if f1s:
                agg_entry["test_f1_macro"] = mean_std_ci(f1s)
            if accs:
                agg_entry["test_accuracy"] = mean_std_ci(accs)
            if eces:
                agg_entry["test_ece"] = mean_std_ci(eces)
            if briers:
                agg_entry["test_brier"] = mean_std_ci(briers)
            if thrus:
                agg_entry["test_throughput_examples_per_sec"] = mean_std_ci(thrus)
            aggregate["combos"][key] = agg_entry
            aggregate["seeds_per_combo"][key] = [int(r["seed"]) for r in runs]
            # CSV row (flatten selected fields)
            row: Dict[str, Any] = {"combo_key": key, "n_runs": len(metrics_list)}
            if "test_f1_macro" in agg_entry:
                row.update({
                    "f1_mean": agg_entry["test_f1_macro"]["mean"],
                    "f1_ci_low": agg_entry["test_f1_macro"]["ci_low"],
                    "f1_ci_high": agg_entry["test_f1_macro"]["ci_high"],
                })
            if "test_accuracy" in agg_entry:
                row.update({
                    "acc_mean": agg_entry["test_accuracy"]["mean"],
                })
            if "test_ece" in agg_entry:
                row.update({"ece_mean": agg_entry["test_ece"]["mean"]})
            if "test_brier" in agg_entry:
                row.update({"brier_mean": agg_entry["test_brier"]["mean"]})
            if "test_throughput_examples_per_sec" in agg_entry:
                row.update({"throughput_mean": agg_entry["test_throughput_examples_per_sec"]["mean"]})
            csv_rows.append(row)
        try:
            (out_root / "aggregate_summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
            write_csv_summary(out_root / "aggregate_summary.csv", csv_rows, [
                "combo_key", "n_runs", "f1_mean", "f1_ci_low", "f1_ci_high", "acc_mean", "ece_mean", "brier_mean", "throughput_mean"
            ])
            print(f"[run_experiment_grid] Wrote aggregate summaries under {out_root}")
        except Exception as e:
            print(f"[run_experiment_grid] Failed to write aggregate summaries: {e}")

    # Optional post-run paired bootstrap CI on provided pairs
    if getattr(args, "post_ci_pairs", None):
        print("\n[run_experiment_grid] Running post-run paired bootstrap CI...")
        ci_dir = out_root / "ci"
        ci_dir.mkdir(parents=True, exist_ok=True)
        for i, pair in enumerate(args.post_ci_pairs, start=1):
            try:
                a, b = pair.split(",", 1)
            except ValueError:
                print(f"[run_experiment_grid] Skipping malformed pair: {pair}")
                continue
            # Resolve prediction CSVs
            def resolve_pred(p: str) -> Path:
                pp = Path(p)
                if pp.is_dir():
                    cand = pp / "test_predictions.csv"
                    if cand.exists():
                        return cand
                return pp
            pa = resolve_pred(a)
            pb = resolve_pred(b)
            if not pa.exists() or not pb.exists():
                print(f"[run_experiment_grid] Skipping pair (missing files): {pa}, {pb}")
                continue
            outp = ci_dir / f"pair_{i:02d}"
            # Call paired_bootstrap_ci.py via subprocess
            cmd = f"python tools/paired_bootstrap_ci.py --predictions_a {pa} --predictions_b {pb} --output {outp} --n_bootstrap {int(args.post_ci_n)} --alpha {float(args.post_ci_alpha)} --seed {int(args.post_ci_seed)}"
            rc = run_command(cmd)
            if rc != 0:
                print(f"[run_experiment_grid] Warning: CI computation failed for pair {i}")
        print(f"[run_experiment_grid] CI outputs under {ci_dir}")


if __name__ == "__main__":
    main()
