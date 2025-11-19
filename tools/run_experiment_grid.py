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
from typing import Any, Dict, List, Tuple

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


def build_command(cfg: ExperimentConfig, combo: Dict[str, Any], idx: int, total: int) -> Tuple[str, Path]:
    """Construct the command line string and output_dir for one combination.

    The training script is called via `python <script> ...` with:
    - --input, --use_splits as defined in cfg.base
    - --output_dir derived from output_root + combo suffix
    - --tracking, --experiment_name, --mlflow_experiment
    - all fixed params as --key value
    - all combo params as --key value
    """
    suffix = combo_to_suffix(combo)
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

    # Fixed params
    for k, v in cfg.fixed.items():
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
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")

    cfg = load_experiment_config(cfg_path)
    combos = cartesian_grid(cfg.grid)
    print(f"Loaded config '{cfg.experiment_name}' with {len(combos)} combinations")

    # Ensure output root exists
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)

    for idx, combo in enumerate(combos, start=1):
        cmd, out_dir = build_command(cfg, combo, idx, len(combos))
        print(f"\n[run_experiment_grid] ({idx}/{len(combos)}) -> output_dir={out_dir}")
        if args.dry_run:
            print(cmd)
        else:
            rc = run_command(cmd)
            if rc != 0:
                # Decide whether to stop on error; for now, continue but report
                print(f"[run_experiment_grid] Warning: run {idx} exited with code {rc}")


if __name__ == "__main__":
    main()
