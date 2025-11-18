#!/usr/bin/env python3
"""
Experiment utilities: configuration management, global seeding, and experiment tracking.

- Config loading/merging from YAML and CLI overrides
- Global seeding for random, numpy, torch/transformers (if available) and sklearn
- Lightweight tracking backend abstraction with MLflow (default) and optional Weights & Biases
- Deterministic helpers and environment metadata capture

Dependencies (optional):
  pyyaml, mlflow, wandb, torch
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # optional

# Optional backends
try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

# Optional libs for seeding
try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    import torch  # type: ignore
except Exception:
    torch = None


def env_metadata() -> Dict[str, Optional[str]]:
    meta: Dict[str, Optional[str]] = {"python": sys.version.replace("\n", " ")}
    for name in ("numpy", "pandas", "sklearn", "torch", "transformers"):
        try:
            m = __import__(name)
            meta[name] = getattr(m, "__version__", None)
        except Exception:
            meta[name] = None
    return meta


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True  # type: ignore
                torch.backends.cudnn.benchmark = False  # type: ignore
        except Exception:
            pass


@dataclass
class Config:
    # Generic experiment fields
    experiment_name: str = "default"
    seed: int = 42
    tracking: str = "mlflow"  # one of: none, mlflow, wandb
    output_dir: str = "runs/default"
    # MLflow
    mlflow_tracking_uri: Optional[str] = None  # if None, uses local ./mlruns
    mlflow_experiment: Optional[str] = None
    # W&B
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: Optional[str] = None  # e.g., "offline"
    # Arbitrary extra params bag
    params: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: Optional[str]) -> "Config":
        if path is None:
            return Config()
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Config YAML not found: {p}")
        if yaml is None:
            raise SystemExit("pyyaml not installed. Install with: pip install pyyaml")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cfg = Config(**{k: v for k, v in data.items() if k in Config.__dataclass_fields__})
        # store remaining keys into params
        for k, v in data.items():
            if k not in Config.__dataclass_fields__:
                cfg.params[k] = v
        return cfg

    def merge_overrides(self, overrides: Dict[str, Any]) -> None:
        for k, v in (overrides or {}).items():
            if k in Config.__dataclass_fields__:
                setattr(self, k, v)
            else:
                self.params[k] = v


class Tracker:
    def __init__(self, cfg: Config, run_name: Optional[str] = None):
        self.cfg = cfg
        self.run_name = run_name or cfg.experiment_name
        self._mlflow_active = False
        self._wandb_active = False

    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()

    def start(self) -> None:
        if self.cfg.tracking.lower() == "mlflow" and mlflow is not None:
            if self.cfg.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
            if self.cfg.mlflow_experiment:
                mlflow.set_experiment(self.cfg.mlflow_experiment)
            mlflow.start_run(run_name=self.run_name)
            self._mlflow_active = True
            # log base info
            mlflow.log_params({
                "seed": self.cfg.seed,
                "experiment_name": self.cfg.experiment_name,
                "output_dir": self.cfg.output_dir,
            })
            mlflow.log_dict(env_metadata(), "env.json")
            if self.cfg.params:
                mlflow.log_dict(self.cfg.params, "params.json")
        elif self.cfg.tracking.lower() == "wandb" and wandb is not None:
            wandb.init(project=self.cfg.wandb_project or "khmer-sentiment",
                       entity=self.cfg.wandb_entity,
                       name=self.run_name,
                       mode=self.cfg.wandb_mode,
                       config={"seed": self.cfg.seed, **self.cfg.params})
            self._wandb_active = True
        else:
            # no tracking
            pass

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._mlflow_active:
            try:
                mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))})
                # dump full args as artifact
                mlflow.log_dict(params, "all_args.json")
            except Exception:
                pass
        if self._wandb_active:
            try:
                wandb.config.update(params, allow_val_change=True)
            except Exception:
                pass

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._mlflow_active:
            flat = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    flat[k] = v
            try:
                mlflow.log_metrics(flat, step=step)
                # save full metrics as artifact as well
                mlflow.log_dict(metrics, f"metrics_{int(time.time())}.json")
            except Exception:
                pass
        if self._wandb_active:
            try:
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None:
        if not path.exists():
            return
        if self._mlflow_active:
            try:
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
            except Exception:
                pass
        if self._wandb_active:
            try:
                wandb.save(str(path))
            except Exception:
                pass

    def set_tags(self, tags: Dict[str, str]) -> None:
        if self._mlflow_active:
            try:
                mlflow.set_tags(tags)
            except Exception:
                pass
        if self._wandb_active:
            try:
                wandb.run.tags = list(set(list(getattr(wandb.run, 'tags', [])) + list(tags.values())))  # type: ignore
            except Exception:
                pass

    def finish(self) -> None:
        if self._mlflow_active:
            try:
                mlflow.end_run()
            except Exception:
                pass
        if self._wandb_active:
            try:
                wandb.finish()
            except Exception:
                pass


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def prepare_experiment(cfg: Config) -> Tracker:
    # ensure output dir
    ensure_dir(Path(cfg.output_dir))
    # set seed
    set_global_seed(cfg.seed, deterministic=True)
    # start tracker
    tracker = Tracker(cfg)
    tracker.start()
    tracker.set_tags({"exp_name": cfg.experiment_name})
    # persist config snapshot to output_dir
    snap = {
        "experiment_name": cfg.experiment_name,
        "seed": cfg.seed,
        "tracking": cfg.tracking,
        "output_dir": cfg.output_dir,
        "params": cfg.params,
        "env": env_metadata(),
    }
    (Path(cfg.output_dir) / "experiment_config.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    return tracker
