#!/usr/bin/env python3
"""
Experiment utilities: configuration management, global seeding, and experiment tracking.

Architectural improvements:
- Pluggable backend architecture via backend classes (MLflow, W&B, Null)
- Backward-compatible Tracker facade with the same public API
- Safer, defensive logging with per-backend isolation
- Enriched environment capture (env, hardware, optional git info)
- Deterministic seeding across common libs
- Config snapshotting into output dir

Optional dependencies (import-lazy):
  pyyaml, mlflow, wandb, torch, psutil
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, List, Union
import os
import platform

# ---------------------
# Optional dependencies
# ---------------------
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # optional

try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# ---------------
# Env/gather utils
# ---------------

def env_metadata() -> Dict[str, Optional[str]]:
    """Return lightweight environment metadata for experiment logging.

    Only minimal metadata is collected to avoid heavy imports.
    """
    meta: Dict[str, Optional[str]] = {"python": sys.version.replace("\n", " ")}
    for name in ("numpy", "pandas", "sklearn", "torch", "transformers"):
        try:
            m = __import__(name)
            meta[name] = getattr(m, "__version__", None)
        except Exception:
            meta[name] = None
    return meta


def hardware_metadata() -> Dict[str, Any]:
    """Collect best-effort hardware/system information.

    Fails softly on all lookups.
    """
    info: Dict[str, Any] = {}
    try:
        info["os"] = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "python_implementation": platform.python_implementation(),
        }
    except Exception:
        pass

    try:
        info["cpu"] = {
            "count_logical": os.cpu_count(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        }
    except Exception:
        pass

    try:
        if psutil is not None:
            vm = psutil.virtual_memory()
            info.setdefault("ram", {})["total_gb"] = round(float(vm.total) / (1024 ** 3), 2)
    except Exception:
        pass

    try:
        if torch is not None:
            cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())  # type: ignore[attr-defined]
            info["cuda_available"] = cuda_ok
            if cuda_ok:
                dev_count = torch.cuda.device_count()
                info["cuda_device_count"] = int(dev_count)
                try:
                    name0 = torch.cuda.get_device_name(0)
                    cap = torch.cuda.get_device_capability(0)
                    info["gpu0"] = {"name": name0, "capability": f"{cap[0]}.{cap[1]}"}
                except Exception:
                    pass
                try:
                    drv = torch.version.cuda  # type: ignore[attr-defined]
                    info["cuda_version"] = drv
                except Exception:
                    pass
    except Exception:
        pass

    return info


def get_git_info(root: Optional[Path] = None) -> Dict[str, Any]:
    """Attempt to collect git metadata (commit, branch, remote) without requiring gitpython.

    Returns empty dict on any failure.
    """
    root = root or Path.cwd()
    info: Dict[str, Any] = {}
    try:
        import subprocess
        def _run(args: List[str]) -> Optional[str]:
            try:
                out = subprocess.check_output(args, cwd=str(root), stderr=subprocess.DEVNULL)
                return out.decode("utf-8", errors="ignore").strip()
            except Exception:
                return None
        # Only proceed if inside a repo
        status = _run(["git", "rev-parse", "--is-inside-work-tree"]) or ""
        if status.lower() != "true":
            return info
        info["commit"] = _run(["git", "rev-parse", "HEAD"]) or None
        info["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or None
        info["remote_origin_url"] = _run(["git", "config", "--get", "remote.origin.url"]) or None
        info["dirty"] = bool((_run(["git", "status", "--porcelain"]) or ""))
    except Exception:
        pass
    return info

# -----------------
# Global seed helper
# -----------------

def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Best-effort global seeding across common libraries.

    Never raises; skips missing backends.
    """
    random.seed(seed)

    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():  # type: ignore[union-attr]
                torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass

# ----------------------
# Config and Tracker API
# ----------------------

@dataclass
class Config:
    """Configuration container for experiments.

    Public fields mirror stable keys used across scripts. Extra keys go into
    :attr:`params` and are also snapshot in experiment_config.json.
    """

    # Generic experiment fields
    experiment_name: str = "default"
    seed: int = 42
    tracking: str = "none"  # one of: none, mlflow, wandb or comma-separated (e.g. "mlflow,wandb")
    output_dir: str = "runs/default"

    # MLflow
    mlflow_tracking_uri: Optional[str] = None  # if None, uses local ./mlruns
    mlflow_experiment: Optional[str] = None

    # W&B
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: Optional[str] = None  # e.g., "offline"

    # Optional tags to apply
    tags: Dict[str, str] = field(default_factory=dict)

    # Arbitrary extra params bag
    params: Dict[str, Any] = field(default_factory=dict)

    # Behavior flags
    autosave_config: bool = True

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

    def merge_overrides(self, overrides: Optional[Mapping[str, Any]]) -> None:
        if not overrides:
            return
        for k, v in overrides.items():
            if k in self.__dataclass_fields__:
                setattr(self, k, v)
            else:
                self.params[k] = v

# ---------------------
# Backend abstractions
# ---------------------

class TrackerBackend:
    """Abstract tracker backend interface."""

    def start(self) -> None: ...
    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None: ...
    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None: ...
    def set_tags(self, tags: Mapping[str, str]) -> None: ...
    def finish(self) -> None: ...


class NullBackend(TrackerBackend):
    def start(self) -> None: pass
    def log_params(self, params: Mapping[str, Any]) -> None: pass
    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None: pass
    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None: pass
    def set_tags(self, tags: Mapping[str, str]) -> None: pass
    def finish(self) -> None: pass


class MlflowBackend(TrackerBackend):
    def __init__(self, cfg: Config, run_name: Optional[str]) -> None:
        self.cfg = cfg
        self.run_name = run_name or cfg.experiment_name
        self.active = False

    def start(self) -> None:
        if mlflow is None:
            return
        try:
            if self.cfg.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
            if self.cfg.mlflow_experiment:
                mlflow.set_experiment(self.cfg.mlflow_experiment)
            mlflow.start_run(run_name=self.run_name)
            self.active = True
            mlflow.log_params({
                "seed": self.cfg.seed,
                "experiment_name": self.cfg.experiment_name,
                "output_dir": self.cfg.output_dir,
            })
            mlflow.log_dict(env_metadata(), "env.json")
            if self.cfg.params:
                mlflow.log_dict(self.cfg.params, "params.json")
        except Exception:
            self.active = False

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.active:
            return
        try:
            mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))})
            mlflow.log_dict(dict(params), "all_args.json")
        except Exception:
            pass

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        if not self.active:
            return
        try:
            flat = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            if flat:
                mlflow.log_metrics(flat, step=step)
            # Save full metrics snapshot as an artifact for later inspection
            mlflow.log_dict(dict(metrics), f"metrics_{int(time.time())}.json")
        except Exception:
            pass

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None:
        if not self.active or not path.exists():
            return
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception:
            pass

    def set_tags(self, tags: Mapping[str, str]) -> None:
        if not self.active:
            return
        try:
            mlflow.set_tags(tags)
        except Exception:
            pass

    def finish(self) -> None:
        if not self.active:
            return
        try:
            mlflow.end_run()
        except Exception:
            pass


class WandbBackend(TrackerBackend):
    def __init__(self, cfg: Config, run_name: Optional[str]) -> None:
        self.cfg = cfg
        self.run_name = run_name or cfg.experiment_name
        self.active = False

    def start(self) -> None:
        if wandb is None:
            return
        try:
            wandb.init(
                project=self.cfg.wandb_project or "khmer-sentiment",
                entity=self.cfg.wandb_entity,
                name=self.run_name,
                mode=self.cfg.wandb_mode,
                config={"seed": self.cfg.seed, **self.cfg.params},
            )
            self.active = True
        except Exception:
            self.active = False

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.active:
            return
        try:
            wandb.config.update(dict(params), allow_val_change=True)
        except Exception:
            pass

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        if not self.active:
            return
        try:
            wandb.log(dict(metrics), step=step)
        except Exception:
            pass

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None:
        if not self.active or not path.exists():
            return
        try:
            # W&B doesn't have nested artifact_path via this API; save file to run
            wandb.save(str(path))
        except Exception:
            pass

    def set_tags(self, tags: Mapping[str, str]) -> None:
        if not self.active:
            return
        try:
            run = getattr(wandb, "run", None)
            if run is not None:
                cur_tags = set(list(getattr(run, "tags", [])))
                cur_tags |= set(tags.values())
                run.tags = list(cur_tags)
        except Exception:
            pass

    def finish(self) -> None:
        if not self.active:
            return
        try:
            wandb.finish()
        except Exception:
            pass

# --------------
# Tracker facade
# --------------

class Tracker:
    """Facade over one or more backends with a stable, simple API.

    Backends are isolated so one failing backend doesn't break others.
    """

    def __init__(self, cfg: Config, run_name: Optional[str] = None) -> None:
        self.cfg = cfg
        self.run_name = run_name or cfg.experiment_name
        self.backends: List[TrackerBackend] = []
        self._started = False

    def __enter__(self) -> "Tracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.finish()

    def _init_backends(self) -> None:
        backends: List[TrackerBackend] = []
        # Parse tracking string; allow comma-separated list
        desired: List[str] = []
        try:
            tval = (self.cfg.tracking or "none").lower()
            desired = [x.strip() for x in tval.split(",") if x.strip()]
        except Exception:
            desired = ["none"]
        if not desired:
            desired = ["none"]

        if "mlflow" in desired:
            backends.append(MlflowBackend(self.cfg, self.run_name))
        if "wandb" in desired:
            backends.append(WandbBackend(self.cfg, self.run_name))
        if not backends:
            backends.append(NullBackend())
        self.backends = backends

    def start(self) -> None:
        if self._started:
            return
        self._init_backends()
        for b in self.backends:
            try:
                b.start()
            except Exception:
                # isolate failing backend
                pass
        self._started = True

    def log_params(self, params: Mapping[str, Any]) -> None:
        for b in self.backends:
            try:
                b.log_params(params)
            except Exception:
                pass

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        for b in self.backends:
            try:
                b.log_metrics(metrics, step=step)
            except Exception:
                pass

    def log_artifact(self, path: Path | str, artifact_path: Optional[str] = None) -> None:
        p = Path(path)
        for b in self.backends:
            try:
                b.log_artifact(p, artifact_path=artifact_path)
            except Exception:
                pass

    def set_tags(self, tags: Mapping[str, str]) -> None:
        for b in self.backends:
            try:
                b.set_tags(tags)
            except Exception:
                pass

    def finish(self) -> None:
        for b in self.backends:
            try:
                b.finish()
            except Exception:
                pass

# ----------------------
# Orchestration helpers
# ----------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def prepare_experiment(cfg: Config) -> Tracker:
    """Prepare experiment output dir, seeding, tracking, and snapshots.

    - Ensures output directory exists
    - Applies global seeding
    - Initializes tracking backends
    - Writes experiment_config.json and hardware.json to the output directory
    """
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    set_global_seed(cfg.seed, deterministic=True)

    tracker = Tracker(cfg)
    tracker.start()

    # Standard tags derived from environment and config
    tags: Dict[str, str] = {"exp_name": cfg.experiment_name}
    env = env_metadata()
    hw = hardware_metadata()
    try:
        if hw.get("cuda_available"):
            tags["device"] = "cuda"
        else:
            tags["device"] = "cpu"
        if isinstance(hw.get("gpu0"), dict) and hw["gpu0"].get("name"):
            tags["gpu_name"] = str(hw["gpu0"]["name"])  # type: ignore[index]
        if hw.get("cuda_version"):
            tags["cuda_version"] = str(hw.get("cuda_version"))
        if env.get("torch"):
            tags["torch_version"] = str(env.get("torch"))
    except Exception:
        pass

    # Merge user-provided tags
    try:
        tags.update({k: str(v) for k, v in (cfg.tags or {}).items()})
    except Exception:
        pass

    if tags:
        tracker.set_tags(tags)

    # Persist snapshots
    snap = {
        "experiment_name": cfg.experiment_name,
        "seed": cfg.seed,
        "tracking": cfg.tracking,
        "output_dir": cfg.output_dir,
        "params": cfg.params,
        "env": env,
        "git": get_git_info(Path(cfg.output_dir)).copy(),
    }
    try:
        out_dir.joinpath("experiment_config.json").write_text(
            json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

    try:
        out_dir.joinpath("hardware.json").write_text(
            json.dumps(hw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tracker.log_artifact(out_dir / "hardware.json")
    except Exception:
        pass

    # Also upload experiment_config.json if possible
    try:
        tracker.log_artifact(out_dir / "experiment_config.json")
    except Exception:
        pass

    return tracker
