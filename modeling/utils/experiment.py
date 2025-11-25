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

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import os
import platform

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

# Optional system info
try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def env_metadata() -> Dict[str, Optional[str]]:
    """Return lightweight environment metadata for experiment logging.

    Only a small set of common ML libraries is inspected via ``__import__`` to
    avoid importing heavy dependencies unnecessarily. Missing libraries are
    recorded as ``None`` rather than raising.
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

    Returns a dict with CPU, RAM, GPU (if torch+CUDA available), and OS info.
    All fields are optional and failures are swallowed to keep this non-fatal.
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
                # Query first device for summary
                try:
                    name0 = torch.cuda.get_device_name(0)
                    cap = torch.cuda.get_device_capability(0)
                    info["gpu0"] = {
                        "name": name0,
                        "capability": f"{cap[0]}.{cap[1]}",
                    }
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


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Best-effort global seeding across common libraries.

    The function never raises: if a backend is missing or misconfigured the
    corresponding block is silently skipped. This keeps training scripts
    robust while still providing determinism where possible.
    """
    random.seed(seed)

    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            # Numpy not available / misconfigured – ignore
            pass

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():  # type: ignore[union-attr]
                torch.cuda.manual_seed_all(seed)
            if deterministic:
                # cuDNN flags may not exist on some builds
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            # CUDA or backend may not be available; keep going.
            pass


@dataclass
class Config:
    """Configuration container for experiments.

    Dataclass fields are the core, well-known options. Arbitrary additional
    hyperparameters can be stored in :attr:`params` (typically coming from
    YAML/CLI) and are forwarded to trackers and snapshots.
    """

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
        """Load configuration from a YAML file into a :class:`Config`.

        Keys that correspond to dataclass fields are mapped directly, while any
        additional keys are stored in :attr:`params` so they can still be
        tracked and used by training scripts.
        """
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
        """Merge a flat mapping of overrides into this config.

        Keys matching dataclass fields update those fields; all remaining keys
        are stored in :attr:`params` for logging. This allows CLI/YAML to
        contain arbitrary hyperparameters without having to extend the
        dataclass definition.
        """
        if not overrides:
            return
        for k, v in overrides.items():
            if k in self.__dataclass_fields__:
                setattr(self, k, v)
            else:
                self.params[k] = v


class Tracker:
    """Unified interface over optional tracking backends (MLflow / W&B).

    The tracker is intentionally defensive: backend failures are swallowed so
    that experiments can proceed without tracking rather than crashing.
    """

    def __init__(self, cfg: Config, run_name: Optional[str] = None) -> None:
        self.cfg = cfg
        self.run_name = run_name or cfg.experiment_name
        self._mlflow_active: bool = False
        self._wandb_active: bool = False

    def __enter__(self) -> "Tracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.finish()

    def start(self) -> None:
        """Initialize the configured tracking backend, if any.

        All backend-specific failures are swallowed so that experiments can
        still run if MLflow/W&B are misconfigured or unavailable.
        """
        tracking = (self.cfg.tracking or "none").lower()

        if tracking == "mlflow" and mlflow is not None:
            try:
                if self.cfg.mlflow_tracking_uri:
                    mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
                if self.cfg.mlflow_experiment:
                    mlflow.set_experiment(self.cfg.mlflow_experiment)
                mlflow.start_run(run_name=self.run_name)
                self._mlflow_active = True
                # log base info
                mlflow.log_params(
                    {
                        "seed": self.cfg.seed,
                        "experiment_name": self.cfg.experiment_name,
                        "output_dir": self.cfg.output_dir,
                    }
                )
                mlflow.log_dict(env_metadata(), "env.json")
                if self.cfg.params:
                    mlflow.log_dict(self.cfg.params, "params.json")
            except Exception:
                # Disable MLflow on failure and continue without tracking.
                self._mlflow_active = False

        elif tracking == "wandb" and wandb is not None:
            try:
                wandb.init(
                    project=self.cfg.wandb_project or "khmer-sentiment",
                    entity=self.cfg.wandb_entity,
                    name=self.run_name,
                    mode=self.cfg.wandb_mode,
                    config={"seed": self.cfg.seed, **self.cfg.params},
                )
                self._wandb_active = True
            except Exception:
                self._wandb_active = False

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log hyperparameters to the active tracking backend(s).

        Non-serializable objects are ignored for MLflow's flat parameter
        logging but preserved in the full JSON artifact.
        """
        if self._mlflow_active:
            try:
                mlflow.log_params(
                    {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}
                )
                # dump full args as artifact
                mlflow.log_dict(dict(params), "all_args.json")
            except Exception:
                pass

        if self._wandb_active:
            try:
                wandb.config.update(dict(params), allow_val_change=True)
            except Exception:
                pass

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to the active backends.

        For MLflow, only numeric metrics are logged as scalar metrics; the full
        metrics mapping is stored as a JSON artifact for later inspection.
        """
        if self._mlflow_active:
            flat: Dict[str, float] = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    flat[k] = float(v)
            try:
                mlflow.log_metrics(flat, step=step)
                # save full metrics as artifact as well
                mlflow.log_dict(dict(metrics), f"metrics_{int(time.time())}.json")
            except Exception:
                pass

        if self._wandb_active:
            try:
                wandb.log(dict(metrics), step=step)
            except Exception:
                pass

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None:
        """Log a file as an artifact to the active backends, if any."""
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

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set experiment tags on the active backends."""
        if self._mlflow_active:
            try:
                mlflow.set_tags(tags)
            except Exception:
                pass
        if self._wandb_active:
            try:
                wandb.run.tags = list(
                    set(list(getattr(wandb.run, "tags", [])) + list(tags.values()))  # type: ignore[attr-defined]
                )
            except Exception:
                pass

    def finish(self) -> None:
        """Finalize the active tracking runs, if any."""
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
    """Create directory *p* (and parents) if it does not already exist."""
    p.mkdir(parents=True, exist_ok=True)


def prepare_experiment(cfg: Config) -> Tracker:
    """Prepare experiment output directory, seeding, and tracking.

    This is the main entry point used by training scripts. It ensures the
    output directory exists, applies global seeding, initializes the tracker,
    and writes a self-contained configuration snapshot under
    ``<output_dir>/experiment_config.json``.
    """
    # ensure output dir
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    # set seed
    set_global_seed(cfg.seed, deterministic=True)

    # start tracker
    tracker = Tracker(cfg)
    tracker.start()
    tracker.set_tags({"exp_name": cfg.experiment_name})

    # persist config snapshot to output_dir
    env = env_metadata()
    snap = {
        "experiment_name": cfg.experiment_name,
        "seed": cfg.seed,
        "tracking": cfg.tracking,
        "output_dir": cfg.output_dir,
        "params": cfg.params,
        "env": env,
    }
    out_dir.joinpath("experiment_config.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # collect hardware metadata and store/log
    hw = hardware_metadata()
    try:
        out_dir.joinpath("hardware.json").write_text(
            json.dumps(hw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tracker.log_artifact(out_dir / "hardware.json")
    except Exception:
        pass

    # set tags with a few useful hints for experiment browsing
    tags: Dict[str, str] = {}
    try:
        if hw.get("cuda_available"):
            tags["device"] = "cuda"
        else:
            tags["device"] = "cpu"
        if "gpu0" in hw and isinstance(hw["gpu0"], dict):
            tags["gpu_name"] = str(hw["gpu0"].get("name", "unknown"))
        if hw.get("cuda_version"):
            tags["cuda_version"] = str(hw["cuda_version"])  # type: ignore[arg-type]
        if env.get("torch"):
            tags["torch_version"] = str(env.get("torch"))
    except Exception:
        pass
    if tags:
        tracker.set_tags(tags)

    return tracker