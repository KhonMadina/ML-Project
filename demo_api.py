#!/usr/bin/env python3
"""Interactive demo API for Khmer sentiment with conformal prediction.

This module exposes a small but well-structured FastAPI application that can be
used both as an internal demo backend (served together with demo_ui.html) and
as a lightweight external service.

Key capabilities
----------------
- Single and batch prediction with conformal prediction sets.
- Optional transformer model (if available on disk).
- Simple uncertainty metrics (max_proba, entropy, prediction set size).
- Lightweight feature importance for baseline (LR + TF–IDF).
- Machine-readable metadata endpoint for external clients.
- Model-comparison endpoint for programmatic baseline vs transformer evaluation.
- Optional API key authentication and basic input limiting for safe external use.

Existing public API (backwards compatible)
------------------------------------------
- GET /health
- GET /meta
- POST /predict
- POST /predict/batch
- POST /predict/compare
- GET /  (serves demo_ui.html for interactive exploration)

Usage
-----
  # 1) Ensure you have a trained baseline model directory, e.g. by running run_demo.bat.
  #    By default, we expect `models/baseline_chargram` to exist.

  # 2) Install extra deps (only once):
  #    pip install fastapi uvicorn[standard] scikit-learn joblib numpy

  # 3) Start the API (local demo usage):
  #    python demo_api.py --model_dir models --tau 0.5

  #    For external usage, set DEMO_API_KEY and configure a reverse proxy with TLS.

  # 4) External usage examples (curl):
  #    curl -X POST "http://127.0.0.1:8000/predict" \
  #         -H "Content-Type: application/json" \
  #         -H "X-API-Key: $DEMO_API_KEY" \
  #         -d '{"text": "អរគុណច្រើន សេវាកម្មល្អបំផុត 🙄"}'

  #    curl -X POST "http://127.0.0.1:8000/predict/compare" \
  #         -H "Content-Type: application/json" \
  #         -H "X-API-Key: $DEMO_API_KEY" \
  #         -d '{"text": "...", "tau": 0.2, "normalize": true}'

The application code is intentionally explicit and avoids heavy dependencies
beyond FastAPI and the modeling utilities in this repository.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import sys

try:
    import numpy as np  # type: ignore
    import joblib
except Exception as e:  # pragma: no cover - import failure is fatal at runtime, not under tests
    raise SystemExit(
        "Missing dependency for baseline model. Install with: pip install scikit-learn joblib numpy\n"
        f"Underlying import error: {e}"
    )

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover - import failure is fatal at runtime, not under tests
    raise SystemExit(
        "Missing dependency for API. Install with: pip install fastapi uvicorn[standard]\n"
        f"Underlying import error: {e}"
    )

from modeling.text_normalization import load_norm_config, normalize_text
from modeling import predict_transformer

# ---------------------------------------------------------------------------
# Logging configuration (basic)
# ---------------------------------------------------------------------------

logger = logging.getLogger("demo_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s in %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Constants and simple configuration
# ---------------------------------------------------------------------------

LABELS = ["POS", "NEG", "NEU"]
DEFAULT_MODEL_TYPE = "baseline"  # "baseline" or "transformer"

# Hard-coded demo mapping: baseline vs transformer model directories (under a base models/ dir)
BASELINE_MODEL_NAME = "baseline_chargram"
TRANSFORMER_MODEL_NAME = "transformer_khmer_demo"

# Input limits for external safety
MAX_TEXT_LEN = int(os.getenv("DEMO_MAX_TEXT_LEN", "4000"))
MAX_BATCH_SIZE = int(os.getenv("DEMO_MAX_BATCH_SIZE", "256"))

# Simple API key-based auth (optional). If DEMO_API_KEY is unset/empty, auth is disabled.
API_KEY_ENV = "DEMO_API_KEY"
API_KEY = os.getenv(API_KEY_ENV, "").strip()


# ---------------------------------------------------------------------------
# Pydantic models (public API schema)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for /predict and /predict/batch.

    This is intentionally minimal for easy external integration.
    """

    text: str
    tau: float | None = None
    model_type: str | None = None  # "baseline" or "transformer"
    normalize: bool | None = None


class PredictResponse(BaseModel):
    """Prediction output for a single input.

    This schema is reusable across single, batch, and comparison endpoints.
    """

    text: str
    normalized_text: str
    pred_label: str
    probabilities: Dict[str, float]
    conformal_set: List[str]
    tau: float
    model_type: str
    max_proba: float
    entropy: float
    pred_set_size: int
    importance: Optional[List[Dict[str, Union[str, float]]]] = None


class CompareResponse(BaseModel):
    """Response for /predict/compare.

    Returns a list of per-model predictions for the same input.
    The UI uses a similar structure client-side; this makes it explicit for
    external consumers.
    """

    baseline: Optional[PredictResponse] = None
    transformer: Optional[PredictResponse] = None


class MetaMetrics(BaseModel):
    """Optional summary metrics for each model.

    These values are meant as lightweight, human-readable metadata for clients
    and documentation. They can be populated from experiments or reports.
    """

    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None


class MetaResponse(BaseModel):
    """Metadata about the running service and models.

    This is a machine-readable entry point for external clients that want to
    discover available models, labels, and default configuration.
    """

    model_dir: str
    labels: List[str]
    tau: float
    supported_model_types: List[str]
    default_model_type: str
    default_normalize: bool
    has_transformer: bool
    metrics: Dict[str, MetaMetrics] | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def load_baseline_model(model_dir: Path):
    """Load baseline vectorizer + model from a directory.

    Expected files in model_dir:
    - vectorizer.pkl
    - model.pkl

    The directory may also contain a normalization config consumed by
    :func:`load_norm_config`.
    """

    vec_p = model_dir / "vectorizer.pkl"
    mdl_p = model_dir / "model.pkl"
    if not vec_p.exists() or not mdl_p.exists():
        raise SystemExit(f"vectorizer.pkl or model.pkl not found in {model_dir}. Run run_demo.bat first.")
    vectorizer = joblib.load(vec_p)
    model = joblib.load(mdl_p)
    if not hasattr(model, "predict_proba"):
        raise SystemExit("Model does not expose predict_proba; required for probabilities.")
    norm_cfg = load_norm_config(model_dir)
    return vectorizer, model, norm_cfg


def _compute_uncertainty(proba: np.ndarray) -> Dict[str, float]:
    """Compute simple uncertainty metrics from a probability vector."""

    if proba.size == 0:
        return {"max_proba": 0.0, "entropy": 0.0}
    max_proba = float(proba.max())
    eps = 1e-12
    entropy = float(-np.sum(proba * np.log(proba + eps)))
    return {"max_proba": max_proba, "entropy": entropy}


def _explain_baseline_importance(
    vectorizer,
    model,
    norm_text: str,
    top_k: int = 5,
) -> List[Dict[str, Union[str, float]]]:
    """Return top contributing features for the predicted class (baseline LR only)."""

    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except Exception:
        return []

    if not isinstance(model, LogisticRegression):
        return []

    X = vectorizer.transform([norm_text])
    if not hasattr(model, "coef_"):
        return []

    vocab = getattr(vectorizer, "vocabulary_", None)
    if vocab is None:
        return []
    idx2term = {idx: term for term, idx in vocab.items()}

    proba = model.predict_proba(X)
    proba = np.asarray(proba, dtype=float)[0]
    pred_idx = int(proba.argmax())

    coefs = np.asarray(model.coef_, dtype=float)
    if coefs.ndim != 2 or pred_idx >= coefs.shape[0]:
        return []
    weights = coefs[pred_idx]

    row = X.getrow(0)
    inds = row.indices
    vals = row.data
    contributions: List[Dict[str, Union[str, float]]] = []
    for i, v in zip(inds, vals):
        term = idx2term.get(int(i), f"f{i}")
        contrib = float(v * weights[int(i)])
        contributions.append({"feature": term, "weight": contrib})

    contributions.sort(key=lambda d: abs(float(d["weight"])), reverse=True)
    return contributions[:top_k]


def _predict_baseline(vectorizer, model, norm_cfg, text: str, tau: float, normalize_flag: bool) -> Dict:
    """Run a baseline prediction and return a plain dict matching PredictResponse."""

    # Normalize
    norm_text = normalize_text(text, norm_cfg) if (norm_cfg and normalize_flag) else text

    # Vectorize + probabilities
    X = vectorizer.transform([norm_text])
    proba = model.predict_proba(X)
    proba = np.asarray(proba, dtype=float)[0]

    # Argmax label
    pred_idx = int(proba.argmax())
    pred_label = LABELS[pred_idx] if pred_idx < len(LABELS) else str(pred_idx)

    # Conformal-style prediction set: include labels where (1 - p) <= tau
    pred_set = [LABELS[i] for i, p in enumerate(proba) if (1.0 - float(p)) <= tau]

    probabilities = {f"proba_{lbl}": float(proba[i]) for i, lbl in enumerate(LABELS)}
    unc = _compute_uncertainty(proba)
    importance = _explain_baseline_importance(vectorizer, model, norm_text)

    return {
        "text": text,
        "normalized_text": norm_text,
        "pred_label": pred_label,
        "probabilities": probabilities,
        "conformal_set": pred_set,
        "tau": float(tau),
        "model_type": "baseline",
        "max_proba": unc["max_proba"],
        "entropy": unc["entropy"],
        "pred_set_size": len(pred_set),
        "importance": importance,
    }


def _predict_transformer(tokenizer, model, norm_cfg, labels: List[str], device, text: str, tau: float, normalize_flag: bool) -> Dict:
    """Run a transformer prediction and return a plain dict matching PredictResponse."""

    import torch  # type: ignore

    norm_text = normalize_text(text, norm_cfg) if (norm_cfg and normalize_flag) else text
    enc = tokenizer([norm_text], truncation=True, padding=False, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        proba = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    pred_idx = int(proba.argmax())
    pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
    pred_set = [labels[i] for i, p in enumerate(proba) if (1.0 - float(p)) <= tau]
    probabilities = {f"proba_{lbl}": float(proba[i]) for i, lbl in enumerate(labels)}
    unc = _compute_uncertainty(np.asarray(proba, dtype=float))

    return {
        "text": text,
        "normalized_text": norm_text,
        "pred_label": pred_label,
        "probabilities": probabilities,
        "conformal_set": pred_set,
        "tau": float(tau),
        "model_type": "transformer",
        "max_proba": unc["max_proba"],
        "entropy": unc["entropy"],
        "pred_set_size": len(pred_set),
        "importance": None,
    }


# ---------------------------------------------------------------------------
# Auth & validation helpers
# ---------------------------------------------------------------------------

def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """Simple API key guard.

    - If DEMO_API_KEY is unset/empty, auth is disabled and any request passes.
    - If DEMO_API_KEY is set, require the same value in X-API-Key.
    """

    if not API_KEY:
        # Auth disabled (e.g. local demo).
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def validate_text_length(text: str) -> None:
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"text too long (>{MAX_TEXT_LEN} characters). Consider truncating or using batch processing.",
        )


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

def create_app(model_dir: Path, tau: float) -> FastAPI:
    """Create FastAPI app with baseline and optional transformer model support.

    Parameters
    ----------
    model_dir:
        Base models directory. We look for a baseline subdir and an optional
        transformer subdir by name. If a concrete baseline directory is given
        instead, this still works.
    tau:
        Default conformal threshold used when the client does not supply one.
    """

    # Baseline model (required for demo)
    baseline_dir = model_dir / BASELINE_MODEL_NAME if model_dir.is_dir() else model_dir
    vectorizer, model, norm_cfg = load_baseline_model(baseline_dir)

    # Optional transformer model: explicit sibling directory under the same base
    transformer_cfg = None
    tfm_dir = model_dir / TRANSFORMER_MODEL_NAME if model_dir.is_dir() else None
    if tfm_dir is not None and tfm_dir.exists():
        try:
            tokenizer, t_model, t_norm_cfg, t_labels, device = predict_transformer.load_model(tfm_dir)
            transformer_cfg = {
                "tokenizer": tokenizer,
                "model": t_model,
                "norm_cfg": t_norm_cfg,
                "labels": t_labels,
                "device": device,
                "dir": tfm_dir,
            }
        except SystemExit:
            # If loading transformer fails, we gracefully degrade to baseline-only.
            transformer_cfg = None

    app = FastAPI(title="Khmer Sentiment Demo API", version="1.5.0")

    # CORS configuration: allow all origins by default for demo, but this can be
    # restricted via DEMO_CORS_ORIGINS (comma-separated list).
    origins_env = os.getenv("DEMO_CORS_ORIGINS", "*")
    if origins_env.strip() == "*":
        allow_origins = ["*"]
    else:
        allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # ---- Small internal helpers bound to this app ----

    def _resolve_model_type(req_model_type: Optional[str]) -> str:
        """Resolve requested model type to a concrete one.

        - If the client explicitly requests "baseline" or "transformer", honor it
          when possible (falling back to baseline if transformer is unavailable).
        - If not provided, choose DEFAULT_MODEL_TYPE when transformer is available,
          otherwise use baseline.
        """

        if req_model_type in ("baseline", "transformer"):
            if req_model_type == "transformer" and transformer_cfg is None:
                return "baseline"
            return req_model_type
        if transformer_cfg is None:
            return "baseline"
        return DEFAULT_MODEL_TYPE

    def _resolve_normalize_flag(req_normalize: Optional[bool]) -> bool:
        if req_normalize is None:
            return True
        return bool(req_normalize)

    # ---- Exception handlers ----

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):  # type: ignore[override]
        """Catch-all handler to avoid leaking stack traces in production.

        For debugging locally, you can still see logs in the server console.
        """

        logger.exception("Unhandled exception during request %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "internal server error"},
        )

    # ---- Public endpoints ----

    @app.get("/health")
    def health():  # type: ignore[override]
        """Liveness/readiness endpoint for monitoring.

        This returns a very small payload suitable for health checks.
        """

        return {
            "status": "ok",
            "model_dir": str(model_dir),
            "labels": LABELS,
            "has_transformer": bool(transformer_cfg is not None),
        }

    @app.get("/meta", response_model=MetaResponse)
    def meta():  # type: ignore[override]
        """Return machine-readable metadata about available models.

        External clients can call this once at startup to discover:
        - which model types are available,
        - which labels are used,
        - default tau and normalization behavior,
        - (optionally) coarse-grained validation metrics per model.
        """

        supported = ["baseline"] + (["transformer"] if transformer_cfg is not None else [])
        default_type = DEFAULT_MODEL_TYPE if transformer_cfg is not None else "baseline"

        metrics: Dict[str, MetaMetrics] = {}
        metrics["baseline"] = MetaMetrics(accuracy=None, f1_macro=None)
        if transformer_cfg is not None:
            metrics["transformer"] = MetaMetrics(accuracy=None, f1_macro=None)

        return MetaResponse(
            model_dir=str(model_dir),
            labels=LABELS,
            tau=float(tau),
            supported_model_types=supported,
            default_model_type=default_type,
            default_normalize=True,
            has_transformer=bool(transformer_cfg is not None),
            metrics=metrics or None,
        )

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
    def predict(req: PredictRequest):  # type: ignore[override]
        """Predict sentiment and conformal prediction set for a single input.

        This endpoint is suitable for both UI and programmatic use.
        """

        validate_text_length(req.text)
        local_tau = float(req.tau) if req.tau is not None else tau
        model_type = _resolve_model_type(req.model_type)
        normalize_flag = _resolve_normalize_flag(req.normalize)

        logger.info(
            "predict: model_type=%s tau=%.4f normalize=%s text_len=%d",
            model_type,
            local_tau,
            normalize_flag,
            len(req.text),
        )

        if model_type == "transformer" and transformer_cfg is not None:
            res = _predict_transformer(
                transformer_cfg["tokenizer"],
                transformer_cfg["model"],
                transformer_cfg["norm_cfg"],
                transformer_cfg["labels"],
                transformer_cfg["device"],
                req.text,
                local_tau,
                normalize_flag,
            )
        else:
            res = _predict_baseline(vectorizer, model, norm_cfg, req.text, local_tau, normalize_flag)
        return res

    @app.post("/predict/batch", response_model=List[PredictResponse], dependencies=[Depends(verify_api_key)])
    def predict_batch(reqs: List[PredictRequest]):  # type: ignore[override]
        """Batch prediction endpoint.

        This accepts a JSON list of request objects and returns a list of
        predictions. It mirrors the schema of the single /predict endpoint
        and is intended for scripting or offline analysis.
        """

        if len(reqs) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"too many items in batch (>{MAX_BATCH_SIZE}). Split your input into smaller batches.",
            )

        out: List[Dict] = []
        for r in reqs:
            validate_text_length(r.text)
            local_tau = float(r.tau) if r.tau is not None else tau
            model_type = _resolve_model_type(r.model_type)
            normalize_flag = _resolve_normalize_flag(r.normalize)
            if model_type == "transformer" and transformer_cfg is not None:
                out.append(
                    _predict_transformer(
                        transformer_cfg["tokenizer"],
                        transformer_cfg["model"],
                        transformer_cfg["norm_cfg"],
                        transformer_cfg["labels"],
                        transformer_cfg["device"],
                        r.text,
                        local_tau,
                        normalize_flag,
                    )
                )
            else:
                out.append(_predict_baseline(vectorizer, model, norm_cfg, r.text, local_tau, normalize_flag))

        logger.info("predict_batch: size=%d", len(out))
        return out

    @app.post("/predict/compare", response_model=CompareResponse, dependencies=[Depends(verify_api_key)])
    def predict_compare(req: PredictRequest):  # type: ignore[override]
        """Return predictions from baseline and transformer (if available).

        This is a convenience endpoint for external clients that want to
        perform model comparison on the same input without issuing two
        separate HTTP calls.
        """

        validate_text_length(req.text)
        local_tau = float(req.tau) if req.tau is not None else tau
        normalize_flag = _resolve_normalize_flag(req.normalize)

        logger.info(
            "predict_compare: tau=%.4f normalize=%s text_len=%d has_transformer=%s",
            local_tau,
            normalize_flag,
            len(req.text),
            bool(transformer_cfg is not None),
        )

        baseline_dict = _predict_baseline(vectorizer, model, norm_cfg, req.text, local_tau, normalize_flag)
        baseline_res = PredictResponse(**baseline_dict)

        transformer_res: Optional[PredictResponse] = None
        if transformer_cfg is not None:
            tfm_dict = _predict_transformer(
                transformer_cfg["tokenizer"],
                transformer_cfg["model"],
                transformer_cfg["norm_cfg"],
                transformer_cfg["labels"],
                transformer_cfg["device"],
                req.text,
                local_tau,
                normalize_flag,
            )
            transformer_res = PredictResponse(**tfm_dict)

        return CompareResponse(baseline=baseline_res, transformer=transformer_res)

    @app.get("/", response_class=HTMLResponse)
    def root() -> Response:  # type: ignore[override]
        """Serve the simple HTML UI for interactive demo.

        This keeps the API self-contained: visiting the root path in a
        browser brings up the interactive demo, while external clients can
        consume the JSON endpoints above.
        """

        ui_path = Path(__file__).parent / "demo_ui.html"
        try:
            html = ui_path.read_text(encoding="utf-8")
        except Exception:
            # Fallback HTML if file missing
            return HTMLResponse(
                content="<h1>Khmer Sentiment Analysis Demo</h1><p>demo_ui.html not found.</p>",
                status_code=200,
            )
        return HTMLResponse(content=html, status_code=200)

    return app


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the API server.

    This is a thin convenience wrapper around uvicorn.run so that the module
    can also be imported for testing without requiring uvicorn.
    """

    ap = argparse.ArgumentParser(description="Run the Khmer sentiment demo API.")
    ap.add_argument(
        "--model_dir",
        default="models",
        help="Base models directory (containing baseline/transformer subdirs)",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Default conformal TPS threshold tau (1 - p) <= tau for inclusion",
    )
    ap.add_argument("--host", default="127.0.0.1", help="Host for the API server")
    ap.add_argument("--port", type=int, default=8000, help="Port for the API server")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)

    # Defer import of uvicorn so that importing this module for tests doesn't require it.
    try:
        import uvicorn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency for running the server. Install with: pip install uvicorn[standard]\n"
            f"Underlying import error: {e}"
        )

    app = create_app(model_dir, tau=args.tau)

    logger.info(
        "Starting server on %s:%d (model_dir=%s, default_tau=%.4f, api_key_configured=%s)",
        args.host,
        args.port,
        model_dir,
        args.tau,
        bool(API_KEY),
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
