#!/usr/bin/env python3
"""Minimal interactive demo API for Khmer sentiment with conformal prediction.

Features:
- Single POST /predict endpoint
- Input: Khmer text
- Output: JSON with
    - normalized_text
    - pred_label
    - per-class probabilities
    - conformal prediction set (thresholded probability set)

This reuses the existing modeling code:
- Baseline TF-IDF + LogisticRegression model (modeling/predict.py logic)
- Conformal TPS-style prediction set (similar to modeling/conformal_predict.py)
- Text normalization from modeling/text_normalization.py

Usage:
  # 1) Ensure you have a trained baseline model directory, e.g. by running run_demo.bat
  #    The default here assumes models/baseline_chargram exists.

  # 2) Install extra deps (only once):
  #    pip install fastapi uvicorn[standard]

  # 3) Start the API:
  #    python demo_api.py --model_dir models/baseline_chargram --tau 0.5
  #    # or if you have a calibrated model with better probabilities:
  #    # python demo_api.py --model_dir models/baseline_chargram_cal --tau 0.5

  # 4) Call from curl:
  #    curl -X POST "http://127.0.0.1:8000/predict" ^
  #         -H "Content-Type: application/json" ^
  #         -d "{\"text\": \"អរគុណច្រើន សេវាកម្មល្អបំផុត 🙄\"}"

For a browser UI, see the auto-generated Swagger UI:
  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import sys

try:
    import numpy as np  # type: ignore
    import joblib
except Exception as e:
    raise SystemExit(
        "Missing dependency for baseline model. Install with: pip install scikit-learn joblib numpy\n"
        f"Underlying import error: {e}"
    )

try:
    from fastapi import FastAPI, Response
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except Exception as e:
    raise SystemExit(
        "Missing dependency for API. Install with: pip install fastapi uvicorn[standard]\n"
        f"Underlying import error: {e}"
    )

from modeling.text_normalization import load_norm_config, normalize_text
from modeling import predict_transformer

LABELS = ["POS", "NEG", "NEU"]
DEFAULT_MODEL_TYPE = "baseline"  # "baseline" or "transformer"

# Hard-coded demo mapping: baseline vs transformer model directories (under a base models/ dir)
BASELINE_MODEL_NAME = "baseline_chargram"
TRANSFORMER_MODEL_NAME = "transformer_khmer_demo"


class PredictRequest(BaseModel):
    text: str
    tau: float | None = None
    model_type: str | None = None  # "baseline" or "transformer"
    normalize: bool | None = None


class MetaResponse(BaseModel):
    model_dir: str
    labels: List[str]
    tau: float
    supported_model_types: List[str]
    default_model_type: str
    default_normalize: bool


class PredictResponse(BaseModel):
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


def load_baseline_model(model_dir: Path):
    """Load baseline vectorizer + model from a directory containing vectorizer.pkl and model.pkl."""
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


def create_app(model_dir: Path, tau: float) -> FastAPI:
    """Create FastAPI app with baseline and optional transformer model support.

    model_dir is interpreted as the base models/ directory. We look for a baseline
    subdir and an optional transformer subdir by name.
    """

    # Baseline model (required for demo)
    # If model_dir is already a concrete baseline directory, this still works.
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
            transformer_cfg = None

    app = FastAPI(title="Khmer Sentiment Demo API", version="1.3.0")

    @app.get("/health")
    def health():  # type: ignore[override]
        return {
            "status": "ok",
            "model_dir": str(model_dir),
            "labels": LABELS,
            "has_transformer": bool(transformer_cfg is not None),
        }

    def _resolve_model_type(req_model_type: Optional[str]) -> str:
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

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):  # type: ignore[override]
        local_tau = float(req.tau) if req.tau is not None else tau
        model_type = _resolve_model_type(req.model_type)
        normalize_flag = _resolve_normalize_flag(req.normalize)
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

    @app.post("/predict/batch")
    def predict_batch(reqs: List[PredictRequest]):  # type: ignore[override]
        out: List[Dict] = []
        for r in reqs:
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
        return out

    @app.get("/meta", response_model=MetaResponse)
    def meta():  # type: ignore[override]
        supported = ["baseline"] + (["transformer"] if transformer_cfg is not None else [])
        default_type = DEFAULT_MODEL_TYPE if transformer_cfg is not None else "baseline"
        return MetaResponse(
            model_dir=str(model_dir),
            labels=LABELS,
            tau=float(tau),
            supported_model_types=supported,
            default_model_type=default_type,
            default_normalize=True,
        )

    @app.get("/", response_class=HTMLResponse)
    def root() -> Response:  # type: ignore[override]
        """Serve the simple HTML UI for interactive demo."""
        ui_path = Path(__file__).parent / "demo_ui.html"
        try:
            html = ui_path.read_text(encoding="utf-8")
        except Exception:
            # Fallback JSON if file missing
            return HTMLResponse(
                content="<h1>Khmer Sentiment Analysis Demo</h1><p>demo_ui.html not found.</p>",
                status_code=200,
            )
        return HTMLResponse(content=html, status_code=200)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a minimal API demo for Khmer sentiment.")
    ap.add_argument(
        "--model_dir",
        default="models",
        help="Base models directory (containing baseline/transformer subdirs)",
    )
    ap.add_argument("--tau", type=float, default=0.5, help="Conformal TPS threshold tau (1 - p) <= tau for inclusion")
    ap.add_argument("--host", default="127.0.0.1", help="Host for the API server")
    ap.add_argument("--port", type=int, default=8000, help="Port for the API server")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)

    # Defer import of uvicorn so that module import for testing doesn't require it.
    try:
        import uvicorn  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency for running the server. Install with: pip install uvicorn[standard]\n"
            f"Underlying import error: {e}"
        )

    app = create_app(model_dir, tau=args.tau)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
