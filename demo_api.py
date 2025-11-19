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
from typing import Dict, List

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

LABELS = ["POS", "NEG", "NEU"]


class PredictRequest(BaseModel):
    text: str


class MetaResponse(BaseModel):
    model_dir: str
    labels: List[str]
    tau: float


class PredictResponse(BaseModel):
    text: str
    normalized_text: str
    pred_label: str
    probabilities: Dict[str, float]
    conformal_set: List[str]
    tau: float


def load_baseline_model(model_dir: Path):
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


def predict_with_conformal(vectorizer, model, norm_cfg, text: str, tau: float) -> Dict:
    # Normalize
    norm_text = normalize_text(text, norm_cfg) if norm_cfg else text

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

    return {
        "text": text,
        "normalized_text": norm_text,
        "pred_label": pred_label,
        "probabilities": probabilities,
        "conformal_set": pred_set,
        "tau": float(tau),
    }


def create_app(model_dir: Path, tau: float) -> FastAPI:
    vectorizer, model, norm_cfg = load_baseline_model(model_dir)

    app = FastAPI(title="Khmer Sentiment Demo API", version="1.1.0")

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):  # type: ignore[override]
        res = predict_with_conformal(vectorizer, model, norm_cfg, req.text, tau)
        return res

    @app.get("/meta", response_model=MetaResponse)
    def meta():  # type: ignore[override]
        return MetaResponse(model_dir=str(model_dir), labels=LABELS, tau=float(tau))

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
    ap.add_argument("--model_dir", default="models/baseline_chargram", help="Directory with vectorizer.pkl and model.pkl")
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
