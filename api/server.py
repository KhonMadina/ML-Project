#!/usr/bin/env python3
"""
FastAPI server exposing versioned endpoints for inference, model management,
stress transformations/evaluation, dataset validation, IAA, tokenization preview,
and report exports/overview.

Run:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Requirements:
  pip install fastapi uvicorn pydantic[dotenv]
  # plus project requirements for models (transformers, scikit-learn, torch, pandas, numpy)
"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Project imports
from modeling.text_normalization import load_norm_config, normalize_corpus
from modeling import stress_eval as stress

app = FastAPI(title="Khmer+English Sentiment API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CurrentModel:
    def __init__(self) -> None:
        self.model_dir: Optional[Path] = None
        self.mode: Optional[str] = None  # 'baseline' or 'transformer'
        self.vectorizer = None
        self.model = None
        self.tokenizer = None
        self.norm_cfg: Optional[Dict[str, Any]] = None
        self.label_order: List[str] = ["POS", "NEG", "NEU"]

    def unload(self):
        self.model_dir = None
        self.mode = None
        self.vectorizer = None
        self.model = None
        self.tokenizer = None
        self.norm_cfg = None

    def load(self, model_dir: Path):
        vec_p = model_dir / "vectorizer.pkl"
        mdl_p = model_dir / "model.pkl"
        if vec_p.exists() and mdl_p.exists():
            import joblib
            self.vectorizer = joblib.load(vec_p)
            self.model = joblib.load(mdl_p)
            self.tokenizer = None
            self.mode = "baseline"
        else:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Transformers not available: {e}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
                self.vectorizer = None
                self.mode = "transformer"
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Unable to load model from {model_dir}: {e}")
        self.model_dir = model_dir
        self.norm_cfg = load_norm_config(model_dir)

    def predict_batch(self, texts: List[str], max_length: Optional[int] = None, top_k: int = 1):
        top_k = max(1, min(5, int(top_k)))
        if self.norm_cfg:
            texts = normalize_corpus(texts, self.norm_cfg)
        if self.mode == "baseline":
            X = self.vectorizer.transform(texts)
            preds = self.model.predict(X)
            proba = self.model.predict_proba(X) if hasattr(self.model, "predict_proba") else None
            out = []
            for i, p in enumerate(preds):
                label = str(p)
                if proba is not None:
                    probs = {str(c): float(proba[i][j]) for j, c in enumerate(getattr(self.model, "classes_", []))}
                    top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
                else:
                    probs = {}
                    top = [(label, 1.0)]
                out.append({
                    "label": label,
                    "top": [{"label": l, "prob": float(p)} for l, p in top],
                    "probs": probs,
                    "tokens": None,
                    "seq_len": None,
                    "truncated": None,
                })
            return out
        elif self.mode == "transformer":
            import torch
            import numpy as np
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            out = []
            for i in range(0, len(texts), 32):
                batch = texts[i:i+32]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length or self.tokenizer.model_max_length, return_tensors="pt")
                truncated_flags = None
                try:
                    truncated_flags = [len(ids) >= (max_length or self.tokenizer.model_max_length) for ids in enc["input_ids"].tolist()]
                except Exception:
                    truncated_flags = [None] * len(batch)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = self.model(**enc).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    top_idx = np.argsort(-probs, axis=1)[:, :top_k]
                for j, text in enumerate(batch):
                    top = [(self._label_from_idx(int(idx)), float(probs[j][idx])) for idx in top_idx[j]]
                    out.append({
                        "label": top[0][0],
                        "top": [{"label": l, "prob": p} for l, p in top],
                        "probs": {self._label_from_idx(k): float(v) for k, v in enumerate(probs[j])},
                        "tokens": None,
                        "seq_len": int(enc["input_ids"][j].shape[0]) if hasattr(enc["input_ids"], "shape") else None,
                        "truncated": truncated_flags[j] if truncated_flags is not None else None,
                    })
            return out
        else:
            raise HTTPException(status_code=400, detail="No model loaded")

    def tokenize(self, text: str, max_length: Optional[int] = None) -> Dict[str, Any]:
        if self.mode != "transformer" or self.tokenizer is None:
            raise HTTPException(status_code=400, detail="Tokenization preview requires a transformer model")
        if self.norm_cfg:
            text = normalize_corpus([text], self.norm_cfg)[0]
        enc = self.tokenizer(text, padding=False, truncation=True, max_length=max_length or self.tokenizer.model_max_length, return_tensors=None)
        ids = enc.get("input_ids", [])
        toks = self.tokenizer.convert_ids_to_tokens(ids) if hasattr(self.tokenizer, "convert_ids_to_tokens") else []
        truncated = len(ids) >= (max_length or self.tokenizer.model_max_length)
        return {"text": text, "tokens": toks, "ids": ids, "seq_len": len(ids), "truncated": truncated}

    def _label_from_idx(self, idx: int) -> str:
        try:
            id2label = getattr(getattr(self.model, "config", None), "id2label", None)
            if isinstance(id2label, dict) and idx in id2label:
                return str(id2label[idx])
        except Exception:
            pass
        if 0 <= idx < len(self.label_order):
            return self.label_order[idx]
        return str(idx)


CURRENT = CurrentModel()


# Schemas
class InferItem(BaseModel):
    id: Optional[str] = None
    text: str
    lang: Optional[str] = None

class InferRequest(BaseModel):
    items: List[InferItem]
    top_k: int = Field(default=1, ge=1, le=5)
    max_length: Optional[int] = None

class InferResponseItem(BaseModel):
    id: Optional[str]
    label: str
    top: List[Dict[str, float]]
    probs: Optional[Dict[str, float]]
    tokens: Optional[List[str]]
    seq_len: Optional[int]
    truncated: Optional[bool]

class InferResponse(BaseModel):
    model_dir: Optional[str]
    mode: Optional[str]
    results: List[InferResponseItem]

class SelectModelRequest(BaseModel):
    model_dir: str

class StressTransformRequest(BaseModel):
    preset: str
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    prob: float = Field(default=0.7, ge=0.0, le=1.0)
    texts: List[str]

class StressTransformResponse(BaseModel):
    preset: str
    severity: float
    prob: float
    texts: List[str]

class StressEvalItem(BaseModel):
    id: str
    text: str
    label: str
    category: Optional[str] = "overall"

class StressEvalRequest(BaseModel):
    items: List[StressEvalItem]
    preset: Optional[str] = None
    severity: float = 0.5
    prob: float = 0.7

class ReportExportRequest(BaseModel):
    models: Optional[List[str]] = None
    glob: Optional[str] = None
    out_dir: str = "reports/final"
    pred_pair: Optional[List[str]] = None
    stress: Optional[List[str]] = None
    top_k: int = 10

class TokenizeRequest(BaseModel):
    text: str
    max_length: Optional[int] = None

class ValidateRequest(BaseModel):
    input_csv: str
    group_column: Optional[str] = None
    lang_column: Optional[str] = None
    split_column: Optional[str] = None
    allowed_langs: Optional[List[str]] = None

class IAARequest(BaseModel):
    input_csv: str
    lang_column: Optional[str] = None


# Endpoints
@app.get("/v1/health")
def health():
    return {
        "status": "ok",
        "model_dir": str(CURRENT.model_dir) if CURRENT.model_dir else None,
        "mode": CURRENT.mode,
    }


@app.get("/v1/models")
def list_models():
    roots = [Path("models"), Path("runs")]
    out = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.iterdir():
            if not p.is_dir():
                continue
            entry = {"path": str(p), "name": p.name, "has_metrics": (p / "metrics.json").exists()}
            if (p / "vectorizer.pkl").exists() and (p / "model.pkl").exists():
                entry["type"] = "baseline"
            elif (p / "config.json").exists() or (p / "pytorch_model.bin").exists():
                entry["type"] = "transformer"
            else:
                entry["type"] = "unknown"
            if entry["has_metrics"]:
                try:
                    m = json.loads((p / "metrics.json").read_text(encoding="utf-8"))
                    test = m.get("test", {})
                    entry["test_accuracy"] = test.get("eval_accuracy", test.get("accuracy", None))
                    entry["test_f1_macro"] = test.get("eval_f1_macro", test.get("f1_macro", None))
                except Exception:
                    pass
            out.append(entry)
    return {"models": out}


@app.get("/v1/models/card")
def model_card(model_dir: Optional[str] = None):
    p = Path(model_dir) if model_dir else CURRENT.model_dir
    if p is None:
        raise HTTPException(status_code=400, detail="No model provided or loaded")
    data: Dict[str, Any] = {"path": str(p)}
    try:
        m = json.loads((p / "metrics.json").read_text(encoding="utf-8")) if (p / "metrics.json").exists() else {}
        data["metrics"] = m
    except Exception:
        data["metrics"] = {}
    try:
        data["normalization"] = json.loads((p / "normalization.json").read_text(encoding="utf-8")) if (p / "normalization.json").exists() else {}
    except Exception:
        data["normalization"] = {}
    try:
        data["tokenizer_report"] = json.loads((p / "tokenizer_report.json").read_text(encoding="utf-8")) if (p / "tokenizer_report.json").exists() else {}
    except Exception:
        data["tokenizer_report"] = {}
    for fn in ["val_reliability.png", "test_reliability.png"]:
        data[fn] = str((p / fn)) if (p / fn).exists() else None
    return data


@app.post("/v1/models/select")
def select_model(req: SelectModelRequest):
    p = Path(req.model_dir)
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail=f"Model directory not found: {req.model_dir}")
    CURRENT.unload()
    CURRENT.load(p)
    return {"ok": True, "model_dir": str(p), "mode": CURRENT.mode}


@app.post("/v1/infer")
def infer(req: InferRequest):
    if CURRENT.mode is None:
        raise HTTPException(status_code=400, detail="No model loaded; call /v1/models/select first")
    texts = [it.text for it in req.items]
    results = CURRENT.predict_batch(texts, max_length=req.max_length, top_k=req.top_k)
    out_items: List[InferResponseItem] = []
    for it, res in zip(req.items, results):
        out_items.append(InferResponseItem(
            id=it.id,
            label=res["label"],
            top=res.get("top", []),
            probs=res.get("probs"),
            tokens=res.get("tokens"),
            seq_len=res.get("seq_len"),
            truncated=res.get("truncated"),
        ))
    return {"model_dir": str(CURRENT.model_dir), "mode": CURRENT.mode, "results": [i.dict() for i in out_items]}


@app.post("/v1/tokenize")
def tokenize(req: TokenizeRequest):
    if CURRENT.mode is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    return CURRENT.tokenize(req.text, max_length=req.max_length)


@app.post("/v1/stress/transform")
def stress_transform(req: StressTransformRequest):
    if req.preset not in stress.PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{req.preset}'")
    fn = stress.PRESETS[req.preset]
    out = [fn(t, req.severity, req.prob) for t in req.texts]
    return {"preset": req.preset, "severity": req.severity, "prob": req.prob, "texts": out}


@app.post("/v1/stress/eval")
def stress_eval(req: StressEvalRequest):
    if CURRENT.mode is None:
        raise HTTPException(status_code=400, detail="No model loaded; call /v1/models/select first")
    rows = [{"id": it.id, "text": it.text, "label": it.label.upper(), "category": it.category or "overall"} for it in req.items]
    texts = [r["text"] for r in rows]
    if CURRENT.norm_cfg:
        texts = normalize_corpus(texts, CURRENT.norm_cfg)
    if req.preset:
        fn = stress.PRESETS.get(req.preset)
        if fn is None:
            raise HTTPException(status_code=400, detail=f"Unknown preset '{req.preset}'")
        texts = [fn(t, req.severity, req.prob) for t in texts]
    gold = [r["label"] for r in rows]
    cats = [r["category"] for r in rows]

    if CURRENT.mode == "baseline":
        preds = stress.predict_baseline(CURRENT.vectorizer, CURRENT.model, texts)
    else:
        preds = stress.predict_transformer(CURRENT.tokenizer, CURRENT.model, texts)

    acc, f1m, report, cm = stress.compute_metrics(gold, preds)
    per_cat: Dict[str, Dict[str, Any]] = {}
    for cat in sorted(set(cats)):
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        g = [gold[i] for i in idx]
        p = [preds[i] for i in idx]
        a2, f2, rpt2, cm2 = stress.compute_metrics(g, p)
        per_cat[cat] = {"count": len(idx), "accuracy": a2, "f1_macro": f2, "report": rpt2, "confusion": cm2}

    return {"overall": {"accuracy": acc, "f1_macro": f1m, "confusion": cm, "report": report}, "per_category": per_cat}


@app.post("/v1/reports/export")
def reports_export(req: ReportExportRequest):
    import subprocess, sys as _sys
    cmd = [_sys.executable, str(Path("tools") / "export_report.py"), "--out_dir", req.out_dir]
    if req.glob: cmd.extend(["--glob", req.glob])
    if req.models: cmd.extend(["--models", *req.models])
    for p in (req.pred_pair or []): cmd.extend(["--pred_pair", p])
    for s in (req.stress or []): cmd.extend(["--stress", s])
    cmd.extend(["--top_k", str(int(req.top_k))])
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    if rc != 0:
        raise HTTPException(status_code=500, detail=f"export_report failed with code {rc}")
    return {"ok": True, "out_dir": req.out_dir, "seconds": dt}


@app.get("/v1/reports/overview")
def reports_overview(dir: str = "reports/final"):
    import csv
    out: Dict[str, Any] = {}
    base = Path(dir)
    def read_csv(path: Path):
        rows = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        return rows
    out["overall"] = read_csv(base / "overall.csv")
    out["per_language"] = read_csv(base / "per_language.csv")
    out["calibration"] = read_csv(base / "calibration.csv")
    # List plots
    plots_dir = base / "plots"
    out["plots"] = [str(p) for p in plots_dir.glob("*.png")] if plots_dir.exists() else []
    return out


@app.post("/v1/validate")
def validate_dataset(req: ValidateRequest):
    # Calls tools/validate_dataset.py and returns summary + issues. Input CSV must be a local path.
    from tools import validate_dataset as vds
    path = Path(req.input_csv)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"CSV not found: {path}")
    # Read
    rows, fields = vds.read_csv_rows(path)
    msgs = vds.validate(rows, fields, allow_extra_labels=False, group_column=req.group_column, lang_column=req.lang_column, split_column=req.split_column, allowed_langs=req.allowed_langs, allow_extra_langs=False)
    summary = vds.summarize(rows, group_column=req.group_column, lang_column=req.lang_column, split_column=req.split_column, allowed_langs=req.allowed_langs)
    return {"issues": msgs, "summary": summary}


@app.post("/v1/iaa")
def iaa_compute(req: IAARequest):
    from tools.compute_iaa import compute_iaa as _compute
    import csv as _csv
    path = Path(req.input_csv)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"CSV not found: {path}")
    # Minimal CSV read
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            rows.append(r)
    res = _compute(rows, req.lang_column)
    return res
