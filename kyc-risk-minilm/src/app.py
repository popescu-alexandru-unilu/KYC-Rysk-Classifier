# src/app.py
# REST wrapper around your trained MiniLM + policy/why
# WHERE CHANGES NEED TO BE MADE:
CKPT_PATH = "minilm_cls_best.pt"   # <<< CHANGE if your checkpoint name/path differs
MAX_LEN   = 256                    # <<< CHANGE if you want 384/512

import torch, yaml, hmac, hashlib, json, time, os, re
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from infer_minilm import (
    load_model, classify_batch, sanctions_hit, build_reasons,
    override_high_payload, LABELS, CFG
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = load_model(CKPT_PATH, device)  # load once

app = FastAPI(title="KYC Risk Classifier", version="v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Req(BaseModel):
    text: str = Field(min_length=10, max_length=5000)
    override: bool = True
    format: str = "chat"          # "json" or "chat"
    max_len: int = MAX_LEN
    topk_reasons: int = 3

def redact(t: str) -> str:
    t = re.sub(r'\b[0-9]{6,}\b', '<REDACTED_NUM>', t)
    t = re.sub(r'\b[A-Z0-9]{6,9}\b', '<REDACTED_ID>', t)
    return t

def log_decision(inp: str, out: dict):
    SECRET = os.getenv("AUDIT_HMAC", "change_me")
    blob = {
        "ts": time.time(),
        "text_sha": hashlib.sha256(inp.encode()).hexdigest(),
        "label": out["label"],
        "rule": out.get("rule"),
        "probs": out["probs"],
        "why": out.get("why")
    }
    mac = hmac.new(SECRET.encode(), json.dumps(blob, sort_keys=True).encode(), "sha256").hexdigest()
    blob["hmac"] = mac
    with open("logs/audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(blob) + "\n")

OV = Counter("override_total", "policy overrides")
PRED = Counter("pred_total", "preds", ["label"])
LAT = Histogram("latency_seconds", "inference latency")

@app.get("/health")
def health():
    config_loaded = True  # Assuming CFG loaded since imported
    return {"status": "ok", "device": device, "labels": LABELS, "config_loaded": config_loaded}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

def _pct(probs: dict) -> str:
    return ", ".join(f"{k}={v*100:.1f}%" for k, v in probs.items())

def format_chat(result: dict, text: str, topk: int = 3) -> dict:
    """Return a chat-friendly payload (kept as dict for API)."""
    label = result["label"]; rule = result.get("rule", "model_only")
    probs = result["probs"]; why = (result.get("why") or [])[:topk]
    emoji = {"low":"🟢","medium":"🟡","high":"🔴"}.get(label, "▫️")
    return {
        "summary": f"{emoji} Risk: {label.upper()}",
        "decision": "Policy override: sanctions code detected" if rule=="sanctions_override" else "Model classification",
        "confidence": _pct(probs),
        "why": why
    }

@app.post("/classify_batch")
def classify_batch_endpoint(reqs: list[dict]):
    results = []
    for req in reqs:
        text = req['text']
        override = req.get('override', True)
        format = req.get('format', 'json')
        max_len = req.get('max_len', MAX_LEN)
        topk_reasons = req.get('topk_reasons', 3)
        # Similar logic
        if override and sanctions_hit(text):
            res = override_high_payload()
            res["why"] = build_reasons(text, res["label"], res["probs"], res["rule"])
            OV.inc()
        else:
            with LAT.time():
                res = classify_batch(model, device, [text], max_len)[0]
            res["rule"] = "model_only"
            res["why"] = build_reasons(text, res["label"], res["probs"], res["rule"])
        PRED.labels(label=res["label"]).inc()
        log_decision(redact(text), res)
        results.append(res if format == "json" else format_chat(res, text, topk_reasons))
    return results

@app.post("/classify")
def classify(req: Req):
    text = req.text
    with LAT.time():
        if req.override and sanctions_hit(text):
            res = override_high_payload()
            res["why"] = build_reasons(text, res["label"], res["probs"], res["rule"])
            OV.inc()
        else:
            res = classify_batch(model, device, [text], req.max_len)[0]
            res["rule"] = "model_only"
            res["why"]  = build_reasons(text, res["label"], res["probs"], res["rule"])
    PRED.labels(label=res["label"]).inc()
    log_decision(redact(text), res)
    return res if req.format=="json" else format_chat(res, text, req.topk_reasons)
