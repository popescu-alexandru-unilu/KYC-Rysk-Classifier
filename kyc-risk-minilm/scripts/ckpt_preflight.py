#!/usr/bin/env python3
"""
Checkpoint preflight validator.

Run before `docker compose up` to ensure the mounted MODEL_CKPT is a valid
Hugging Face classification checkpoint directory or a .pt/.pth file.

Usage:
  python kyc-risk-minilm/scripts/ckpt_preflight.py [--path PATH]

Rules:
- If PATH is a directory: require config.json (with model_type),
  pytorch_model.bin, and at least one tokenizer file present.
- If PATH is a file: accept .pt/.pth.
- Exit non-zero on failure so CI/CD can block deploys.
"""
import argparse, json, os, sys
from pathlib import Path

TOK_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.txt", "merges.txt", "vocab.json"
]

def validate(path: Path):
    out = {
        "ok": False,
        "type": None,
        "exists": False,
        "reason": None,
        "num_labels": None,
        "id2label": None,
        "missing": [],
    }
    if not path.exists():
        out["reason"] = f"not found: {path}"
        return out
    out["exists"] = True
    if path.is_dir():
        out["type"] = "hf"
        # required
        if not (path / "config.json").exists():
            out["missing"].append("config.json")
        if not (path / "pytorch_model.bin").exists():
            out["missing"].append("pytorch_model.bin")
        if not any((path / f).exists() for f in TOK_FILES):
            out["missing"].append("tokenizer_files")
        # config introspection
        if (path / "config.json").exists():
            try:
                cfg = json.loads((path / "config.json").read_text("utf-8"))
                out["num_labels"] = cfg.get("num_labels")
                out["id2label"] = cfg.get("id2label")
                if not cfg.get("model_type"):
                    out["missing"].append("config.model_type")
            except Exception as e:
                out["reason"] = f"config.json parse failed: {e}"
        if not out["missing"] and not out.get("reason"):
            out["ok"] = True
        else:
            if not out.get("reason"):
                out["reason"] = f"missing: {', '.join(out['missing'])}"
        return out
    else:
        ext = path.suffix.lower()
        if ext in (".pt", ".pth"):
            out["type"] = "torch"
            out["ok"] = True
            return out
        out["type"] = "unknown"
        out["reason"] = f"unsupported file ext: {ext}"
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="Checkpoint path (dir or file). Defaults to ./kyc-risk-minilm/minilm_cls_best or .pt file.")
    args = ap.parse_args()

    if args.path:
        path = Path(args.path)
    else:
        # Try common locations relative to repo root
        here = Path(__file__).resolve().parents[2]  # repo root
        candidates = [
            here / "kyc-risk-minilm" / "minilm_cls_best",
            here / "kyc-risk-minilm" / "minilm_cls_best.pt",
        ]
        cand = next((p for p in candidates if p.exists()), candidates[0])
        path = cand

    res = validate(path)
    print(json.dumps({"path": str(path), **res}, indent=2))
    sys.exit(0 if res.get("ok") else 2)

if __name__ == "__main__":
    main()

