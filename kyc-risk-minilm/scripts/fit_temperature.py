#!/usr/bin/env python3
"""
Fit temperature scaling (Platt-like calibration) on a labeled validation set.

Usage:
  python scripts/fit_temperature.py \
    --checkpoint api/minilm_cls_best.pt \
    --file data/valid.jsonl \
    --batch_size 32 \
    --max_len 256 \
    --write_config config/risk_rules.yaml \
    --out config/calib.json

Keeps policy bands unchanged; only updates policy.temperature when --write_config provided.
"""
import argparse, json, yaml, time
from pathlib import Path
import torch


def load_model(ckpt_path: str, device: str):
    from api.src.model_minilm import MiniLMClassifier
    m = MiniLMClassifier().to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device))
    m.eval()
    return m


def load_valid_jsonl(path: str, limit: int = 0):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])  # expects 'text'
            labels.append(int(obj["label"]))  # expects 'label' as int index
    return texts, labels


@torch.no_grad()
def compute_logits(model, device, texts, max_len=256, batch_size=32):
    logits = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        emb = model.featurize(chunk, device=device, max_len=max_len)
        logit = model.head(emb)  # [B, C]
        logits.append(logit.cpu())
    return torch.cat(logits, dim=0) if logits else torch.empty(0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, steps=300, lr=0.01, verbose=False):
    # Optimize a scalar T > 0 to minimize cross-entropy of softmax(logits / T)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.Adam([T], lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    last_loss = None
    for s in range(steps):
        opt.zero_grad()
        scaled = logits / torch.clamp(T, min=1e-6)
        loss = ce(scaled, labels)
        loss.backward()
        opt.step()
        # optional: damp growth
        with torch.no_grad():
            T.clamp_(min=1e-3, max=100.0)
        if verbose and (s % 50 == 0 or s == steps-1):
            print(f"step {s:04d} loss={loss.item():.6f} T={T.item():.4f}")
        if last_loss is not None and abs(last_loss - loss.item()) < 1e-6:
            break
        last_loss = loss.item()
    return float(T.item())


def write_config_temperature(cfg_path: str, temperature: float):
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    policy = data.setdefault("policy", {})
    policy["temperature"] = float(temperature)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="api/minilm_cls_best.pt")
    ap.add_argument("--file", default="data/valid.jsonl")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write_config", help="Write policy.temperature into YAML config")
    ap.add_argument("--out", default="config/calib.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    texts, labels = load_valid_jsonl(args.file, limit=args.limit)
    if not texts:
        print("No data loaded from", args.file)
        return

    logits = compute_logits(model, device, texts, max_len=args.max_len, batch_size=args.batch_size)
    y = torch.tensor(labels, dtype=torch.long)
    T = fit_temperature(logits, y, steps=300, lr=0.01, verbose=args.verbose)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"temperature": T, "ts": time.time(), "n": len(labels)}, f)
    print(f"Fitted temperature: {T:.4f} (n={len(labels)}) -> {args.out}")

    if args.write_config:
        write_config_temperature(args.write_config, T)
        print(f"Updated policy.temperature in {args.write_config}")


if __name__ == "__main__":
    main()

