import argparse, json, re, torch, time
from collections import Counter
from model_minilm import MiniLMClassifier, NUM_LABELS

# --- labels ---
LABELS = ["low","medium","high"] if NUM_LABELS==3 else ["low","high"]
IDX = {name:i for i,name in enumerate(LABELS)}

# --- sanctions override helpers ---
NONE_LIKE = {"", "none", "unknown", "na", "n/a", "null", "0"}
SPLIT = re.compile(r"[;\|\s,]+")
def sanctions_hit(t: str) -> bool:
    m = re.search(r"\[SANCTIONS\]\s*list\s*=\s*([^\]\r\n]+)", t, re.I)
    if not m: return False
    toks = [x for x in SPLIT.split(m.group(1).strip().lower()) if x]
    return any(x not in NONE_LIKE for x in toks)

@torch.no_grad()
def predict_batch(model, device, texts, max_len=256):
    emb = model.featurize(texts, device=device, max_len=max_len)   # [B, D]
    logits = model.head(emb)
    probs = logits.softmax(-1).cpu()                               # [B, C]
    preds = probs.argmax(dim=-1).tolist()
    return preds, probs.tolist()

def accuracy(y_true, y_pred):
    ok = sum(int(a==b) for a,b in zip(y_true, y_pred))
    return ok / max(1, len(y_true))

def confusion_matrix(y_true, y_pred, num_labels):
    cm = [[0]*num_labels for _ in range(num_labels)]
    for yt, yp in zip(y_true, y_pred): cm[yt][yp] += 1
    return cm


def prf_per_class(cm):
    L = len(cm)
    prec = []
    rec = []
    f1 = []
    for i in range(L):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(L) if r != i)
        fn = sum(cm[i][c] for c in range(L) if c != i)
        p = tp / max(1, (tp + fp))
        r = tp / max(1, (tp + fn))
        f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        prec.append(p); rec.append(r); f1.append(f)
    macro_f1 = sum(f1)/L
    return prec, rec, f1, macro_f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/valid.jsonl")  # <<< CHANGE if your path differs
    ap.add_argument("--checkpoint", default="minilm_cls_best.pt")  # <<< CHANGE if needed
    ap.add_argument("--batch_size", type=int, default=16)  # <<< LOWER if you still hit OOM
    ap.add_argument("--max_len", type=int, default=256)    # <<< keep 256 for low RAM
    ap.add_argument("--limit", type=int, default=0, help="eval only first N rows (0=all)")
    ap.add_argument("--override", action="store_true", help="apply sanctions->HIGH rule (ternary only)")
    ap.add_argument("--json_out", help="Write metrics JSON to this path")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniLMClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # --- load small lists (ok for tens of thousands). Use --limit if huge.
    texts, y_true = [], []
    with open(args.file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit: break
            if not line.strip(): continue
            obj = json.loads(line)
            texts.append(obj["text"])
            y_true.append(int(obj["label"]))

    # --- model-only predictions in batches
    y_pred, all_probs = [], []
    for i in range(0, len(texts), args.batch_size):
        chunk = texts[i:i+args.batch_size]
        preds, probs = predict_batch(model, device, chunk, max_len=args.max_len)
        y_pred.extend(preds); all_probs.extend(probs)

    # --- optional policy override (only meaningful if NUM_LABELS==3)
    if args.override and NUM_LABELS == 3:
        y_pred_rule = []
        for t, p in zip(texts, y_pred):
            if sanctions_hit(t):
                y_pred_rule.append(IDX["high"])
            else:
                y_pred_rule.append(p)
    else:
        y_pred_rule = y_pred

    # --- metrics
    acc_model = accuracy(y_true, y_pred)
    acc_rule  = accuracy(y_true, y_pred_rule)
    print(f"Valid acc (model-only):   {acc_model:.3f}")
    print(f"Valid acc (with override):{acc_rule:.3f}" if (args.override and NUM_LABELS==3) else "")

    cm = confusion_matrix(y_true, y_pred_rule, len(LABELS))
    print("Confusion (rows=true, cols=pred):")
    for i,row in enumerate(cm):
        print(f"{LABELS[i]:>7} {row}")

    # per-class metrics
    prec, rec, f1, macro_f1 = prf_per_class(cm)
    print("Per-class metrics:")
    for i, name in enumerate(LABELS):
        print(f"  {name:>7} P={prec[i]:.3f} R={rec[i]:.3f} F1={f1[i]:.3f}")
    print(f"Macro-F1: {macro_f1:.3f}")

    # High recall (with override) and Low precision (auto-clear proxy)
    rec_hi = None
    prec_lo = None
    if NUM_LABELS == 3:
        hi_idx = IDX['high']
        # recall on high
        tp_hi = sum(1 for yt, yp in zip(y_true, y_pred_rule) if yt == hi_idx and yp == hi_idx)
        total_hi = sum(1 for yt in y_true if yt == hi_idx)
        rec_hi = tp_hi / max(1, total_hi)
        print(f"High recall (with override): {rec_hi:.3f}")

        lo_idx = IDX['low']
        # precision for low predictions (model-only proxy)
        tp_lo = sum(1 for yt, yp in zip(y_true, y_pred) if yp == lo_idx and yt == lo_idx)
        pred_lo = sum(1 for yp in y_pred if yp == lo_idx)
        prec_lo = tp_lo / max(1, pred_lo)
        print(f"Low precision (model-only): {prec_lo:.3f}")

    # optional JSON output
    if args.json_out:
        payload = {
            "ts": time.time(),
            "checkpoint": args.checkpoint,
            "file": args.file,
            "limit": args.limit,
            "override": bool(args.override),
            "labels": LABELS,
            "accuracy_model": acc_model,
            "accuracy_with_override": acc_rule,
            "macro_f1": macro_f1,
            "per_class": {LABELS[i]: {"precision": prec[i], "recall": rec[i], "f1": f1[i]} for i in range(len(LABELS))},
            "confusion": cm,
            "recall_high_with_override": rec_hi,
            "precision_low_model_only": prec_lo,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"Wrote metrics JSON -> {args.json_out}")

if __name__ == "__main__":
    main()
