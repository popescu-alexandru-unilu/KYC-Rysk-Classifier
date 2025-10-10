import argparse, json, re, torch
from collections import Counter
from .model_minilm import MiniLMClassifier, NUM_LABELS

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
    probs = model.head(emb).softmax(-1).cpu()                      # [B, C]
    preds = probs.argmax(dim=-1).tolist()
    return preds, probs.tolist()

def accuracy(y_true, y_pred):
    ok = sum(int(a==b) for a,b in zip(y_true, y_pred))
    return ok / max(1, len(y_true))

def confusion_matrix(y_true, y_pred, num_labels):
    cm = [[0]*num_labels for _ in range(num_labels)]
    for yt, yp in zip(y_true, y_pred): cm[yt][yp] += 1
    return cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/valid.jsonl")  # <<< CHANGE if your path differs
    ap.add_argument("--checkpoint", default="minilm_cls_best.pt")  # <<< CHANGE if needed
    ap.add_argument("--batch_size", type=int, default=16)  # <<< LOWER if you still hit OOM
    ap.add_argument("--max_len", type=int, default=256)    # <<< keep 256 for low RAM
    ap.add_argument("--limit", type=int, default=0, help="eval only first N rows (0=all)")
    ap.add_argument("--override", action="store_true", help="apply sanctions->HIGH rule (ternary only)")
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
    y_pred = []
    for i in range(0, len(texts), args.batch_size):
        chunk = texts[i:i+args.batch_size]
        preds, _ = predict_batch(model, device, chunk, max_len=args.max_len)
        y_pred.extend(preds)

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

if __name__ == "__main__":
    main()
