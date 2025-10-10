import argparse
import json
import os
import sys
import time
from collections import Counter

import requests


ORDER = {"low": 0, "medium": 1, "high": 2}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_label(obj):
    # Accept a variety of keys: label (str or int), expected, y, target
    for k in ("label", "expected", "y", "target"):
        if k in obj:
            return obj[k]
    raise KeyError("No label key found in object (expected one of: label, expected, y, target)")


def normalize_labels(truth_values):
    # Convert string labels to canonical names; keep ints as-is initially
    names = set()
    for v in truth_values:
        if isinstance(v, str):
            names.add(v.strip().lower())
    if names:
        # Order by ORDER map if possible
        ordered = sorted(names, key=lambda x: ORDER.get(x, 999))
        name_to_idx = {name: i for i, name in enumerate(ordered)}
    else:
        # No names present; assume ints starting at 0
        uniq = sorted(set(int(v) for v in truth_values))
        name_to_idx = {str(i): i for i in uniq}
        ordered = [str(i) for i in uniq]
    idx_to_name = {i: n for n, i in name_to_idx.items()}
    return name_to_idx, idx_to_name


def confusion_matrix(y_true, y_pred, n_labels):
    cm = [[0] * n_labels for _ in range(n_labels)]
    for yt, yp in zip(y_true, y_pred):
        cm[yt][yp] += 1
    return cm


def classification_report(cm):
    n = len(cm)
    per_class = []
    for i in range(n):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n) if r != i)
        fn = sum(cm[i][c] for c in range(n) if c != i)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = tp + fn
        per_class.append((prec, rec, f1, support))
    return per_class


def main():
    ap = argparse.ArgumentParser(description="Evaluate /classify API on a JSONL file of text+labels")
    ap.add_argument("--file", default="data/manual_test.jsonl", help="Path to JSONL with fields: text, label (or expected/y/target)")
    ap.add_argument("--api", default="http://localhost:8000", help="Base URL of the API (no trailing slash)")
    ap.add_argument("--batch_size", type=int, default=16, help="How many to send per progress step (requests are sequential)")
    ap.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout per request (seconds)")
    ap.add_argument("--override", action="store_true", default=True, help="Set override=true when calling /classify")
    ap.add_argument("--no-override", dest="override", action="store_false", help="Set override=false when calling /classify")
    ap.add_argument("--min_acc", type=float, default=0.0, help="If >0, fail (exit 1) when accuracy falls below this threshold")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"[skip] File not found: {args.file}")
        sys.exit(0)

    rows = load_jsonl(args.file)
    texts = []
    truth_raw = []
    for r in rows:
        texts.append(r["text"])
        truth_raw.append(pick_label(r))

    # Normalize labels and build mapping
    name_to_idx, idx_to_name = normalize_labels(truth_raw)
    # Convert truth to indices
    y_true = []
    for v in truth_raw:
        if isinstance(v, int):
            y_true.append(v)
        elif isinstance(v, str):
            y_true.append(name_to_idx[v.strip().lower()])
        else:
            raise ValueError(f"Unsupported label type: {type(v)}")

    # Query API sequentially
    api = args.api.rstrip("/")
    y_pred = []
    t0 = time.time()
    for i, t in enumerate(texts, 1):
        payload = {"text": t, "override": args.override, "format": "json"}
        try:
            r = requests.post(f"{api}/classify", json=payload, timeout=args.timeout)
            r.raise_for_status()
            obj = r.json()
            label_name = str(obj.get("label", "")).strip().lower()
            if label_name in name_to_idx:
                y_pred.append(name_to_idx[label_name])
            elif label_name in ORDER:
                # unseen but canonical label; extend map
                nid = max(name_to_idx.values(), default=-1) + 1
                name_to_idx[label_name] = nid
                idx_to_name[nid] = label_name
                y_pred.append(nid)
            else:
                raise ValueError(f"API returned unknown label: {label_name}")
        except Exception as e:
            print(f"[error] row={i}: {e}")
            y_pred.append(-1)  # mark as invalid
        if i % max(1, args.batch_size) == 0:
            elapsed = time.time() - t0
            print(f".. processed {i}/{len(texts)} in {elapsed:.1f}s")

    # Filter out invalid (-1) predictions
    paired = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp >= 0]
    if not paired:
        print("No valid predictions to score.")
        sys.exit(1)
    y_true, y_pred = zip(*paired)

    # Metrics
    n_labels = len(idx_to_name)
    total = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    acc = correct / total if total else 0.0
    print(f"\nAccuracy: {acc:.3f}  ({correct}/{total})\n")

    cm = confusion_matrix(y_true, y_pred, n_labels)
    per_cls = classification_report(cm)

    # Pretty print per-class metrics
    print("Per-class precision/recall/F1 (support):")
    for i in range(n_labels):
        name = idx_to_name.get(i, str(i)).upper()
        p, r, f1, sup = per_cls[i]
        print(f"  {name:>6}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (n={sup})")

    # Confusion matrix
    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "       " + " ".join(f"{idx_to_name.get(i, str(i)).upper():>6}" for i in range(n_labels))
    print(header)
    for i in range(n_labels):
        row_name = idx_to_name.get(i, str(i)).upper()
        print(f"{row_name:>6}  " + " ".join(f"{cm[i][j]:>6}" for j in range(n_labels)))

    if args.min_acc and acc < args.min_acc:
        print(f"\nFAIL: accuracy {acc:.3f} < min_acc {args.min_acc:.3f}")
        sys.exit(2)


if __name__ == "__main__":
    main()

