# >>> CHANGE the path to your downloaded file
CSV_IN  = "data/raw/targets.simple.csv"
CSV_OUT = "data/raw/opensanctions_targets_min.csv"

import pandas as pd

df = pd.read_csv(CSV_IN, dtype=str, keep_default_na=False)  # UTF-8 by default

# Auto-detect dataset column
cands = [c for c in df.columns if any(k in c.lower() for k in ["list","dataset","program"])]
dataset_col = None
for c in cands:
    # pick the first with >=1% non-empty and lots of A-Z+hyphen tokens
    s = df[c].astype(str).fillna("")
    if (s!="").mean() >= 0.01:
        sample = " ".join(s.head(500).to_list()).lower()
        if any(tok in sample for tok in ["us-", "eu-", "uk-", "ofac", "sdn"]):
            dataset_col = c; break
# fallback if nothing matched
if dataset_col is None and "lists" in df.columns: dataset_col = "lists"
if dataset_col is None and "dataset" in df.columns: dataset_col = "dataset"
if dataset_col is None: dataset_col = cands[0] if cands else None
print(f"[builder] using dataset_col = {dataset_col!r}")

# >>> CHANGE these mappings to match your actual headers
COLS = {
    "name":       "name",                # e.g., "Grodno Azot", "CASPRO TECHNOLOGY LTD"
    "aka":        "aliases",             # sometimes "aka" or "aliases"
    "birthDate":  "birth_date",          # map to actual column name
    "countries":  "countries",           # often "countries" (alpha-2 codes or names)
    "dataset":    dataset_col,           # auto-detected column
    "sourceUrl":  "first_seen",          # or last_seen/last_change, assuming first_seen as sourceUrl proxy
}

# keep only columns that exist; fill missing ones
have = {k:v for k,v in COLS.items() if v in df.columns}
slim = df[list(have.values())].copy()
slim.columns = list(have.keys())
for k in ["aka","birthDate","countries","dataset","sourceUrl"]:
    if k not in slim.columns:
        slim[k] = ""

# normalize: keep first dataset tag; collapse whitespace
slim["dataset"] = slim["dataset"].apply(lambda s: (s.split(";")[0].strip() if s else ""))

# sanity checks
print("rows:", len(slim))
print("empty names:", (slim["name"].str.len()==0).sum())
print("example row:\n", slim.head(1).to_dict(orient="records")[0])

slim.to_csv(CSV_OUT, index=False)
print("Wrote:", CSV_OUT)
# === ADD BELOW (after writing opensanctions_targets_min.csv) ===
import json, random

# >>> CHANGE: cap rows so training fits your machine (start with 40_000)
MAX_ROWS_FOR_JSONL = 40000

# >>> CHANGE: choose label scheme
#   "binary"  -> 0=low (no list), 1=high (on any list)
#   "ternary" -> 0=low, 1=medium, 2=high  (we'll synthesize some "medium")
LABEL_SCHEME = "ternary"  # or "binary"

# >>> CHANGE: output paths if you like
TRAIN_OUT = "data/train.jsonl"
VALID_OUT = "data/valid.jsonl"
TRAIN_SPLIT = 0.8
SEED = 7

# Use the DataFrame you just wrote
slim = pd.read_csv("data/raw/opensanctions_targets_min.csv", dtype=str, keep_default_na=False)
if MAX_ROWS_FOR_JSONL:
    slim = slim.head(MAX_ROWS_FOR_JSONL)

def make_text(r):
    parts = []
    if r.get("name"):      parts.append(f"[KYC] Name: {r['name']}.")
    if r.get("aka"):       parts.append(f"[AKA] {str(r['aka']).split(';')[0]}")
    if r.get("birthDate"): parts.append(f"[DOB] {r['birthDate']}")
    if r.get("countries"): parts.append(f"[COUNTRY] {str(r['countries']).split(';')[0]}")
    parts.append(f"[SANCTIONS] list={r.get('dataset','') or 'none'}")
    return " ".join(parts)

def label_row(r):
    code = (r.get("dataset") or "").strip().lower()
    if code not in {"", "none", "unknown", "na", "n/a", "null", "0"}:
        return 2  # HIGH
    # else: synthesize some mediums
    import random
    return 1 if random.random() < 0.2 else 0

rows = slim.to_dict(orient="records")
random.Random(SEED).shuffle(rows)
cut = int(TRAIN_SPLIT * len(rows))

with open(TRAIN_OUT, "w", encoding="utf-8") as ftr, open(VALID_OUT, "w", encoding="utf-8") as fva:
    for i, r in enumerate(rows):
        obj = {"client_id": f"S_{i}", "text": make_text(r), "label": label_row(r)}
        (ftr if i < cut else fva).write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"[jsonl] wrote {TRAIN_OUT} and {VALID_OUT} with LABEL_SCHEME={LABEL_SCHEME}")
# === END ADD ===
