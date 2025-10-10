# src/stress_infer.py
import argparse, json, re, time, torch
from model_minilm import MiniLMClassifier, NUM_LABELS

# ==== CONFIG (CHANGE if needed) ====
CKPT_PATH = "minilm_cls_best.pt"     # <<< CHANGE if your checkpoint has another name
LABELS = ["low","medium","high"] if NUM_LABELS==3 else ["low","high"]
NONE_LIKE = {"", "none", "unknown", "na", "n/a", "null", "0"}
CODE_SPLIT = re.compile(r"[;\|\s,]+")
# ===================================

def sanctions_hit(text: str) -> bool:
    m = re.search(r"\[SANCTIONS\]\s*list\s*=\s*([^\]\r\n]+)", text, re.I)
    if not m: return False
    raw = m.group(1).strip().lower()
    toks = [t for t in CODE_SPLIT.split(raw) if t]
    return any(t not in NONE_LIKE for t in toks)

@torch.no_grad()
def run_model(model, device, texts, max_len=256):
    # Directly use encoder + head to allow custom max_len
    emb = model.featurize(texts, device=device, max_len=max_len)
    probs = model.head(emb).softmax(-1).cpu().tolist()
    labels = [LABELS[int(max(range(len(LABELS)), key=lambda i: p[i]))] for p in probs]
    return probs, labels

def gen_cases():
    # Sanctions override should fire
    t1 = "[KYC] Name: CASPRO TECHNOLOGY LTD. [COUNTRY] Hong Kong [SANCTIONS] list=US-BIS-EL"
    t2 = "[KYC] Name: Grodno Azot. [SANCTIONS] list=EU-BLR; US-SDN"
    t3 = "[KYC] Name: Foo Bar. [SANCTIONS] list=unknown"          # should NOT fire
    t4 = "[KYC] Name: Baz. [SANCTIONS] list=None"                  # NOT fire (different casing)
    t5 = "[KYC] Name: Qux. [SANCTIONS] list=  us-sdn  | jp-meti "  # weird separators
    # Long text (truncation check)
    long_body = " [NOTE] " + ("benign " * 2000)
    t6 = f"[KYC] Name: Longy. [COUNTRY] DE. [SANCTIONS] list=none.{long_body}"
    # Unicode / non-latin
    t7 = "[KYC] 名前: 山田太郎. [COUNTRY] JP. [SANCTIONS] list=none."
    # Empty/minimal
    t8 = "[KYC] Name: . [SANCTIONS] list=none."
    # Adverse-looking but no sanctions
    t9 = "[KYC] Name: Random Co. [MEDIA] 3 mentions. [SANCTIONS] list=none."
    return [
        ("sanctions_simple", t1),
        ("sanctions_multi",  t2),
        ("unknown_value",    t3),
        ("none_value",       t4),
        ("weird_sep",        t5),
        ("very_long",        t6),
        ("unicode",          t7),
        ("minimal",          t8),
        ("media_only",       t9),
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT_PATH)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--override", action="store_true", help="Enable sanctions HIGH override")
    ap.add_argument("--bench", type=int, default=0, help="If >0, run throughput test with N repeats")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  | ckpt: {args.checkpoint} | max_len: {args.max_len} | override: {args.override}")

    model = MiniLMClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # ---------- Functional stress cases ----------
    cases = gen_cases()
    texts = [t for _, t in cases]

    # Apply override if requested (policy)
    results = []
    to_run = []
    for (name, t) in cases:
        if args.override and sanctions_hit(t):
            results.append({"case": name, "rule":"sanctions_override", "probs":{"low":0.0,"medium":0.0,"high":1.0} if NUM_LABELS==3 else {"low":0.0,"high":1.0}, "label": "high"})
        else:
            to_run.append((name, t))

    # Model-only for those not overridden
    if to_run:
        names = [n for n,_ in to_run]
        batch = [t for _,t in to_run]
        # chunk into batches
        for i in range(0, len(batch), args.batch_size):
            chunk = batch[i:i+args.batch_size]
            probs, labels = run_model(model, device, chunk, max_len=args.max_len)
            for n, p, lbl in zip(names[i:i+args.batch_size], probs, labels):
                results.append({"case": n, "rule": "model_only", "probs": {LABELS[i]: p[i] for i in range(len(LABELS))}, "label": lbl})

    # Print compact table
    print("\nCASE\tRULE\tLABEL\tPROBS")
    for r in results:
        pj = ";".join([f"{k}:{r['probs'][k]:.3f}" for k in LABELS])
        print(f"{r['case']}\t{r['rule']}\t{r['label']}\t{pj}")

    # ---------- Throughput benchmark (optional) ----------
    if args.bench > 0:
        base = "[KYC] Name: Benchmark. [COUNTRY] DE. [SANCTIONS] list=none. " + ("text " * 200)
        batch = [base] * args.batch_size
        # warmup
        for _ in range(3):
            run_model(model, device, batch, max_len=args.max_len)
        # timed repeats
        t0 = time.perf_counter()
        n_calls = args.bench
        for _ in range(n_calls):
            run_model(model, device, batch, max_len=args.max_len)
        dt = time.perf_counter() - t0
        total_items = n_calls * args.batch_size
        print(f"\nBenchmark: {total_items} samples in {dt:.2f}s | throughput ≈ {total_items/dt:.1f} samples/s")

if __name__ == "__main__":
    main()
