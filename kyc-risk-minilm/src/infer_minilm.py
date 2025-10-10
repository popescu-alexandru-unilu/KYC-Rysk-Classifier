import argparse, json, sys, torch, re, yaml
from model_minilm import MiniLMClassifier, NUM_LABELS

LABELS = ["low","medium","high"] if NUM_LABELS == 3 else ["low","high"]

CFG = yaml.safe_load(open("config/risk_rules.yaml"))
THR = CFG['thresholds']

NONE_LIKE = {"", "none", "unknown", "na", "n/a", "null", "0"}
CODE_SPLIT = re.compile(r"[;\|\s,]+")

def extract_sanctions_codes(text: str):
    """
    Return clean sanctions codes like ['us-bis-el','eu-blr'].
    Handles trailing punctuation and stops at the next [TAG].
    """
    m = re.search(r"\[SANCTIONS\]\s*list\s*=\s*([^\r\n]+)", text, re.I)
    if not m:
        return []
    raw = m.group(1)
    raw = raw.split("[", 1)[0]                 # stop at next tag e.g. "[MEDIA]"
    toks = CODE_SPLIT.split(raw)
    toks = [re.sub(r"[^a-z0-9\-]+", "", t.lower()) for t in toks]  # strip punctuation
    return [t for t in toks if t]

def _extract(text, pat, cast=float, default=None):
    m = re.search(pat, text, re.I)
    if not m: return default
    try: return cast(m.group(1))
    except: return default

def parse_signals(text: str):
    """Pull structured signals from your evidence text."""
    # sanctions codes
    sanc_codes = extract_sanctions_codes(text)
    sanc_hit   = any(t not in NONE_LIKE for t in sanc_codes)

    inflow_ratio = _extract(text, r"Inflow\s*ratio\s*=\s*([0-9]+(?:\.[0-9]+)?)", float)
    burst_trades = _extract(text, r"Burst\s*trades.*?=\s*([0-9]+)", int)
    media_cnt    = _extract(text, r"\[MEDIA\]\s*([0-9]+)\s+mentions?", int)

    country = None
    mc = re.search(r"\[COUNTRY\]\s*([A-Za-z][A-Za-z\-\s']{1,40})", text)
    if mc: country = mc.group(1).strip()

    return {
        "sanc_hit": sanc_hit,
        "sanc_codes": sanc_codes,
        "inflow_ratio": inflow_ratio,
        "burst_trades": burst_trades,
        "media_cnt": media_cnt,
        "country": country,
    }

def build_reasons(text: str, label: str, probs: dict, rule: str):
    s = parse_signals(text)
    why = []

    # Always explain the rule path first
    if rule == "sanctions_override" and s["sanc_hit"]:
        why.append(f"Sanctions/PEP code(s) present: {', '.join(s['sanc_codes'])}")
        return why

    # Evidence-based reasons
    if s["sanc_hit"]:
        why.append(f"Sanctions/PEP code(s) present: {', '.join(s['sanc_codes'])}")

    if s["media_cnt"] is not None:
        if s["media_cnt"] >= THR["media_high"]:
            why.append(f"Adverse media mentions: {s['media_cnt']} count exceeds threshold {THR['media_high']}, indicating significant reputational risk")
        elif s["media_cnt"] == 0:
            why.append("No adverse media mentions found")

    if s["inflow_ratio"] is not None:
        if s["inflow_ratio"] >= THR["inflow_ratio_high"]:
            why.append(f"Inflow ratio of {s['inflow_ratio']:.1f} exceeds high threshold {THR['inflow_ratio_high']:.1f}, suggesting unusual transaction volume that could signal money laundering risks")
        elif s["inflow_ratio"] < THR["inflow_ratio_low"]:
            why.append(f"Inflow ratio of {s['inflow_ratio']:.1f} is below low threshold {THR['inflow_ratio_low']:.1f}, indicating stable and expected activity")

    if s["burst_trades"] is not None:
        if s["burst_trades"] >= THR["burst_high"]:
            why.append(f"Trading burst: {s['burst_trades']} transactions in last 7 days exceed threshold {THR['burst_high']}, pointing to potential market manipulation or abnormal activity")
        elif s["burst_trades"] < THR["burst_low"]:
            why.append(f"No unusual trading burst observed ({s['burst_trades']} ≤ {THR['burst_low']}) – activity within normal range")

    if s["country"]:
        country_upper = s["country"].upper()
        if any(country_upper == c.upper() for c in CFG['fatf']['high_risk']):
            why.append(f"Country: {s['country']} – classified as FATF high-risk jurisdiction due to heightened regulatory scrutiny and potential for sanctions evasion")
        else:
            why.append(f"Country: {s['country']} – no additional jurisdictional risk flags")

    # If still empty, fall back to model confidence
    if not why:
        why.append(f"Model confidence: {label} ({probs[label]:.2f})")

    return why

NONE_LIKE = {"", "none", "unknown", "na", "n/a", "null", "0"}
CODE_SPLIT = re.compile(r"[;\|\s,]+")

def sanctions_hit(text: str) -> bool:
    codes = extract_sanctions_codes(text)
    return any(t not in NONE_LIKE for t in codes)

def override_high_payload():
    probs = {k: 0.0 for k in LABELS}
    probs["high"] = 1.0
    return {"probs": probs, "label": "high", "rule": "sanctions_override"}

def apply_additional_rules(res, text, CFG):
    if not CFG.get('fatf_bump', True): return res
    s = parse_signals(text)
    country_upper = s["country"].upper() if s["country"] else None
    if not country_upper: return res
    if any(country_upper == c.upper() for c in CFG['fatf']['high_risk']):
        if res['label'] == 'low':
            res['label'] = 'medium'
        elif res['label'] == 'medium':
            res['label'] = 'high'
    return res

def load_model(ckpt_path: str, device: str):
    m = MiniLMClassifier().to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device))
    m.eval()
    return m

@torch.no_grad()
def classify_batch(model, device, texts, max_len):
    # call featurize with a smaller max_len to save RAM
    emb = model.featurize(texts, device=device, max_len=max_len)   # [B, D]
    logits = model.head(emb).softmax(-1).cpu().tolist()            # [B, C]
    out = []
    for p_idx, p in enumerate(logits):
        lbl = LABELS[int(max(range(len(LABELS)), key=lambda i: p[i]))]
        res = {"probs": {LABELS[i]: p[i] for i in range(len(LABELS))}, "label": lbl}
        res = apply_additional_rules(res, texts[p_idx], CFG)
        out.append(res)
    return out

def _pct(p):
    # p can be dict label->prob or list
    if isinstance(p, dict):
        return {k: f"{v*100:.1f}%" for k,v in p.items()}
    return [f"{x*100:.1f}%" for x in p]

def format_chat(result: dict, text: str, topk: int = 3) -> str:
    label = result["label"]
    probs = result["probs"]
    rule  = result.get("rule","model_only")
    why   = result.get("why", [])[:topk]

    emoji = {"low":"🟢","medium":"🟡","high":"🔴"}.get(label, "▫️")
    rule_str = "Policy override: sanctions code detected" if rule=="sanctions_override" else "Model classification"
    probs_str = ", ".join(f"{k}={v}" for k,v in _pct(probs).items())
    lines = [
        f"{emoji} Risk: **{label.upper()}**",
        f"{rule_str}",
        f"Confidence: {probs_str}",
    ]
    if why:
        lines.append("Why:")
        lines += [f"• {w}" for w in why]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="minilm_cls_best.pt")
    ap.add_argument("--text", help="Inline text to classify")
    ap.add_argument("--file", help=".txt (single) or .jsonl (batch with 'text' field per line)")
    ap.add_argument("--batch_size", type=int, default=16)           # <<< adjust if still OOM
    ap.add_argument("--max_len", type=int, default=256)             # <<< shorter than 512 saves a lot of RAM
    ap.add_argument("--override", action="store_true", help="Force any sanctions hit to HIGH")
    ap.add_argument("--format", choices=["json","chat"], default="json",
                    help="Output style: json (default) or chat")
    ap.add_argument("--topk_reasons", type=int, default=3,
                    help="Max reasons to show in chat mode")
    args = ap.parse_args()

    if not (args.text or args.file):
        print("Provide --text or --file", file=sys.stderr); sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    # single text path
    if args.text:
        text = args.text
        if args.override and sanctions_hit(text):
            out = override_high_payload()
            out["why"] = build_reasons(text, out["label"], out["probs"], out["rule"])
            print(format_chat(out, text, args.topk_reasons) if args.format=="chat" else json.dumps(out)); return
        res = classify_batch(model, device, [text], args.max_len)[0]
        res["rule"] = "model_only"
        res["why"] = build_reasons(text, res["label"], res["probs"], res["rule"])
        print(format_chat(res, text, args.topk_reasons) if args.format=="chat" else json.dumps(res)); return

    # file path
    path = args.file
    if path.lower().endswith(".jsonl"):
        # stream: read, batch, print one JSON per line
        buf_texts, buf_ids = [], []
        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                t = obj["text"]
                if args.override and sanctions_hit(t):
                    c_id = obj.get("client_id", idx)
                    out = override_high_payload()
                    out["id"] = c_id
                    out["why"] = build_reasons(t, out["label"], out["probs"], out["rule"])
                    print(format_chat(out, t, args.topk_reasons) if args.format=="chat" else json.dumps(out))
                    continue
                buf_texts.append(obj["text"])
                buf_ids.append(obj.get("client_id", idx))
                if len(buf_texts) >= args.batch_size:
                    outs = classify_batch(model, device, buf_texts, args.max_len)
                    for i, (c_id, o) in enumerate(zip(buf_ids, outs)):
                        t = buf_texts[i]
                        o["rule"] = "model_only"
                        o["why"] = build_reasons(t, o["label"], o["probs"], "model_only")
                        payload = {"id": c_id, **o}
                        print(format_chat(payload, t, args.topk_reasons) if args.format=="chat" else json.dumps(payload))
                    buf_texts, buf_ids = [], []
        if buf_texts:
            outs = classify_batch(model, device, buf_texts, args.max_len)
            for i, (c_id, o) in enumerate(zip(buf_ids, outs)):
                t = buf_texts[i]
                o["rule"] = "model_only"
                o["why"] = build_reasons(t, o["label"], o["probs"], "model_only")
                payload = {"id": c_id, **o}
                print(format_chat(payload, t, args.topk_reasons) if args.format=="chat" else json.dumps(payload))
    else:
        texts = [open(path, encoding="utf-8").read()]
        text = texts[0]
        if args.override and sanctions_hit(text):
            out = override_high_payload()
            out["why"] = build_reasons(text, out["label"], out["probs"], out["rule"])
            print(format_chat(out, text, args.topk_reasons) if args.format=="chat" else json.dumps(out))
            return
        res = classify_batch(model, device, texts, args.max_len)[0]
        res["rule"] = "model_only"
        res["why"] = build_reasons(text, res["label"], res["probs"], res["rule"])
        print(format_chat(res, text, args.topk_reasons) if args.format=="chat" else json.dumps(res))

if __name__ == "__main__":
    main()
