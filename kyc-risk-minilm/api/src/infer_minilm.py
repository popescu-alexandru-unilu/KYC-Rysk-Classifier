import argparse, json, sys, torch, re, os
import yaml
from .model_minilm import MiniLMClassifier, NUM_LABELS
from .config import load_rules, RiskRules

LABELS = ["low","medium","high"] if NUM_LABELS == 3 else ["low","high"]
ORDER  = {"low": 0, "medium": 1, "high": 2}
REV    = {v: k for k, v in ORDER.items()}

# Centralized config via Pydantic
RULES: RiskRules = load_rules()
THR = RULES.thresholds

# Unified raw config (YAML) for features not fully represented in Pydantic models
def _load_cfg():
    path = os.getenv("RISK_RULES", "config/risk_rules.yaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        try:
            return RULES.model_dump()
        except Exception:
            return {}

CFG = _load_cfg()

NONE_LIKE = {"", "none", "unknown", "na", "n/a", "null", "0"}
CODE_SPLIT = re.compile(r"[;\|\s,]+")
try:
    sanctions_config = (RULES.policy or {}).get("sanctions", {})
    valid_list = sanctions_config.get("valid_codes", [])
    _valid_codes = set(
        c.strip().lower()
        for c in valid_list
        if c is not None
    )
except Exception:
    _valid_codes = set()

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
    toks = [re.sub(r"[^a-z0-9\-]+", "", t.lower()) for t in toks]
    toks = [t for t in toks if t and (t not in NONE_LIKE)]
    if _valid_codes:
        toks = [t for t in toks if t in _valid_codes]
    return toks
 
def should_override(req_override_flag: bool) -> bool:
    """Decide if sanctions override policy is enabled server-side.
    If policy override_enabled is True, honor request flag; otherwise deny override.
    Defaults to enabled when unspecified.
    """
    try:
        pol = RULES.policy or {}
        enabled = bool(pol.get("override_enabled", True))
        if enabled:
            return bool(req_override_flag)
    except Exception:
        pass
    return bool(req_override_flag)

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

    # optional metadata tags
    def _tag(name, pat=r"([^\r\n\[]+)"):
        m = re.search(fr"\[{name}\]\s*{pat}", text, re.I)
        return m.group(1).strip() if m else None

    owner = _tag("OWNER")
    confidentiality = _tag("CONF")
    version = _tag("VER")
    language = _tag("LANG", pat=r"([A-Za-z\-]{2,10})")

    return {
        "sanc_hit": sanc_hit,
        "sanc_codes": sanc_codes,
        "inflow_ratio": inflow_ratio,
        "burst_trades": burst_trades,
        "media_cnt": media_cnt,
        "country": country,
        "owner": owner,
        "conf": confidentiality,
        "ver": version,
        "lang": language,
    }


def country_bucket(country: str | None) -> str | None:
    if not country:
        return None
    c = str(country).strip().upper()
    fatf = (CFG.get("fatf", {}) or {})
    mode = str(fatf.get("mode", "list")).lower()

    if mode == "list":
        high = [str(x).upper() for x in (fatf.get("high_risk", []) or [])]
        monitor = [str(x).upper() for x in (fatf.get("monitor", []) or fatf.get("monitored", []) or [])]
        if any(c == x for x in high):
            return "high_risk"
        if any(c == x for x in monitor):
            return "monitor"
        return None

    if mode == "score":
        scores = fatf.get("scores", {}) or {}
        try:
            cut = float(fatf.get("bump_threshold", 80))
        except Exception:
            cut = 80.0
        s = scores.get(c)
        try:
            s_val = float(s) if s is not None else None
        except Exception:
            s_val = None
        if s_val is not None and s_val >= cut:
            return "high_score"
        return None

    return None


def apply_fatf_bump(label: str, country: str | None) -> str:
    b = country_bucket(country)
    # Only bump for high-risk buckets; monitoring does not bump
    if b not in ("high_risk", "high_score"):
        return label
    if label == "low":
        return "medium"
    if label == "medium":
        return "high"
    return label

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
        if s["media_cnt"] >= THR.media_high:
            why.append(f"Adverse media mentions: {s['media_cnt']} count exceeds threshold {THR.media_high}, indicating significant reputational risk")
        elif s["media_cnt"] == 0:
            why.append("No adverse media mentions found")

    if s["inflow_ratio"] is not None:
        if s["inflow_ratio"] >= THR.inflow_ratio_high:
            why.append(f"Inflow ratio of {s['inflow_ratio']:.1f} exceeds high threshold {THR.inflow_ratio_high:.1f}, suggesting unusual transaction volume that could signal money laundering risks")
        elif s["inflow_ratio"] < THR.inflow_ratio_low:
            why.append(f"Inflow ratio of {s['inflow_ratio']:.1f} is below low threshold {THR.inflow_ratio_low:.1f}, indicating stable and expected activity")

    if s["burst_trades"] is not None:
        if s["burst_trades"] >= THR.burst_high:
            why.append(f"Trading burst: {s['burst_trades']} transactions in last 7 days exceed threshold {THR.burst_high}, pointing to potential market manipulation or abnormal activity")
        elif s["burst_trades"] < THR.burst_low:
            why.append(f"No unusual trading burst observed ({s['burst_trades']} ≤ {THR.burst_low}) – activity within normal range")

    if s["country"]:
        bucket = country_bucket(s["country"])  # list/score driven
        if bucket == "high_risk":
            why.append(f"Country: {s['country']} - FATF high-risk jurisdiction (policy bump applied)")
        elif bucket == "monitor":
            why.append(f"Country: {s['country']} - FATF on monitoring list")
        elif bucket == "high_score":
            cut = (CFG.get("fatf", {}) or {}).get("bump_threshold", 80)
            why.append(f"Country: {s['country']} - FATF score ≥ {cut} (policy bump applied)")
        else:
            why.append(f"Country: {s['country']} - no additional FATF flags")

    # If still empty, fall back to model confidence
    if not why:
        why.append(f"Model confidence: {label} ({probs[label]:.2f})")

    return why

def sanctions_hit(text: str) -> bool:
    codes = extract_sanctions_codes(text)
    return any(t not in NONE_LIKE for t in codes)

def override_high_payload(codes: list[str] | None = None):
    probs = {k: 0.0 for k in LABELS}
    probs["high"] = 1.0
    codes = codes or []
    return {
        "probs": probs,
        "label": "high",
        "rule": "sanctions_override",
        "override_reason": "sanctions_hit",
        "sanctions_codes": codes,
        "sanc_codes": codes,
    }

def apply_additional_rules(res: dict, text: str, rules: RiskRules | None = None):
    # FATF bump via unified config (list or score); high lifts by 1 step
    s = parse_signals(text)
    res['label'] = apply_fatf_bump(res.get('label', 'low'), s.get('country'))
    return res

def bump_label_by_rules(parsed: dict, base_label: str) -> str:
    """Lift the label when structured signals cross thresholds (soft policy)."""
    lvl = ORDER.get(base_label, 0)

    # media bump
    mc = parsed.get("media_cnt")
    if mc is not None and mc >= getattr(THR, "media_high", 999999):
        lvl = max(lvl, ORDER["high"])  # strong media => HIGH

    # inflow bump
    inflow = parsed.get("inflow_ratio")
    if inflow is not None:
        hi = getattr(THR, "inflow_ratio_high", None)
        lo = getattr(THR, "inflow_ratio_low", None)
        if hi is not None and inflow >= hi:
            lvl = max(lvl, ORDER["high"])  # strong inflow => HIGH
        elif lo is not None and inflow >= lo:
            lvl = max(lvl, ORDER["medium"])  # moderate inflow => MEDIUM

    # burst trades bump
    burst = parsed.get("burst_trades")
    if burst is not None:
        bh = getattr(THR, "burst_high", None)
        bl = getattr(THR, "burst_low", None)
        if bh is not None and burst >= bh:
            lvl = max(lvl, ORDER["high"])  # strong burst => HIGH
        elif bl is not None and burst >= bl:
            lvl = max(lvl, ORDER["medium"])  # moderate burst => MEDIUM

    return REV.get(lvl, base_label)

def nudge_probs_to_label(probs: dict, target: str, alpha: float = 0.15) -> dict:
    """Cosmetically pull some probability mass toward the target label.
    Keeps a valid distribution; does not exceed 1.0 for target and 0.0 for others.
    """
    p = probs.copy()
    if target not in p:  # safety
        return p
    add = min(alpha, 1.0 - p[target])
    others = [k for k in p.keys() if k != target]
    total_other = sum(p[k] for k in others) or 1e-9
    for k in others:
        take = add * (p[k] / total_other)
        p[k] = max(0.0, p[k] - take)
    p[target] = min(1.0, p[target] + add)
    return p

def load_model(ckpt_path: str, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    m = MiniLMClassifier()
    if os.path.isdir(ckpt_path):
        m.model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
        m.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    else:
        m.model.load_state_dict(torch.load(ckpt_path, map_location=device))
    m.to(device)
    m.eval()
    return m

@torch.no_grad()
def classify_batch(model, device, texts, max_len):
    # optional pre-truncation for very long texts
    processed = []
    for t in texts:
        processed.append(_preprocess_long_text(t) if len(t) > 4000 else t)

    model.to(device)
    logits = model(processed, device=device)
    # temperature scaling if provided
    T = 1.0
    try:
        T = float((RULES.policy or {}).get("temperature", 1.0))  # type: ignore[attr-defined]
    except Exception:
        T = 1.0
    probs = (logits / max(1e-6, T)).softmax(-1).cpu().tolist()           # [B, C]
    out = []
    for p_idx, p in enumerate(probs):
        lbl = LABELS[int(max(range(len(LABELS)), key=lambda i: p[i]))]
        res = {"probs": {LABELS[i]: p[i] for i in range(len(LABELS))}, "label": lbl}
        # soft policy: bump label based on parsed signals
        parsed = parse_signals(texts[p_idx])
        bumped = bump_label_by_rules(parsed, res["label"])  # soft lift
        if bumped != res["label"]:
            res["label"] = bumped
            res["probs"] = nudge_probs_to_label(res["probs"], res["label"])  # cosmetic shift
        # country-based bump (FATF high-risk) if enabled
        res = apply_additional_rules(res, texts[p_idx], RULES)
        # unknown wording nudge: if clearly unknown and no risk signals, lean LOW
        s = parse_signals(texts[p_idx])
        if res["label"] == "medium":
            if (not s.get("sanc_hit")) and (s.get("media_cnt") in (None, 0)):
                inflow = s.get("inflow_ratio") or 0.0
                burst = s.get("burst_trades") or 0
                if inflow < getattr(THR, "inflow_ratio_low", 1.8) and burst < getattr(THR, "burst_low", 10):
                    if re.search(r"\bunknown\b", texts[p_idx], re.I):
                        res["label"] = "low"
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
def _preprocess_long_text(text: str) -> str:
    """
    Preserve critical tags like [SANCTIONS], [COUNTRY], and short name, then
    append a truncated remainder to stabilize very long inputs.
    """
    # simple heuristics: collect lines with key tags
    keep_lines = []
    for line in text.splitlines():
        L = line.strip()
        if not L:
            continue
        if any(tag in L.upper() for tag in ("[SANCTIONS]", "[COUNTRY]", "[KYC]")):
            keep_lines.append(L)
    head = "\n".join(keep_lines[:6])
    # append truncated tail
    rest = text if not keep_lines else "\n".join(l for l in text.splitlines() if l.strip() not in keep_lines)
    # token-ish truncation by words
    tail = " ".join(rest.split()[:512])
    out = (head + "\n" + tail).strip()
    # clamp overall size
    if len(out) > 8000:
        out = out[:8000]
    return out or text
