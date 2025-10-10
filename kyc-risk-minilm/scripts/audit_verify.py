#!/usr/bin/env python3
import os, json, hmac, hashlib, argparse


def verify_line(obj, secret: str) -> bool:
    body = {
        "ts": obj.get("ts"),
        "text_sha": obj.get("text_sha"),
        "label": obj.get("label"),
        "rule": obj.get("rule"),
        "probs": obj.get("probs"),
        "why": obj.get("why"),
    }
    mac = hmac.new(secret.encode(), json.dumps(body, sort_keys=True).encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, obj.get("hmac", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/audit.jsonl")
    ap.add_argument("--secret", default=os.getenv("AUDIT_HMAC", "change_me_in_prod"))
    args = ap.parse_args()

    total = ok = 0
    with open(args.log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
                if verify_line(obj, args.secret):
                    ok += 1
            except Exception:
                pass
    pct = (ok / total * 100.0) if total else 100.0
    print(f"Verified {ok}/{total} lines ({pct:.1f}%).")
    if total and ok < total:
        exit(1)


if __name__ == "__main__":
    main()

