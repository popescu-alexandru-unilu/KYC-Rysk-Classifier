import hmac, hashlib, json, os

SECRET = os.getenv("AUDIT_HMAC", "change_me")
ok = True
with open("logs/audit.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        mac = obj.pop("hmac", None)
        calc = hmac.new(SECRET.encode(), json.dumps(obj, sort_keys=True).encode(), "sha256").hexdigest()
        if mac != calc:
            print(f"Line {i}: HMAC MISMATCH")
            ok = False
print("OK" if ok else "FAIL")

