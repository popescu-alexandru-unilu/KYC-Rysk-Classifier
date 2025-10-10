#!/usr/bin/env python3
import os, json, time, argparse
import requests


def stream_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/valid.jsonl")
    ap.add_argument("--out", dest="out", default="logs/nightly_results.jsonl")
    ap.add_argument("--api", dest="api", default=os.getenv("API_BASE", "http://localhost:8000"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--override", action="store_true")
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    buf = []
    total = 0
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as fout:
        for obj in stream_jsonl(args.inp):
            buf.append({"id": obj.get("client_id"), "text": obj["text"]})
            if len(buf) >= args.batch:
                r = requests.post(f"{args.api}/classify_batch", json={
                    "items": buf,
                    "override": args.override,
                    "max_len": args.max_len,
                }, timeout=60)
                r.raise_for_status()
                for item in r.json():
                    fout.write(json.dumps(item) + "\n")
                total += len(buf)
                buf = []
        if buf:
            r = requests.post(f"{args.api}/classify_batch", json={
                "items": buf,
                "override": args.override,
                "max_len": args.max_len,
            }, timeout=60)
            r.raise_for_status()
            for item in r.json():
                fout.write(json.dumps(item) + "\n")
            total += len(buf)

    dt = time.time() - t0
    print(f"Processed {total} items in {dt:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()

