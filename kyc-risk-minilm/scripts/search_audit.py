#!/usr/bin/env python3
import argparse, json, sys, time
from datetime import datetime


def parse_time(s):
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        # try ISO8601-like
        try:
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            raise


def match(obj, args):
    if args.label and obj.get('label') != args.label:
        return False
    if args.rule and obj.get('rule') != args.rule:
        return False
    if args.band and obj.get('band_label') != args.band:
        return False
    if args.country and (obj.get('country') or '').upper() != args.country.upper():
        return False
    if args.owner:
        mt = obj.get('meta_tags') or {}
        if (mt.get('owner') or '').lower() != args.owner.lower():
            return False
    if args.lang:
        mt = obj.get('meta_tags') or {}
        if (mt.get('lang') or '').lower() != args.lang.lower():
            return False
    if args.since or args.until:
        ts = float(obj.get('ts') or 0)
        if args.since and ts < args.since:
            return False
        if args.until and ts > args.until:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description='Search audit.jsonl by fields')
    ap.add_argument('--log', default='logs/audit.jsonl')
    ap.add_argument('--label')
    ap.add_argument('--rule')
    ap.add_argument('--band')
    ap.add_argument('--country')
    ap.add_argument('--owner')
    ap.add_argument('--lang')
    ap.add_argument('--since', help='timestamp or ISO8601')
    ap.add_argument('--until', help='timestamp or ISO8601')
    ap.add_argument('--limit', type=int, default=100)
    args = ap.parse_args()
    args.since = parse_time(args.since)
    args.until = parse_time(args.until)

    count = 0
    with open(args.log, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if match(obj, args):
                print(json.dumps(obj))
                count += 1
                if args.limit and count >= args.limit:
                    break

    print(f"matched {count} lines", file=sys.stderr)


if __name__ == '__main__':
    main()

