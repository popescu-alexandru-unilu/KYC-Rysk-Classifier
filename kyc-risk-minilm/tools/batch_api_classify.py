import os, sys, json, time, argparse, csv
import urllib.request


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def post_json(url, payload, timeout=60, api_key=None):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    if api_key:
        req.add_header('X-API-Key', api_key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode('utf-8')
        return json.loads(body), dict(resp.getheaders())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='Input JSONL with at least a text field per line')
    ap.add_argument('--out', dest='out', required=True, help='Output results JSONL')
    ap.add_argument('--csv', dest='csv_path', required=True, help='Output CSV path')
    ap.add_argument('--api', dest='api', default=os.environ.get('API', 'http://localhost:8000'))
    ap.add_argument('--override', dest='override', default='true')
    ap.add_argument('--chunk', dest='chunk', type=int, default=200)
    ap.add_argument('--timeout', dest='timeout', type=int, default=60)
    args = ap.parse_args()

    api_base = args.api.rstrip('/')
    url = f"{api_base}/classify_batch"
    api_key = os.environ.get('API_KEY')

    # Read JSONL
    records = []
    with open(args.inp, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get('text')
            if not isinstance(t, str) or not t.strip():
                continue
            rec = {'text': t}
            if 'id' in obj:
                rec['id'] = obj['id']
            records.append(rec)

    if not records:
        print('No valid input lines with a text field.', file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path) or '.', exist_ok=True)
    out_f = open(args.out, 'w', encoding='utf-8')
    csv_f = open(args.csv_path, 'w', encoding='utf-8', newline='')
    writer = csv.writer(csv_f)
    writer.writerow(['idx','id','label','rule','low','medium','high','why'])

    total = 0
    start = time.time()
    for batch in chunks(records, max(1, args.chunk)):
        payload = {'items': batch, 'override': str(args.override).lower() in ('1','true','yes','y')}
        # simple retry with backoff
        for attempt in range(5):
            try:
                data, headers = post_json(url, payload, timeout=args.timeout, api_key=api_key)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))
        if not isinstance(data, list):
            raise RuntimeError('Bad response')
        for idx, item in enumerate(data):
            out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
            low = item.get('probs', {}).get('low', 0)
            med = item.get('probs', {}).get('medium', 0)
            high = item.get('probs', {}).get('high', 0)
            why = '; '.join(item.get('why') or [])
            writer.writerow([total+idx, item.get('id',''), item.get('label'), item.get('rule'), low, med, high, why])
        total += len(data)

    out_f.close(); csv_f.close()
    print(f"Processed {total} items in {time.time()-start:.1f}s. Wrote {args.out} and {args.csv_path}")


if __name__ == '__main__':
    main()

