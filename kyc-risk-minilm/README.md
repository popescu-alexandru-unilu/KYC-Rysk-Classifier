kyc-risk-minilm/
├─ data/
│  ├─ raw/
│  │  └─ opensanctions_targets.csv        # CHANGE: put downloaded CSV here
│  ├─ train.jsonl                         # auto-generated
│  └─ valid.jsonl                         # auto-generated
├─ src/
│  ├─ build_dataset.py                    # builds train/valid JSONL (weak labels)
│  ├─ model_minilm.py                     # MiniLM-L6 encoder + Linear(384→3)
│  ├─ train_minilm.py                     # fine-tune (head-first, then optional unfreeze)
│  ├─ infer_minilm.py                     # single-text inference → {probs,label}
│  ├─ app.py                              # OPTIONAL FastAPI /classify endpoint
│  └─ config.py                           # thresholds/paths in one place
├─ scripts/
│  ├─ fetch_opensanctions.sh              # OPTIONAL: curl the OpenSanctions CSV
│  └─ run_pipeline.sh                     # build → train → infer (example)
├─ requirements.txt
├─ Dockerfile                             # OPTIONAL: CPU container for Hetzner
├─ docker-compose.yml                     # OPTIONAL: FastAPI + restart policy
└─ README.md
Scope (what this app is)

Purpose: Internal tool for compliance analysts to classify client risk (low/medium/high) from KYC evidence (docs + public signals).

Model: MiniLM-L6 encoder + small softmax head (fine-tuned).

Output: {label, probabilities, top factors} for analyst triage (not an automated decision).

In-scope (v1) – what we will do

Risk Tier Classification:
Input: short evidence text (KYC summary + statement facts + sanctions/PEP result + adverse-media snippet).
Output: low/medium/high + probabilities + brief factors.

Sanctions/PEP signal ingestion:
Use OpenSanctions/OFAC/OFSI to add match/no-match and basic metadata.

Adverse media signal (light):
Count 0/1/≥2 credible mentions; store title/year/source for top 1–2.

Auditability:
Log: model version, thresholds, evidence pointers, and risk factors.

WHERE CHANGES NEED TO BE MADE

src/config.py: enable/disable adverse media (ADD_MEDIA), set thresholds.

src/build_dataset.py: compose evidence text; adjust weak-label rules.

src/model_minilm.py: keep head dim = 384 for MiniLM-L6.

src/train_minilm.py: tweak class weights, batch size, epochs.

Out-of-scope (v1) – what we won’t do

No automated approvals/declines. Human in the loop only.

No investment advice or trading strategies. This is a compliance tool.

No behavior anomaly detection (that’s a phase-2 use case).

No model pretraining from scratch. Only fine-tuning MiniLM.

Allowed to ask the chatbot

“Classify this client” (provide evidence text).

“Show top factors behind the label.”

“Re-score after adding/removing this evidence.”

“What changed vs. previous score?” (delta explanation)

“Download audit JSON for this case.”

Batch and Uploads

- Ad‑hoc (UI)
  - .txt → fills the textarea for a single run; you can edit and click Assess.
  - .jsonl → runs a batch via `/classify_batch`, shows the first item’s result, and downloads a CSV of all decisions.
  - UI limits: keep JSONL ≤ ~10k lines for responsiveness; a 60s timeout is used.

- File format (JSONL): one JSON object per line with at least a `text` field. Optional `id` is recommended.
  - Example lines:
    - {"id":"C001","text":"[KYC] Name: Jane Roe. [COUNTRY] DE. [SANCTIONS] list=none. [MEDIA] 0 mentions"}
    - {"id":"C002","text":"[KYC] Name: Atlas Trading. [COUNTRY] AE. [SANCTIONS] list=none. [MEDIA] 1 mention. Inflow ratio = 2.2"}
    - {"id":"C003","text":"[KYC] Name: ACME Ltd. [COUNTRY] HK. [SANCTIONS] list=US-BIS-EL"}

- Batch API payload shape: the API expects an object with an `items` array.
  - POST /classify_batch with body:
    - {"items":[{"id":"C001","text":"..."}, {"id":"C002","text":"..."}], "override": true}

Nightly screening (reference)

- Recommended: run a scheduled job against `/classify_batch` in chunks and save outputs.
  - Input: JSONL on disk or object storage.
  - Chunk size: 1–5k; respect server-side `BATCH_MAX` (default 256), e.g., use 200.
  - Output: results.jsonl + CSV; the API also writes tamper‑evident audit lines with HMAC.

Tooling

- Headless batch client: `tools/batch_api_classify.py` (added) calls the API in chunks and writes JSONL+CSV.
  - Usage:
    - python tools/batch_api_classify.py --in clients.jsonl --out results.jsonl --csv results.csv \
      --api http://localhost:8000 --override true --chunk 200 --timeout 60
  - Auth: set `API_KEY` to send `X-API-Key`.

Security & redaction

- Server redacts PII in audit logs (configurable under `redaction:` in `config/risk_rules.yaml`). The client does not log content.

Redact card PAN to last 4; remove CVV/track data.

Redact plaintext passwords/API keys.

For IBAN/account numbers, keep last 4 only.

Prefer text-native PDFs; for scans, OCR then strip images after text extraction.

WHERE CHANGES NEED TO BE MADE

src/app.py (upload endpoint): allowed extensions = {.pdf,.txt,.csv}; max file size (e.g., 10MB); per-request limit (e.g., 5 files).

src/config.py: ALLOWED_EXTS, MAX_FILE_MB, MAX_FILES.

NOT allowed to ask or upload (hard refusals)

Evasion or wrongdoing: how to bypass AML/KYC, launder money, forge docs, or hide funds.

Hacking/malware: exploit code, viruses, password dumps, secret keys.

Illegal or unauthorized data: stolen PII, medical records without authority, minors’ data, copyrighted content you don’t own.

Sensitive traits inference: race, religion, sexual orientation, health status—not used in risk scoring.

Full card/CVV, full passwords or API keys (uploads containing these are rejected).

Personal investment advice (risk tool ≠ advisor).

WHERE CHANGES NEED TO BE MADE

src/app.py: server-side validation to detect/reject disallowed content; return a clear error.

Add a policy check module that scans extracted text for banned patterns (PAN/CVV/API key regex).

Data handling, privacy, and compliance

Use case: internal AML/KYC triage; not client-facing.

PII handling: encrypt at rest; restrict access; mask logs/metrics.

Retention: configurable (e.g., 30 days); purge on request.

Localization: host in EU (Hetzner EU) to align with GDPR.

Audit trail: store hash of inputs, model version, thresholds, timestamp, user id.

WHERE CHANGES NEED TO BE MADE

src/config.py: DATA_RETENTION_DAYS, REGION="EU".

Logging middleware to hash evidence and strip PII from logs.

Minimal use-case flows (v1)

Flow A — Classify a new client

Analyst uploads evidence or pastes structured text.

Backend extracts text, checks policy, masks PII.

MiniLM classifier returns {label, probs, factors}.

App shows result + Why (sanctions flag, inflow ratio, media count).

Analyst acknowledges/escalates.

Flow B — Re-score with added evidence

Analyst adds missing document/snippet.

System re-runs classifier; shows delta (what swung the score).

New audit record written.

Guardrails at inference

Confidence thresholds: if high but confidence < threshold, auto-escalate for review.

Business rule overrides: any sanctions/PEP match → force HIGH regardless of model.

Monitoring & Dashboards

- Prometheus metrics exposed at `/metrics` with:
  - `request_latency_seconds{route}` histogram for p50/p95/p99
  - `decisions_total`, `pred_total{label}`, `autoclear_low_total`
  - `sanctions_present_total`, `sanctions_override_total`, `audit_events_total`
- Grafana dashboard: import `ops/grafana/kyc_api_dashboard.json` and set Prometheus datasource. Panels include latency, QPS, error rate, decisions by label, auto‑clear %, and sanctions present vs override.
- Alerts: example Prometheus rules under `ops/alerts.yaml` (p95 > 300ms, error rate > 5%, throughput drop).

Reproducible Policy & Audit

- Policy config uses `config/risk_rules.yaml`. API responses include `policy` (temperature + bands) and `policy_config_hash` (SHA256 of config) for reproducibility.
- Audit log `logs/audit.jsonl` writes one JSON per decision with HMAC over a stable payload. Verify with `scripts/audit_verify.py`.

Calibration & Bands

- Fit temperature on validation set: `python scripts/fit_temperature.py --checkpoint api/minilm_cls_best.pt --file data/valid.jsonl --out config/calib.json --write_config config/risk_rules.yaml`
- Confidence bands (non‑breaking): defaults `low_min=0.70`, `high_min=0.80`; UI shows a band chip and policy footer.

Quality Gates (CI)

- `.github/workflows/quality-gates.yml` runs:
  - Unit tests (sanctions recall, policy bands, API/audit, metrics smoke)
  - Temperature fitting (bounds 0.5–3.0)
  - Eval on valid set with override; fails if Macro‑F1 < 0.90 or High recall (override) < 0.99.

Calibration: temperature scaling to keep probabilities honest.

WHERE CHANGES NEED TO BE MADE

src/config.py: HIGH_FORCE_ON_SANCTIONS=True, confidence thresholds.

Post-processing in src/infer_minilm.py.

Kickoff (1-week plan)

Day 1

Scaffold repo (structure you already saw).

Set src/config.py.

Wire MiniLM model + head.

Day 2

Implement build_dataset.py weak-labeler (OpenSanctions subset + synthetic lows/mediums).

Generate data/train.jsonl, data/valid.jsonl.

Day 3

Train head-only (train_minilm.py).

Add evaluation (accuracy, per-class F1).

Save minilm_cls_best.pt.

Day 4

Add inference + factor display (sanctions hit, inflow_ratio, media_count).

Implement policy checks on uploads.

Day 5

FastAPI /classify endpoint (src/app.py).

Deploy on Hetzner CX22 (CPU).

Add audit logging + retention job.

Stretch

Add basic UI; integrate OFAC/OFSI live lookups; calibration & thresholds UI.

One-liner user guidance (to show in the app)

Do: upload KYC/statement PDFs or paste short evidence summaries.

Don’t: upload full card numbers, CVV, passwords, API keys, or illegal content.

Note: this tool assists analysts; it does not make final decisions or give investment advice.
What the app does (now)

Input: a short “evidence” string per client (e.g., [KYC] Name: … [COUNTRY] … [SANCTIONS] list=… [MEDIA] …).

Model: MiniLM-L6 classifier (binary or ternary).

Policy: if any sanctions code is present → force HIGH (override).

Output: label + probabilities + compact “why” reasons (sanctions codes, media count, inflow ratio, bursts, country).

Interfaces: CLI + optional REST (FastAPI) or Streamlit UI.

High-impact features you can add (fast)

I’m listing what it adds + where changes need to be made.

PEP + sanctions updater (daily auto-refresh)

Value: fresh lists without redeploys.

Changes:


Manual Eval Harness (prove 18/21)

- Script: `src/eval_manual.py`
  - Reads `data/manual_test.jsonl` with one JSON per line: `{ "text": "...", "label": "low|medium|high" }`.
  - Calls the running API at `http://localhost:8000/classify` (configurable via `--api`).
  - Prints accuracy, per‑class precision/recall/F1, and a confusion matrix.
  - Flags failure when accuracy drops below a threshold via `--min_acc`.

Quick start

- Start API: `docker compose up -d api`
- Run: `python src/eval_manual.py --file data/manual_test.jsonl --api http://localhost:8000 --min_acc 0.85`
  - Use `--no-override` to evaluate without sanctions→HIGH override.

Expected JSONL format

- Minimal line: `{ "text": "...", "label": "high" }`
- Alternate label keys also supported: `expected`, `y`, or `target`.
- Labels may be strings (`low|medium|high`) or integers (`0|1|2`).

CI guardrail (GitHub Actions)

- Workflow: `.github/workflows/eval-manual.yml`
  - Builds and starts the API with Docker Compose.
  - Waits for `GET /health` to pass.
  - Runs `src/eval_manual.py` against `data/manual_test.jsonl` with `--min_acc 0.85`.
  - Skips gracefully if the file is missing.

Why this matters

- Keeps a small hand‑picked test set that captures tricky edge cases.
- Prevents silent regressions from model or rule changes.
- Produces a readable confusion matrix and per‑class F1 for quick diagnosis.

src/data_update.py – download OpenSanctions CSV/JSON, write to data/raw/….

src/build_dataset.py – re-run slimming; keep same schema.

(If using API) src/app.py – add /admin/reload to reload lists without restart.

Country risk weighting (FATF)

Value: raise risk for high-risk jurisdictions.

Changes:

data/raw/fatf_risk.json – simple mapping { "IRAN": "high", ... }.

src/infer_minilm.py – in build_reasons(), append FATF: high-risk; if high-risk & model=medium/low, bump to next tier (optional).

src/ui_streamlit.py – toggle “Apply FATF weighting”.

Adverse media score (light RAG)

Value: real media signal beyond [MEDIA] N.

Changes:

src/ingest_news.py – read a small news dump, embed with MiniLM, store FAISS index.

src/infer_minilm.py – optional retrieval of top-k snippets; if any hit on sanctions/keywords → add reason & nudge risk.

Keep offline corpus for class demo (no live crawling).

Entity matching / name fuzzing

Value: catch aliases & transliterations.

Changes:

requirements: rapidfuzz, unidecode.

src/infer_minilm.py – add match_name(candidate, watchlist_names) using token-set ratio; add reason “alias match ~87%”.

src/build_dataset.py – keep aliases column normalized.

Behavior anomaly checks (toy rules)

Value: demonstrates behavior risk.

Changes:

src/infer_minilm.py – parse [INFLOW], [TRADES_7D], etc.; if TRADES_7D≥50 or INFLOW_RATIO≥3, add reasons & bump medium/high.

Threshold calibration

Value: more stable decisions (not just argmax).

Changes:

src/train_minilm.py – save temperature via simple temperature scaling on valid set.

src/infer_minilm.py – apply logits / T before softmax; store T in a small calib.json.

Audit trail

Value: compliance traceability.

Changes:

src/app.py – on each /classify, append {text_hash, label, probs, rule, why, timestamp} to logs/audit.jsonl.

src/ui_streamlit.py – “Download last N decisions” button.

Human-in-the-loop overrides

Value: analyst can accept/override with note.

Changes:

src/app.py – POST /cases/{id}/override with {label, note}; log it.

src/ui_streamlit.py – “Approve / Escalate” buttons; write to logs/overrides.jsonl.

Batch screening

Value: upload CSV of clients; get bulk results.

Changes:

CLI: already supported via --file.

src/app.py – POST /classify_batch (accepts array of texts).

src/ui_streamlit.py – file uploader → show table with labels/why → export CSV.

Confusion/metrics dashboard

Value: see drift or class imbalance.

Changes:

src/eval_valid.py – already prints accuracy; add classification_report + per-class F1 and save metrics.json.

src/ui_streamlit.py – “Metrics” tab to render confusion & counts.

Guardrails / scope (keep it clean)

Do not upload real PII in class; use synthetic data.

No storage of raw documents unless encrypted; strip IDs/passport numbers.

No live web scraping without ToS review; use offline corpora for demo.

Keep a whitelist of tags in text: [KYC] [COUNTRY] [SANCTIONS] [MEDIA] [INFLOW] [TRADES_7D]. Ignore others.

Minimal roadmap (1–2 hours)

Implement FATF weighting and audit trail.

Files: infer_minilm.py, app.py.

KYC Risk Classifier — Kickoff Doc
1) One-liner

An AI-assisted KYC triage service that classifies customers into Low / Medium / High risk by combining a small transformer model (MiniLM) with policy-first rules (e.g., any sanctions ⇒ HIGH) and returns an explainable “why” list for analysts and audits.

2) Why this matters for banks

Reduce onboarding time: auto-clear low-risk applications; escalate only the few that matter.

Lower false negatives: deterministic overrides catch list hits; the model flags edge cases.

Auditable: every decision includes evidence-based reasons and an HMAC’d audit log.

Continuously screen: same pipeline re-scores the book nightly as lists/news change.

Cheaper: lightweight model runs on CPU; no GPU/LLM calls required.

3) Core features (MVP)

/classify API (FastAPI): input a structured evidence string, return {label, probs, rule, why}.

Deterministic policy override: any sanctions/PEP code → HIGH (configurable).

MiniLM classifier: trained on public sanctions/PEP data + synthetic KYC snippets.

Explainability: human-readable “why” (sanctions hit, media count, inflow spikes, bursts, country).

Audit trail: JSONL with SHA-256 + HMAC for tamper-evidence.

UI: simple web app to test cases, view probabilities, reasons, and copy JSON.

4) Tech stack

Model & Data

Transformer encoder: MiniLM-L6 (384-dim) + linear head (PyTorch).

Tokenization/embeddings: Hugging Face transformers.

Data handling: pandas, small scripts for OpenSanctions/FATF extraction.

Service

API: FastAPI + Uvicorn.

Config: risk_rules.yaml (thresholds, countries, override toggles).

Metrics: prometheus-client counters/histograms.

Frontend



Features: example prompts, policy toggle, probability bars, “why” bullets.

Ops

Packaging: Docker (api + ui).

Orchestration: docker-compose (API on :8000, UI on :3000).

Logging: JSONL files in /app/logs.

5) How it works (data flow)

Input: evidence text like
[KYC] Name: … [COUNTRY] … [SANCTIONS] list=… [MEDIA] … Inflow ratio=… Burst trades=…

Policy check: if sanctions/PEP code present → HIGH with reason (override).

Model scoring: MiniLM encodes text → softmax probabilities → predicted label.

Reasons: parse numeric signals (media counts, ratios, bursts, country/FATF) and compose a short justification list.

Audit: write {text_sha, label, rule, probs, why, hmac} to logs/audit.jsonl.

Response: API returns {label, probs, rule, why}; UI visualizes bars and bullets.

6) Security & Compliance (MVP posture)

Policy-first: deterministic rules outrank ML for sanctions—meets conservative control expectations.

PII handling: request is plain text; audit logs store redacted text hash, not raw PII.

Versioning: freeze model checkpoint + config for each deployment (reproducible outputs).

Config-driven: FATF/jurisdiction risk maintained in YAML—no code change needed.

7) Deployment

Local: docker compose up → UI at http://localhost:3000, API at /api (proxied).

Hetzner: run both containers on a CX instance (CPU-only OK). Add Nginx/Caddy for TLS.

Scaling pattern: stateless API → horizontal scale (Kubernetes or Nomad). UI served as static files.

8) Performance & cost

Model size: ~90 MB MiniLM + ~1 KB head.

Latency: 30–100 ms per request on CPU (max_len=256), ~1–3 ms if cached.

Throughput: ~20–60 RPS per vCPU (depends on max_len/batch). Batch endpoints for nightly re-screening.

9) KPIs to track

Onboarding TAT ↓ 30–60% (auto-clear low).

Analyst throughput ↑ 2–5× (triage).

Recall for sanctions hits ≥ 99% (override on).

Dispute rate / overturn rate of HIGH labels (quality signal).

Model drift: weekly validation accuracy on hold-out set.

10) Roadmap to scale & harden

Short term (1–3 weeks)

Add country/FATF weighting to “why” and optional label bump.

Threshold tuning for inflow/burst/media using your 21-case set (+ new labeled cases).

Probability calibration (temperature scaling) for better confidence thresholds.

Batch endpoints + S3/MinIO support for nightly re-screen.

Medium term (1–2 months)

Name/entity resolution: fuzzy match onboarding name vs. OpenSanctions via RapidFuzz/Embeddings.

Adverse media fetch: news API hits cached and summarized; feed media count into reasons.

Model upgrade: switchable encoders (e5-small, MPNet) via config without code change.

Feature store: persist numeric signals per customer; score deltas over time.

Enterprise hardening

Auth (mTLS/JWT), RBAC, request/response schema validation (Pydantic).

SIEM export: ship audit logs via Filebeat/Fluent Bit.

CICD gates: run unit tests + 21-case eval; block deploy if accuracy drops.

11) Limitations (transparent)

Sanctions detection is string-based in MVP—needs upstream list matching to be bulletproof.

The classifier is trained on public + synthetic text; institution-specific patterns will benefit from fine-tuning on internal labels.

Not a replacement for full AML transaction monitoring—this is an onboarding triage tool.

12) Demo script (2 minutes)

Paste a clean case → LOW, “no adverse media,” “country=DE,” confidence bars.

Paste a sanctions case → instantly HIGH with “Policy override: us-bis-el.”

Paste an edge case with Inflow ratio=3.6 + Burst trades=18 → MEDIUM/HIGH depending on thresholds; show reasons.

Open logs/audit.jsonl → show tamper-evident log line with HMAC.

TL;DR

This app gives a bank a fast, explainable, policy-safe KYC triage they can run cheaply on CPU. It’s useful now because it cuts onboarding time and analyst workload while staying auditable. It’s scalable because the API is stateless, rules are config-driven, models are swappable, and the architecture is containerized for horizontal scale.
Added metadata tags parsing and an audit search tool.

What changed

api/src/infer_minilm.py:1

parse_signals now extracts optional tags:
owner: [OWNER] ...
conf: [CONF] ...
ver: [VER] ...
lang: [LANG] (2–10 letters/dashes)
Kept existing fields (country, sanc_codes, media, inflow, bursts).
api/src/app.py:1

Classify responses include:
meta_tags: {owner, conf, ver, lang} when present
country and sanc_codes
Batch responses include the same.
These are also used for auto_clear checks (existing) and now for logs.
api/src/security.py:1

HMAC’d audit blob now includes meta_tags, country, and sanc_codes for searchable audits.
scripts/search_audit.py:1

Filter logs/audit.jsonl by:
--label, --rule, --band, --country, --owner, --lang
--since / --until (timestamp or ISO 8601)
--limit (defaults 100)
Prints matching JSON lines; outputs match count to stderr.
How to use

API: responses now carry metadata and light signals; no breaking changes for existing UI.
Search:
python scripts/search_audit.py --label high --rule sanctions_override --since "2025-01-01T00:00:00" --country DE
python scripts/search_audit.py --owner teamA --lang en --band low --limit 50
If you want, I can add a compact “Audit Search” page under the UI that calls this via an API shim, but the script should cover most needs for now.
Added an end-to-end Audit Search capability and surfaced metadata tags.

What I added

API search endpoint

api/src/app.py: appended GET /audit_search
Filters by label, rule, band, country, owner, lang, since/until, limit.
Accepts epoch seconds or ISO timestamps for since/until.
Returns matched audit lines (HMAC already present; no raw PII in logs).
Imported Query and added helper _parse_time.
UI search panel

ui/index.html: added “Audit Search” card with inputs and a Search button.
ui/app.js: added logic to build querystring, call /audit_search, and render a compact list with ts, rule, label, band, country, owner, lang.
Metadata in responses/logs

api/src/infer_minilm.py: parse_signals now extracts [OWNER], [CONF], [VER], [LANG].
api/src/app.py: include meta_tags, country, sanc_codes in classify responses.
api/src/security.py: log meta_tags, country, sanc_codes to audit.jsonl.
CLI helper (already added)

scripts/search_audit.py: CLI search for audit.jsonl (same filters as endpoint).
How to use

UI: open the page, fill any subset of filters in “Audit Search” and click Search.
API: GET /audit_search?label=high&rule=sanctions_override&country=DE&limit=50
CLI: python scripts/search_audit.py --label high --rule sanctions_override --since "2025-01-01T00:00:00"
Notes

Results are capped by limit (1–500). You can set AUDIT_LOG_PATH to point the API to a different audit file if needed.
Metadata tags are optional; include [OWNER], [CONF], [VER], [LANG] in the evidence text to populate them in responses and logs.
I’ve updated both UI and API to resolve case sensitivity and improve band filtering.

What changed

UI normalization
Lowercase: label, rule, band
Uppercase: country
Lowercase: owner, lang
File: ui/app.js:147
API search normalization
Case-insensitive filters for label, rule, band, owner, lang; case-insensitive country match is preserved.
Band behavior: when a band is provided, results include entries with missing band_label (old logs) and filter out only entries whose band_label is present and different.
File: api/src/app.py:323
Why this helps

Case-insensitive matching prevents “HIGH” vs “high” mismatches.
Older audit records without band_label are no longer excluded when a band is specified; only records explicitly labeled with a different band are filtered out.
Quick checks

UI: Enter “HIGH” in band/label fields — results should now match “high”.
API direct call examples:
Matches label case-insensitively: GET /audit_search?label=HIGH
Includes old entries when band provided: GET /audit_search?band=high
Owner/lang behave case-insensitively: GET /audit_search?owner=ACME&lang=EN
Added file upload support that auto-fills the textarea with the file’s text.

What I changed

HTML: added an optional file input under the sample chips

File: ui/index.html:27
Element:
Label: “Upload Text File (optional)”
Input: id="file-upload", accept=".txt,.json,.md,.csv"
JavaScript: wired a FileReader to populate the textarea and clear prior results/errors

File: ui/app.js:226
Behavior:
On #file-upload change, read as text.
Set #evidence value to file contents.
Clear error, hide previous result, focus textarea.
Notes

Client-side only; no server changes needed for upload.
The textarea content is replaced; to append instead, change the assignment to ev.value += ....
Works in modern browsers; older versions may need polyfills.
The app supports:

- __Ad-hoc inference__: Yes, via specific endpoints:

  - Single item: `/classify` (used by UI for manual inputs)
  - Retrieves instant risk classification, probabilities, and reasons

- __Batch processing__: Yes, via `/classify_batch`, handling multiple items in a single request. Includes batching logic for performance.

- __Single tenant__: The application appears single-tenant by design—no multi-tenant features like user isolation or dynamic environments. It's containerized (Docker) for isolation per deployment.

Key endpoints:

- Ad-hoc: POST `/classify` with single text
- Batch: POST `/classify_batch` with array of texts

The nightly batch script uses `/classify_batch` for automated processing.
Here’s what I implemented to meet your acceptance checklist.

Headers + Response Shape

Endpoints: /classify and /classify_batch return label, probs, why, rule
File: api/src/app.py:108
Response headers set on both routes:
X-Model-Id: model checkpoint hash or basename
X-Policy-Id: policy config hash
X-Request-Id: per-request UUID
Files:
api/src/app.py:217, api/src/app.py:254
Model ID computation: api/src/app.py:74
Policy hash: api/src/app.py:47
Auth: API Key

Requires API key for /classify and /classify_batch; 401 if missing/invalid
Accepts Authorization: Bearer KEY or X-API-Key: KEY
Files:
api/src/security.py:6
api/src/app.py:141, api/src/app.py:231
Config: set env var API_KEY to enable; if unset, auth is disabled (for dev).
Batch Cap + 413

Enforced max batch size via env BATCH_MAX (default 256)
Exceeds limit → 413 with helpful detail (“batch size N exceeds limit M”)
File: api/src/app.py:236
Audit Logging

One audit line per decision, including redacted text hash and HMAC
Added request_id to each audit log entry
Files:
api/src/security.py:35 (request_id field)
api/src/app.py:220, api/src/app.py:316 (pass request_id)
Existing redaction and HMAC preserved:
api/src/security.py:29, api/src/security.py:59
Metrics

Decision counts, overrides, latency histograms already present:
Counters: api/src/app.py:10–15
Latency: api/src/app.py:16–21; observe at api/src/app.py:224, api/src/app.py:321
Metrics endpoint: api/src/app.py:96
Health

Shows device + config status (and now policy hash, model id):
api/src/app.py:86
How to use

Set env:
API_KEY=your-key
BATCH_MAX=256 (optional)
Call endpoints with headers:
Authorization: Bearer your-key
or X-API-Key: your-key
Inspect headers on responses:
X-Model-Id, X-Policy-Id, X-Request-Id
I implemented the “quick wins” you outlined: server-side override policy, optional sanctions whitelist, richer override payload and audit, per-item override flags, split metrics, and a UI chip for clarity.

Server-side override policy

Adds a policy-controlled override switch and dry-run support.
Helper: api/src/infer_minilm.py:29
should_override(req_override_flag): forces override if policy.override_enabled=true; otherwise honors client flag.
Reads RULES.policy.override_enabled (default true) and RULES.policy.override_dry_run (default false).
classify and classify_batch use the server-side decision, not the UI flag.
Files:
api/src/app.py:148, api/src/app.py:247
Applies override when should_override is true and codes exist; if override_dry_run is true, does not change the label but includes override_would_apply=true.
Flags added to responses:
override_applied: true|false
override_would_apply: true when dry-run would have overridden
Sanctions whitelist (optional)

Whitelist valid sanctions codes via policy.sanctions.valid_codes.
Implementation leaves behavior unchanged unless you configure valid_codes (empty set = no filtering).
Files:
api/src/infer_minilm.py:11, api/src/infer_minilm.py:19
Tokenization still removes NONE_LIKE and punctuation; stops at next tag.
Richer override payload and audit

override_high_payload now accepts codes and includes:
override_reason: "sanctions_hit"
sanctions_codes (and sanc_codes) list
Files: api/src/infer_minilm.py:49
Audit log includes reason and codes if present.
Files: api/src/security.py:63
Metrics: split override vs model

pred_total now includes labels label and path (“model” or “override”).
Counter: api/src/app.py:10
Increments:
classify: api/src/app.py:214
classify_batch: api/src/app.py:311
Existing counters remain (decisions_total, sanctions_present_total, sanctions_override_total, latency histograms).
API handlers adjustments

classify
Uses extract_sanctions_codes and should_override; sets override_applied/override_would_apply, passes codes to override_high_payload.
Files: api/src/app.py:148–170, 206–225
classify_batch
Mirrors single behavior per item, preserves id, sets override flags per item, and passes codes.
Files: api/src/app.py:246–287, 295–322
UI clarity: override chip + codes

New chip shows when override is applied, listing codes if present.
Markup: ui/index.html:49
Logic: ui/app.js:82
UI now includes sanc_codes in mapped response for display.
Files: ui/app.js:30
Config usage

You can add these to config/risk_rules.yaml under policy:
policy.override_enabled: true|false
policy.override_dry_run: true|false
policy.sanctions.valid_codes: [us-bis-el, us-sdn, eu-blr, uk-hmt, …]
If not set, override is enabled by default and dry-run is off; no whitelist filtering is applied.
Notes

Tests: existing tests are preserved (whitelist filters only apply when configured).
Configurable thresholds (FATF, media, inflow)
Added targeted unit tests for sanctions code parsing, aligned to current behavior and parameterized to cover whitelist scenarios.

What I added

New tests: tests/test_sanctions_parsing.py
None value yields no codes and no override.
Basic code parsing and hit: “US-SDN”.
Mixed inputs: “none; US-SDN”.
Stop at next tag: “eu-blr [MEDIA] 0” extracts only “eu-blr”.
Whitelist behavior:
With a non-empty whitelist not containing “jp-meti”, “unknown | jp-meti” yields no codes/hit.
With whitelist containing “jp-meti”, it yields a hit.
Space-separated alias case:
With empty whitelist (default), “US SDN” produces tokens [“us”, “sdn”] and hits.
With a non-empty whitelist that doesn’t include those tokens, they are filtered out and do not hit.
Notes

Tests monkeypatch the internal _valid_codes set in src.infer_minilm so you can validate both default and whitelisted behavior without altering global config files.
These tests complement the existing recall tests and focus specifically on tricky parsing and whitelist filtering.
I’ll unify the YAML schema and wire the new FATF logic into the pipeline, keeping backward compatibility and not breaking existing behavior.

What I changed

Unified YAML schema

Replaced file: config/risk_rules.yaml:1
Adds:
version: 1
thresholds: media_high, inflow_ratio_low, inflow_ratio_high, burst_high/burst_low
fatf:
mode: "list" | "score"
high_risk, monitor, scores, bump_threshold
policy:
override_enabled, override_dry_run
bands.low_min/high_min
auto_clear.max_media/max_inflow
temperature
Keeps sensible defaults for fields the code expects but you didn’t include (min_low_prob, max_burst are defaulted in code).
FATF adapters in code

Load raw CFG from YAML alongside existing Pydantic RULES
File: api/src/infer_minilm.py:12, api/src/infer_minilm.py:16
New helpers:
country_bucket(country): respects fatf.mode with list or score semantics
File: api/src/infer_minilm.py:100
apply_fatf_bump(label, country): bumps low→medium, medium→high when bucket is “high”
File: api/src/infer_minilm.py:123
Apply bump via the unified config in pipeline:
apply_additional_rules now uses apply_fatf_bump instead of the older list-only helper
File: api/src/infer_minilm.py:158
Sanctions override config

Already implemented earlier:
should_override() respects policy.override_enabled / override_dry_run
override_high_payload(codes) includes machine-readable fields
classify and classify_batch use extract_sanctions_codes + should_override and set override_applied / override_would_apply
UI clarity (earlier work)

Shows “Policy override” chip with code list when applied
Notes and rationale

Backward-compatibility: Kept RULES usages for thresholds and policy; added CFG for new fatf.mode/score features. If extra YAML keys aren’t in Pydantic models, they are still read via the raw CFG.
Reasons text: I left the existing country reason lines intact for now (generic “FATF high-risk/no flags”). If you want the precise “mode-driven” reasons (“monitor”, “high_score ≥ cut”), I can replace that block to use country_bucket for messaging as in your spec.
Auto-clear: Your existing guardrails still apply; missing policy fields fall back to defaults (min_low_prob 0.65, max_burst 10). The new YAML only specifies max_media and max_inflow; the code already handles absent keys.
Tests: Your sanctions parsing tests were added previously. If you want FATF list/score bump unit tests, I can add a small set that monkeypatches CFG to simulate modes.
Yes — I updated build_reasons to emit the exact list vs score messages and added unit tests for both country_bucket and apply_fatf_bump.

Code changes

build_reasons: mode-aware FATF messages

File: api/src/infer_minilm.py:190
Behavior:
high_risk → “FATF high-risk jurisdiction (policy bump applied)”
monitor → “FATF on monitoring list”
high_score → “FATF score ≥ {cut} (policy bump applied)”
else → “no additional FATF flags”
apply_fatf_bump: bump only for high buckets

File: api/src/infer_minilm.py:126
Now bumps only when country_bucket is "high_risk" or "high_score"; monitor does not bump.
New tests

tests/test_fatf_policy.py
test_country_bucket_list_mode: verifies list-mode buckets (high_risk/monitor/none).
test_apply_fatf_bump_list_mode: verifies bump for high_risk, no bump for monitor/none.
test_country_bucket_score_mode_and_bump: verifies score-mode bucket with threshold and bump behavior.
Uses monkeypatch to set module-level CFG fatf settings per test.
Notes

Unified CFG was already loaded; these tests and changes align with that logic.
Done. I updated build_reasons to emit the exact list vs score FATF messages and added focused tests.

Changes

build_reasons messages

File: api/src/infer_minilm.py:190
Now emits:
high_risk: “FATF high-risk jurisdiction (policy bump applied)”
monitor: “FATF on monitoring list”
high_score: “FATF score ≥ {cut} (policy bump applied)”
else: “no additional FATF flags”
Only bump on high buckets

File: api/src/infer_minilm.py:126
apply_fatf_bump now bumps only when bucket is high_risk or high_score (monitor does not bump).
New tests

tests/test_build_reasons.py
test_build_reasons_fatf_list_highrisk: asserts “FATF high-risk jurisdiction (policy bump applied)”.
test_build_reasons_fatf_list_monitor: asserts “FATF on monitoring list”.
test_build_reasons_fatf_score_high: asserts “FATF score ≥ 80 (policy bump applied)”.
test_build_reasons_no_flags: asserts “no additional FATF flags”.
Uses monkeypatch to set module CFG and crafted text with [COUNTRY], [SANCTIONS] list=none, [MEDIA] 0.
Implemented the quick validation upgrades and configurability you asked for.

Configurable redaction + metrics

Stronger redaction driven by YAML config
File: api/src/security.py:2, 12
Loads patterns from config and applies:
long_number, alnum_id, iban, swift, email, phone
Counts replacements per type with Prometheus:
Counter: redactions_applied{type=...}
New audit counter
Counter: audited_total increments after each log line
File: api/src/security.py:69
YAML defaults added
File: config/risk_rules.yaml:38
redaction placeholders configured per type
Reason trace for audits

Responses now include a minimal machine-readable trace (kept out of UI by default but logged in audit):
classify: builds trace for sanctions override and non-override paths
classify_batch: mirrors per item
Files:
api/src/app.py:159, 168, 178, 298, 305
Audit log includes trace
File: api/src/security.py:64
HMAC verification helper

Script: tools/verify_audit.py
Recomputes HMAC on each line and prints OK/FAIL
Uses AUDIT_HMAC env var (defaults to change_me)
Input validation guardrails

classify now rejects inputs missing required tags [KYC] or [COUNTRY] with a 400 and non-PII message
File: api/src/app.py:145
How to validate now

Redaction
Send a PII-rich classify request (e.g., with email/IBAN/phone)
Inspect logs/audit.jsonl: PII fields replaced with placeholders; only text_sha is stored
Metrics: scrape /metrics for redactions_applied{type=...} and audited_total
HMAC tamper check
Modify a line in logs/audit.jsonl and run:
AUDIT_HMAC=change_me_in_prod python tools/verify_audit.py
Expect “HMAC MISMATCH” for the edited line and “FAIL”; otherwise “OK”
Explainability
curl /classify and check .why includes FATF messages
Trace is present in audit log lines (not required by UI)
Notes

I kept build_reasons returning strings (compatibility). Trace is assembled in the route handlers to minimize churn.
If you prefer trace to be generated centrally (e.g., a build_reasons_and_trace helper), I can refactor to that pattern.
The new counters coexist with existing ones; dashboards can show both audited_total and audit_events_total.
Added comprehensive tests to cover the missing areas you identified and wired minor code updates to support them.

New tests

tests/test_audit_search.py
Writes a temp audit.jsonl and validates:
Case-insensitive filters for label, band, country, owner/lang.
Band filter includes entries missing band_label.
tests/test_batch_inference.py
Validates /classify_batch returns per-item responses, preserves ids, and sets override_applied.
Enforces BATCH_MAX via env; > cap returns 413 with helpful error.
Asserts response includes X-Request-Id header.
tests/test_config_and_thresholds.py
Overrides media_high threshold and asserts reasons include “Adverse media” when [MEDIA] >= threshold.
tests/test_audit_integrity.py
Uses security.log_decision to write a single audit line to a temp file, recomputes HMAC to ensure integrity, then tampers and detects mismatch.
Directly tests security.redact on email/IBAN/SWIFT/phone/IDs.
Supportive code updates

api/src/app.py:145
Added minimal tag validation for /classify: missing [KYC] or [COUNTRY] → 400 with non-PII error.
api/src/app.py:159, 174, 298, 314
Attach a compact trace object in responses to be included in audit logs.
api/src/security.py
Added configurable redaction with YAML-backed patterns; metrics counters:
audited_total
redactions_applied{type=...}
Audit log now includes trace field.
Config

config/risk_rules.yaml:38
Added redaction defaults per pattern key.
Tooling

tools/verify_audit.py
HMAC verification helper script for logs/audit.jsonl (OK/FAIL).
Notes

API key enforcement remains disabled unless API_KEY env var is set (tests assume it’s unset).
Tests avoid interfering with global audit logs by using temp files where appropriate (audit_search uses AUDIT_LOG_PATH env; HMAC test writes to tmp path via log_decision’s path arg).
Here’s how to wire the dashboard end-to-end and the exact PromQL for each panel, aligned with the app’s metrics.

Setup

Prometheus scrape
prometheus.yaml:
scrape_configs:
job_name: 'kyc-api'
scrape_interval: 10s
static_configs:
targets: ['localhost:8000']
Grafana import
Add Prometheus datasource (name: Prometheus).
Import ops/grafana/kyc_api_dashboard.json and select the datasource.
PromQL Cheatsheet (per panel)

Latency p50/p95/p99 (per route)
p50: histogram_quantile(0.50, sum by (le) (rate(request_latency_seconds_bucket{route="classify"}[5m])))
p95: histogram_quantile(0.95, sum by (le) (rate(request_latency_seconds_bucket{route="classify"}[5m])))
p99: histogram_quantile(0.99, sum by (le) (rate(request_latency_seconds_bucket{route="classify"}[5m])))
QPS (decisions/sec)
sum(rate(decisions_total[1m]))
Error rate (see note)
If you have HTTP status counters: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
If not, use a reverse-proxy/exporter (nginx/apache/otel/http) or blackbox exporter. App doesn’t currently export per-status counters.
Auto‑clear %
100 * (sum(rate(autoclear_low_total[5m])) / sum(rate(decisions_total[5m])))
Sanctions: present vs override
Present: sum(rate(sanctions_present_total[5m]))
Override: sum(rate(sanctions_override_total[5m]))
Override ratio: (sum(rate(sanctions_override_total[5m])) / sum(rate(sanctions_present_total[5m])))
Label distribution (by path=model|override)
sum by (label, path) (rate(pred_total[5m]))
For path-only split: sum by (path) (rate(pred_total[5m]))
Audit integrity/ops (optional panels)
Audit lines/sec: sum(rate(audited_total[5m]))
Redactions/sec: sum by (type) (rate(redactions_applied[5m]))
KPI Targets (example thresholds/alerts)

Latency p95 ≤ 300ms: histogram_quantile(0.95, …) > 0.3 for 5m → alert.
Auto‑clear ≥ 70%: 100 * (rate(autoclear_low_total[15m]) / rate(decisions_total[15m])) < 70 → alert.
Override sanity: ratio of override to present near expected baseline (watch for drops/spikes).
Notes

The dashboard JSON expects a Prometheus datasource named Prometheus.
App exports: request_latency_seconds (histogram, label route), decisions_total, autoclear_low_total, sanctions_present_total, sanctions_override_total, pred_total{label,path}, audit_events_total, plus added audited_total and redactions_applied{type}.
For error rate, consider adding HTTP status metrics at the API or via a sidecar/proxy if you need first-party visibility.
Want me to update ops/grafana/kyc_api_dashboard.json to add panels for redactions and audit lines, and wire the exact queries above?

yea


I added two new Grafana panels to ops/grafana/kyc_api_dashboard.json to visualize audit and redaction metrics.

What’s added

Audit lines/sec

Panel title: “Audit lines/sec”
Query: sum(rate(audited_total{job="$job"}[5m]))
Grid: x=0, y=24, w=12, h=8
Redactions/sec by type

Panel title: “Redactions/sec by type”
Query: sum by (type) (rate(redactions_applied{job="$job"}[5m]))
Grid: x=12, y=24, w=12, h=8
How to use

Ensure Prometheus scrapes /metrics and sees audited_total and redactions_applied{type}.
Import or refresh the dashboard in Grafana (datasource “Prometheus”).
Thanks for the stack trace — the crash is due to the YAML schema change: version was set to an integer, but RiskRules expects a string.

Fixes applied

YAML: changed version from 1 to "1"
File: config/risk_rules.yaml:1
Loader hardening: coerce version to string if it’s numeric
File: api/src/config.py:18
Why this resolves it

RiskRules.model_validate enforces types from the Pydantic model. With version cast to a string (or coerced by the loader), validation succeeds and the app can start.
Next steps

Restart the API. It should import and serve normally now.
If you want broader coercion for other fields in the future, we can extend that loader shim accordingly.
 added full .txt and .jsonl upload flows with status messages, CSV export, and accessibility, and wired them to the override toggle.

What changed

UI (placement + copy)

File: ui/index.html:69
Adds upload control directly under the textarea:
Label: “Upload .txt or .jsonl”
Input: accept=".txt,.jsonl", id="file-upload"
Helper text: “.txt fills the box • .jsonl runs batch”
Status line: #file-status with aria-live="polite"
Removed the mistaken file input that was inside the result section.
JavaScript (TXT + JSONL flows, CSV, drag & drop)

File: ui/app.js:198
TXT
Loads file into textarea (soft cap 100k chars; shows “(truncated)” if over).
Clears previous errors/results, focuses textarea.
Status: “Loaded filename.txt (N chars) into the box.”
JSONL
Parses up to 10k valid lines with a text field.
Sends single /classify_batch request with {items, override: ov.checked}.
Renders the first item’s result on the right for immediate feedback.
Builds and auto-downloads CSV with headers: idx,id,label,rule,low,medium,high,why.
Status: “Processed N items from filename.jsonl. CSV downloaded.”
Uses AbortController with a 60s timeout; disables controls during processing.
Error states
Unsupported type, empty file, malformed JSONL (no valid lines) → clear errors, set messages.
Accessibility
Status messages go to #file-status with aria-live="polite".
Drag & drop
Drop a .txt onto the textarea to load it (same cap/status).
Dropping a .jsonl onto the textarea shows: “Use the upload control for batch files.”
Notes

Both flows respect the “Policy override” toggle: TXT uses it via the Assess button; JSONL passes override: ov.checked to /classify_batch.
No client-side logging of contents.
UI disables controls during batch; re-enables on completion or error.
If you want a feature flag to enable/disable batch quickly, I can add a simple global in config.js the UI checks to hide/disable the .jsonl path.
The existing server redaction is already on; we also added HMAC audit verification tooling earlier.
QA checklist pointers

TXT:
Try small/large/empty files; non-UTF8 shows error reading.
Editing after load works; Assess uses edited text.
JSONL:
Mixed valid/invalid lines; very large file up to a few thousand lines remains responsive; CSV matches fields.
Override toggle changes results accordingly.
Network:
If API down: message appears; controls re-enable.
Slow API: status and disabled state visible; recovers on return.
