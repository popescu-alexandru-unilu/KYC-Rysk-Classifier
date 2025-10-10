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

Allowed uploads (file types & hygiene)

KYC docs (PDF/TXT), bank statements (PDF/CSV), adverse media excerpts (TXT), sanctions screenshots (PDF/PNG).

Hygiene rules:

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

Static SPA: vanilla JS + PicoCSS (or your Next.js variant).

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
