# KYC-Rysk-Classifier
deploy:http://46.62.218.2:3000/

## Deploy/Run Checklist (No-502, No-500 Path)

1) Preflight checkpoint (host)
- Validate a proper HF folder or .pt file before starting containers:
  - `python kyc-risk-minilm/scripts/ckpt_preflight.py`  # auto-detects common paths
  - or: `python kyc-risk-minilm/scripts/ckpt_preflight.py --path kyc-risk-minilm/minilm_cls_best`

2) Choose env preset
- Dev/bootstrapping: `--env-file kyc-risk-minilm/.env.dev`
- Prod (strict): `--env-file kyc-risk-minilm/.env.prod`

3) Start services
- `docker compose -f kyc-risk-minilm/docker-compose.yml --env-file kyc-risk-minilm/.env.dev up -d --build`

4) Verify routing
- From UI container:
  - `docker compose -f kyc-risk-minilm/docker-compose.yml exec ui sh -lc 'apk add --no-cache curl >/dev/null 2>&1 || true; curl -i http://api:8000/health'`
  - `docker compose -f kyc-risk-minilm/docker-compose.yml exec ui sh -lc 'curl -i -H "Content-Type: application/json" -H "X-API-Key: dev" -d "{\"text\":\"[KYC] Name: Test\\n[COUNTRY] France\\n[SANCTIONS] list=none\\n[MEDIA] 0 mentions\"}" http://api:8000/classify'`

5) Test via UI proxy
- Health: `curl -i http://<IP>:3000/api/health`
- Classify: `curl -s -H "Content-Type: application/json" -H "X-API-Key: dev" -d '{"text":"[KYC] Name: ACME LTD.\n[COUNTRY] Spain\n[SANCTIONS] list=none\n[MEDIA] 0 mentions"}' http://<IP>:3000/api/classify`

Notes
- UI nginx uses a static upstream with a trailing slash: `proxy_pass http://api:8000/;` (see `kyc-risk-minilm/ui/nginx.conf`).
- In strict prod (`.env.prod`), the API only becomes healthy when a valid checkpoint is mounted; the UI depends_on can be flipped to `service_healthy` if desired.
