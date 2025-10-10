// k6 load test: p95 latency target <= 300ms (CPU)
// Usage: k6 run scripts/load.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  vus: 10,
  duration: '2m',
};

const body = JSON.stringify({
  text: "[KYC] Name: Test [COUNTRY] FR [SANCTIONS] list=none [MEDIA] 0 mentions",
  override: false,
  format: 'json',
});

export default function () {
  const res = http.post('http://localhost:8000/classify', body, {
    headers: { 'Content-Type':'application/json' },
  });
  check(res, {
    'status 200': r => r.status === 200,
    'p95<300': r => r.timings.duration < 300,
  });
  sleep(0.2);
}

