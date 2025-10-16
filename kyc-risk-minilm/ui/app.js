document.addEventListener('DOMContentLoaded', () => {
  /* global window, document */
  const API = (window.API_BASE || "http://localhost:8000").replace(/\/+$/, "");
  document.getElementById("api-url").textContent = `API: ${API}`;

  const SAMPLES = {
    sanctions: "[KYC] Name: ACME Ltd.\n[COUNTRY] HK\n[SANCTIONS] list=US-BIS-EL\n[MEDIA] 3 mentions",
    clean:     "[KYC] Name: Jane Roe\n[COUNTRY] DE\n[SANCTIONS] list=none\n[MEDIA] 0 mentions",
    fatf:      "[KYC] Name: Example Co\n[COUNTRY] IRAN\n[SANCTIONS] list=none\n[MEDIA] 0 mentions",
  };

  const $ = (id) => document.getElementById(id);
  const ev = $("evidence"), ov = $("override"), btn = $("assess"), clr = $("clear");

  document.querySelectorAll(".chip").forEach(b => b.addEventListener("click", () => {
    ev.value = SAMPLES[b.dataset.sample]; showError(""); hideResult();
  }));

  clr.addEventListener("click", () => { ev.value = ""; showError(""); hideResult(); ev.focus(); });

  ev.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") btn.click();
  });

  btn.addEventListener("click", async () => {
    const text = ev.value.trim();
    if (!text) return;
    disable(true); showError(""); hideResult();

    const t0 = performance.now();
    try {
      const r = await fetch(`${API}/classify`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text, override: ov.checked, format: "json" })
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();

      // map API -> UI (CRITICAL is UI-only)
      const p = {
        low:    data.probs?.low    ?? 0,
        medium: data.probs?.medium ?? 0,
        high:   data.probs?.high   ?? 0,
      };
      const isCritical = data.rule === "sanctions_override" || p.high >= 0.98;

      const ui = {
        label: (isCritical ? "CRITICAL" : (data.label || "low").toUpperCase()),
        meta:  `${Math.round(performance.now() - t0)}ms`,
        probs: {
          LOW: p.low, MEDIUM: p.medium, HIGH: p.high,
          CRITICAL: isCritical ? Math.max(0.99, p.high) : 0
        },
        why: data.why || [],
        rule: data.rule || "model_only",
        band: (data.band_label || "").toUpperCase(),
        sanc_codes: data.sanc_codes || data.sanctions_codes || [],
        policy: data.policy || null,
        policy_hash: data.policy_config_hash || null,
      };
      render(ui);
    } catch (e) {
      showError(String(e.message || e));
    } finally {
      disable(false);
    }
  });

  function disable(on) { btn.setAttribute("aria-busy", on ? "true" : "false"); btn.disabled = on; clr.disabled = on; ev.disabled = on; ov.disabled = on; }

  function showError(msg) {
    const e = $("error");
    if (!msg) { e.hidden = true; e.textContent = ""; return; }
    e.hidden = false; e.textContent = msg;
  }

  function hideResult(){ $("result-card").hidden = true; }

  function render(r) {
    $("result-card").hidden = false;
    const badge = $("badge"), label = $("label"), meta = $("meta");

    badge.textContent = r.label;
    badge.className = "badge " + (r.label === "CRITICAL" ? "crit" : r.label === "HIGH" ? "high" : r.label === "MEDIUM" ? "med" : "low");
    label.textContent = `${r.label} RISK`;
    meta.textContent  = `${r.rule} • ${r.meta}`;

    const band = $("band");
    if (r.band) {
      band.hidden = false;
      band.textContent = `BAND: ${r.band}`;
      const bands = r.policy && r.policy.bands ? r.policy.bands : { low_min: 0.70, high_min: 0.80 };
      band.title = `Policy bands: low>=${bands.low_min}, high>=${bands.high_min}`;
      band.className = "chip " + (r.band === "HIGH" ? "danger" : r.band === "MEDIUM" ? "warning" : "success");
    } else {
      band.hidden = true;
    }

    // Policy override chip
    const ovr = document.getElementById("override-chip");
    if (r.rule === "sanctions_override") {
      ovr.hidden = false;
      if (Array.isArray(r.sanc_codes) && r.sanc_codes.length) {
        ovr.textContent = `Policy override: ${r.sanc_codes.join(', ')}`;
      } else {
        ovr.textContent = 'Policy override';
      }
    } else {
      ovr.hidden = true;
    }

    setBar("low",  r.probs.LOW);
    setBar("med",  r.probs.MEDIUM);
    setBar("high", r.probs.HIGH);
    const criticalRow = document.getElementById("row-critical");
    if (r.probs.CRITICAL > 0) {
      criticalRow.style.display = "grid";
      setBar("crit", r.probs.CRITICAL);
    } else {
      criticalRow.style.display = "none";
    }

    const ul = $("reasons"); ul.innerHTML = "";
    (r.why || ["No reasons provided"]).slice(0, 6).forEach(w => {
      const li = document.createElement("li"); li.textContent = w; ul.appendChild(li);
    });

    const policyMeta = $("policy-meta");
    if (r.policy) {
      const t = r.policy.temperature != null ? r.policy.temperature : 1.0;
      const bands = r.policy.bands || { low_min: 0.70, high_min: 0.80 };
      const hash = r.policy_hash ? String(r.policy_hash).slice(0, 8) : "";
      policyMeta.textContent = `Temp: ${t} • Bands: low>=${bands.low_min}, high>=${bands.high_min}${hash ? " • cfg:"+hash : ""}`;
    } else {
      policyMeta.textContent = "";
    }

    $("copy").onclick = () => {
      const payload = { label: r.label, rule: r.rule, probs: r.probs, why: r.why, band: r.band, policy: r.policy, policy_config_hash: r.policy_hash };
      navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      $("copy").textContent = "Copied"; setTimeout(()=>{$("copy").textContent="Copy JSON";}, 1200);
    };
  }

  function setBar(key, p) {
    const v = Math.round((p || 0) * 100);
    $(`p-${key}`).value = v; $(`t-${key}`).textContent = `${v}%`;
  }

  // ---- AUDIT SEARCH (works with index.html fields s-q, s-limit, s-run, s-export) ----
  const sInput = $("s-q");
  const sBtn = $("s-run");
  const sExportBtn = $("s-export");
  const sResults = $("s-results");
  const sError = $("s-error");

  async function runAuditSearch() {
    sError.hidden = true; sError.textContent = '';
    sResults.textContent = 'Searching...';

    const q = (sInput?.value || '').trim();
    const limit = parseInt($("s-limit").value || '100', 10);
    let url = `${API}/audit_search?limit=${limit}`;
    if (q) url += `&q=${encodeURIComponent(q)}`;

    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();               // { items, total, next_cursor }
      const items = Array.isArray(data.items) ? data.items : [];
      const total = Number.isFinite(data.total) ? data.total : items.length;

      // summary + rows
      sResults.innerHTML = `
        <div style="margin-bottom:10px;padding:10px;background:#f0f0f0;border-radius:4px;">
          Showing ${items.length} of ${total} matches
        </div>
        ${
          items.length
            ? items.map(renderAuditItemLite).join('')
            : '<div class="muted">No matches</div>'
        }
      `;
    } catch (e) {
      sError.hidden = false;
      sError.textContent = String(e.message || e);
      sResults.textContent = '';
    }
  }

  function renderAuditItemLite(it) {
    const dt = it.ts ? new Date(it.ts*1000).toISOString().replace('T',' ').split('.')[0] : '';
    const band = it.band_label ? ` • band=${it.band_label}` : '';
    const ctry = it.country ? ` • country=${it.country}` : '';
    const mt = it.meta_tags || {};
    const owner = mt.owner ? ` • owner=${mt.owner}` : '';
    const lang = mt.lang ? ` • lang=${mt.lang}` : '';
    const why = Array.isArray(it.why) ? it.why.slice(0,2).join(' / ') : '';
    return `
      <div style="border:1px solid #ddd;padding:8px;margin:4px 0;border-radius:4px;">
        <div><strong>${(it.label||'').toUpperCase()}</strong> • ${it.rule||''}${band}</div>
        <div>${dt}${ctry}${owner}${lang}</div>
        <div class="tiny">${why}</div>
      </div>
    `;
  }

  if (sBtn) {
    sBtn.addEventListener('click', runAuditSearch);
  }

  if (sExportBtn) {
    sExportBtn.addEventListener('click', () => {
      const q = (sInput?.value || '').trim();
      let url = `${API}/audit_export`;
      if (q) url += `?q=${encodeURIComponent(q)}`;

      // Force download reliably (avoid popup blockers)
      const a = document.createElement('a');
      a.href = url;
      a.download = 'audit_export.csv';
      document.body.appendChild(a);
      a.click();
      a.remove();
    });
  }

  // ---- FILE UPLOAD: TXT fills textarea, JSONL triggers batch ----
  const fileInput = document.getElementById("file-upload");
  const statusEl = document.getElementById("file-status");
  function setStatus(msg) { if (!statusEl) return; statusEl.textContent = msg || ''; }

  async function runBatch(jsonlText, filename) {
    // Parse JSONL, keep objects with a 'text' field
    const lines = jsonlText.split(/\r?\n/);
    const MAX = 10000;
    const items = [];
    for (let i = 0; i < lines.length && items.length < MAX; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      try {
        const obj = JSON.parse(line);
        if (obj && typeof obj.text === 'string' && obj.text.trim()) {
          items.push({ id: obj.id != null ? obj.id : i, text: obj.text });
        }
      } catch(_) { /* skip malformed */ }
    }
    if (!items.length) { showError("No valid JSON objects with a 'text' field were found."); return; }

    // Prepare request
    const payload = { items, override: ov.checked };
    const controller = new AbortController();
    const to = setTimeout(() => controller.abort(), 60000);
    disable(true); showError(""); hideResult(); setStatus(`Processing ${items.length} items from ${filename}...`);
    const t0 = performance.now();
    try {
      const r = await fetch(`${API}/classify_batch`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload), signal: controller.signal
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      if (!Array.isArray(data) || !data.length) throw new Error('Empty batch result');

      // Show first item compactly on the right
      const first = data[0];
      const p = { low: first.probs?.low ?? 0, medium: first.probs?.medium ?? 0, high: first.probs?.high ?? 0 };
      const isCritical = first.rule === 'sanctions_override' || p.high >= 0.98;
      const ui = {
        label: (isCritical ? 'CRITICAL' : (first.label || 'low').toUpperCase()),
        meta: `${Math.round(performance.now() - t0)}ms`,
        probs: { LOW: p.low, MEDIUM: p.medium, HIGH: p.high, CRITICAL: isCritical ? Math.max(0.99, p.high) : 0 },
        why: first.why || [],
        rule: first.rule || 'model_only',
        band: (first.band_label || '').toUpperCase(),
        sanc_codes: first.sanc_codes || first.sanctions_codes || [],
        policy: first.policy || null,
        policy_hash: first.policy_config_hash || null,
      };
      render(ui);

      const headers = ['idx','id','label','rule','low','medium','high','why'];
      const rows = [headers.join(',')];

      // helper: CSV-escape by doubling quotes and wrapping in quotes
      const csv = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;

      data.forEach((it, idx) => {
        const low = it.probs?.low ?? 0;
        const med = it.probs?.medium ?? 0;
        const high = it.probs?.high ?? 0;
        const why = Array.isArray(it.why) ? it.why.join('; ') : '';
        const idv = (it.id != null ? String(it.id) : '');
        rows.push([idx, csv(idv), it.label, it.rule, low, med, high, csv(why)].join(','));
      });

      const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = (filename ? filename.replace(/\.[^.]+$/, '') : 'batch') + '.csv';
      document.body.appendChild(a);
      a.click();
      a.remove();

      setStatus(`Processed ${data.length} items from ${filename || 'file'}. CSV downloaded.`);
    } catch (e) {
      showError(String(e.message || e));
    } finally {
      clearTimeout(to); disable(false);
    }
  }

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const name = file.name || 'file';
      const ext = (name.split('.').pop() || '').toLowerCase();
      if (ext !== 'txt' && ext !== 'jsonl') { showError('Unsupported file type. Use .txt or .jsonl.'); return; }
      const reader = new FileReader();
      reader.onerror = () => showError('Error reading file.');
      reader.onload = async (evnt) => {
        const text = String(evnt.target.result || '');
        if (!text) { showError('File is empty.'); return; }
        showError('');
        if (ext === 'txt') {
          const MAX = 100000;
          const val = text.length > MAX ? text.slice(0, MAX) : text;
          ev.value = val;
          hideResult(); ev.focus();
          setStatus(`Loaded ${name} (${val.length} chars) into the box${text.length>MAX?' (truncated)':''}.`);
        } else {
          await runBatch(text, name);
        }
      };
      reader.readAsText(file);
    });
  }

  // Drag & drop onto textarea (TXT only)
  if (ev) {
    const wrap = document.getElementById('evidence-wrap');
    ev.addEventListener('dragover', (evt) => { evt.preventDefault(); if (wrap) wrap.classList.add('dragover'); });
    ev.addEventListener('dragleave', () => { if (wrap) wrap.classList.remove('dragover'); });
    ev.addEventListener('drop', (evt) => {
      evt.preventDefault();
      if (wrap) wrap.classList.remove('dragover');
      const f = evt.dataTransfer && evt.dataTransfer.files && evt.dataTransfer.files[0];
      if (!f) return;
      const name = f.name || 'file';
      const ext = (name.split('.').pop() || '').toLowerCase();
      if (ext === 'jsonl') { setStatus('Use the upload control for batch files.'); return; }
      if (ext !== 'txt') { showError('Unsupported file type. Use .txt or .jsonl.'); return; }
      const reader = new FileReader();
      reader.onerror = () => showError('Error reading file.');
      reader.onload = (evnt) => {
        const text = String(evnt.target.result || '');
        if (!text) { showError('File is empty.'); return; }
        const MAX = 100000; const val = text.length > MAX ? text.slice(0, MAX) : text;
        ev.value = val; hideResult(); ev.focus();
        setStatus(`Loaded ${name} (${val.length} chars) into the box${text.length>MAX?' (truncated)':''}.`);
      };
      reader.readAsText(f);
    });
  }
});
