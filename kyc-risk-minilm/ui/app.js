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
  const tabs = document.querySelectorAll('.tab');
  const panes = document.querySelectorAll('.pane');

  // THEME: init + toggle
  (function initTheme(){
    const pref = localStorage.getItem('theme') || 'light';
    document.body.classList.toggle('theme-light', pref === 'light');
    document.body.classList.toggle('theme-dark', pref === 'dark');
    const tbtn = $("theme-toggle");
    if (tbtn) {
      tbtn.textContent = (pref === 'dark') ? '☀️' : '🌙';
      tbtn.addEventListener('click', () => {
        const curr = document.body.classList.contains('theme-dark') ? 'dark' : 'light';
        const next = curr === 'dark' ? 'light' : 'dark';
        document.body.classList.toggle('theme-dark', next === 'dark');
        document.body.classList.toggle('theme-light', next === 'light');
        localStorage.setItem('theme', next);
        tbtn.textContent = (next === 'dark') ? '☀️' : '🌙';
      });
    }
  })();

  // Tabs
  tabs.forEach(tab=>{
    tab.addEventListener('click', () => {
      const t = tab.dataset.tab;
      tabs.forEach(b=>b.classList.remove('is-active'));
      tab.classList.add('is-active');
      panes.forEach(p=>{
        const show = p.id === `pane-${t}`;
        p.hidden = !show;
        p.classList.toggle('is-visible', show);
      });
      tabs.forEach(b=>b.setAttribute('aria-selected', b===tab ? 'true':'false'));
    });
  });

  // Samples
  document.querySelectorAll(".chip[data-sample]").forEach(b => b.addEventListener("click", () => {
    ev.value = SAMPLES[b.dataset.sample]; showError(""); hideResult(); ev.focus();
  }));

  clr.addEventListener("click", () => { ev.value = ""; showError(""); hideResult(); ev.focus(); });

  ev.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") btn.click(); });

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

  function disable(on) {
    btn.setAttribute("aria-busy", on ? "true" : "false");
    btn.disabled = on; clr.disabled = on; ev.disabled = on; ov.disabled = on;
    const sk = $("loading-skel");
    if (sk) sk.hidden = !on;
    document.body.classList.toggle('loading', !!on);
  }

  function showError(msg) {
    const e = $("error");
    if (!msg) { if (e){ e.hidden = true; e.textContent = ""; } hideToast(); return; }
    if (e){ e.hidden = false; e.textContent = msg; }
    showToast(msg);
  }

  // Toast helpers
  let toastTO = null;
  function showToast(text) {
    const t = $("toast"); if (!t) return;
    t.textContent = String(text || '');
    t.hidden = false;
    clearTimeout(toastTO);
    toastTO = setTimeout(hideToast, 4000);
  }
  function hideToast(){ const t=$("toast"); if(!t) return; t.hidden=true; }

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
    (r.why || ["No reasons provided"]).slice(0, 8).forEach(w => {
      const li = document.createElement("li");
      li.className = 'reason-chip';
      li.textContent = w; li.title = w; ul.appendChild(li);
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

    // Tiny donut + radar viz (optional)
    if (typeof window.renderDonut === 'function') { try { window.renderDonut(r.probs, 'viz-donut'); } catch(_){} }
    if (typeof window.renderRadar === 'function') { try { window.renderRadar(r.probs, 'viz-radar'); } catch(_){} }

    $("copy").onclick = () => {
      const payload = { label: r.label, rule: r.rule, probs: r.probs, why: r.why, band: r.band, policy: r.policy, policy_config_hash: r.policy_hash };
      navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      $("copy").textContent = "Copied"; setTimeout(()=>{$("copy").textContent="Copy JSON";}, 1200);
      showToast('Copied result JSON to clipboard');
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

  if (sInput) {
    sInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (sBtn && !sBtn.disabled) runAuditSearch();
      }
    });
  }

  async function runAuditSearch() {
    sError.hidden = true; sError.textContent = '';
    sResults.textContent = 'Searching…';

    const q = (sInput?.value || '').trim();
    const limit = parseInt($("s-limit").value || '100', 10);
    let url = `${API}/audit_search?limit=${limit}`;
    if (q) url += `&q=${encodeURIComponent(q)}`;

    // button busy state
    if (sBtn) { sBtn.disabled = true; sBtn.setAttribute('aria-busy', 'true'); sBtn.textContent = 'Searching…'; }

    try {
      const r = await fetch(url, { cache: 'no-store' });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json(); // expected: { items, total, next_cursor? }
      const items = Array.isArray(data.items) ? data.items : [];
      const total = Number.isFinite(data.total) ? data.total : items.length;

      sResults.innerHTML = `
        <div style="margin-bottom:10px;padding:10px;background:#f0f0f0;border-radius:8px;">
          Showing ${items.length} of ${total} matches
        </div>
        ${ items.length ? items.map(renderAuditItemLite).join('') : '<div class="muted">No matches</div>' }
      `;
    } catch (e) {
      sError.hidden = false;
      sError.textContent = String(e.message || e);
      sResults.textContent = '';
    } finally {
      if (sBtn) { sBtn.disabled = false; sBtn.removeAttribute('aria-busy'); sBtn.textContent = 'Search'; }
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
      <div class="row-card">
        <div class="row-meta">${dt}${ctry}${owner}${lang}</div>
        <div class="row-title">${(it.label||'').toUpperCase()} • ${it.rule||''}${band}</div>
        <div class="row-why">${why}</div>
      </div>
    `;
  }

// -- Quick chips beneath the search box
function populateSearchChips() {
  const host = document.getElementById('s-chips');
  if (!host) return;

  const presets = [
    { label: 'HIGH only', q: 'label:high' },
    { label: 'Sanctions rule', q: 'rule:sanctions' },
    { label: 'Override hits', q: 'rule:sanctions_override' },
    { label: 'Germany', q: 'country:DE' },
    { label: 'FATF high-risk', q: 'country:IRAN OR country:KP' },
    { label: 'Last 24h', q: 'since:24h' }
  ];

  host.innerHTML = presets.map(p =>
    `<button class="chip small" data-q="${p.q}">${p.label}</button>`
  ).join('');

  host.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-q]');
    if (!btn) return;
    const val = (sInput.value || '').trim();
    const add = btn.dataset.q.trim();
    sInput.value = val ? `${val} ${add}` : add;
    sInput.focus();
  });
}

// -- Sidebar "Recent audit" (last 5)
async function loadRecentAudit() {
  const box = document.getElementById('recent-audit');
  if (!box) return;

  try {
    // Prefer last 24h; backend may ignore since if unsupported
    const url = `${API}/audit_search?limit=5&since=24h`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(await r.text());
    const payload = await r.json();

    // Accept both new shape {items} and legacy []
    const items = Array.isArray(payload) ? payload : (payload.items || []);
    if (!items.length) {
      box.innerHTML = `<div class="muted">No recent items</div>`;
      return;
    }

    const rows = items.map(it => {
      const dt = it.ts ? new Date(it.ts*1000).toLocaleString() : '';
      const band = it.band_label ? ` · ${it.band_label}` : '';
      const ctry = it.country ? ` · ${it.country}` : '';
      const rule = it.rule || '';
      const label = (it.label || '').toUpperCase();
      return `
        <div style="border:1px solid var(--line,#E5E7EB);border-radius:8px;padding:8px;margin:6px 0;">
          <div><strong>${label}</strong> · ${rule}${band}${ctry}</div>
          <div class="muted tiny">${dt}</div>
        </div>
      `;
    }).join('');

    box.innerHTML = rows;
  } catch (e) {
    box.innerHTML = `<div class="error tiny">${String(e.message || e)}</div>`;
  }
}

// ---- BATCH ASSESS TAB ----
function initBatchAssess() {
  // Feature flag check
  if (!window.FEATURE_FLAGS?.enableBatchAssess) {
    console.log('Batch Assess feature disabled');
    return;
  }

  // Show tab
  const batchTab = document.querySelector('.batch-tab');
  if (batchTab) batchTab.style.display = 'inline-flex';

  // Get refs
  const dropzone = document.getElementById('batch-dropzone');
  const fileInput = document.getElementById('batch-file-input');
  const statusEl = document.getElementById('batch-status');
  const previewEl = document.getElementById('batch-preview');
  const previewTableEl = document.getElementById('batch-preview-table');
  const progressEl = document.getElementById('batch-progress');
  const progressContentEl = document.getElementById('batch-progress-content');
  const submitBtn = document.getElementById('batch-submit');
  const cancelBtn = document.getElementById('batch-cancel');
  const overrideToggle = document.getElementById('batch-override-toggle');

  let jobId = '';
  let resultsData = [];
  let fileName = '';

  function setStatus(msg, type = 'muted') {
    if (statusEl) {
      statusEl.textContent = msg || '';
      statusEl.className = type === 'error' ? 'batch-status error' : 'batch-status';
    }
  }

  // Drag and drop handlers (scoped to dropzone only)
  function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

  function highlight() {
    dropzone.classList.add('dragover');
  }

  function unhighlight() {
    dropzone.classList.remove('dragover');
  }

  dropzone.addEventListener('dragenter', preventDefaults);
  dropzone.addEventListener('dragover', preventDefaults);
  dropzone.addEventListener('dragleave', preventDefaults);
  dropzone.addEventListener('dragleave', function(e) {
    if (!dropzone.contains(e.relatedTarget)) unhighlight();
  });
  dropzone.addEventListener('dragover', highlight);
  dropzone.addEventListener('dragenter', highlight);
  dropzone.addEventListener('dragleave', unhighlight);
  dropzone.addEventListener('drop', preventDefaults);
  dropzone.addEventListener('drop', unhighlight);

  // Click to browse
  dropzone.addEventListener('click', () => fileInput.click());

  // File input change
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) processFile(file);
  });

  // Drop
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  });

  async function processFile(file) {
    setStatus('Reading file...');
    fileName = file.name;

    // Basic validation
    if (file.size > 50 * 1024 * 1024) { // 50MB
      setStatus('File too large (max 50MB)', 'error');
      return;
    }

    const ext = file.name.split('.').pop().toLowerCase();
    if (!['csv', 'xlsx', 'xls', 'json', 'jsonl', 'txt', 'zip'].includes(ext)) {
      setStatus('Unsupported file type', 'error');
      return;
    }

    try {
      // Read as base64
      const reader = new FileReader();
      reader.onload = async (event) => {
        const fileData = event.target.result.split(',')[1]; // Remove data:;base64,
        setStatus('Uploading...');
        try {
          const r = await fetch(`${API}/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              file_data: fileData,
              file_name: file.name,
              file_type: ext
            })
          });
          if (!r.ok) throw new Error(await r.text());
          const jobStatus = await r.json();
          jobId = jobStatus.job_id;
          setStatus(`Job started: ${jobId}`);
          // Start polling
          pollJobStatus(jobId);
        } catch (e) {
          setStatus(`Upload failed: ${e.message}`, 'error');
        }
      };
      reader.readAsDataURL(file); // Base64 encoding
    } catch (err) {
      setStatus(`Error reading file: ${err.message}`, 'error');
    }
  }

  function showPreview() {
    previewEl.style.display = 'block';
    renderPreviewTable();
  }

  function renderPreviewTable() {
    if (!parsedData.length) return;

    const headers = Object.keys(parsedData[0]);
    const tableHtml = `
      <table>
        <thead>
          <tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>
        </thead>
        <tbody>
          ${parsedData.map(row =>
            `<tr>${headers.map(h => `<td title="${row[h]}">${row[h]}</td>`).join('')}</tr>`
          ).join('')}
        </tbody>
      </table>
    `;
    previewTableEl.innerHTML = tableHtml;
  }

  function hideProgress() {
    progressEl.style.display = 'none';
    progressContentEl.innerHTML = '';
  }

  async function pollJobStatus(jobId) {
    const interval = setInterval(async () => {
      try {
        const r = await fetch(`${API}/batch/${jobId}/status`);
        if (!r.ok) throw new Error(await r.text());
        const status = await r.json();
        updateProgressUI(status);

        if (status.status === 'done') {
          clearInterval(interval);
          if (status.download_url_results) {
            // Download results
            downloadResults(jobId);
          }
        } else if (status.status === 'failed') {
          clearInterval(interval);
          setStatus(`Job failed: ${status.last_error}`, 'error');
        }
      } catch (e) {
        console.error('Poll error:', e);
      }
    }, 2000); // poll every 2s
  }

  function updateProgressUI(status) {
    const { total_rows, processed_rows, success_count, eta_seconds, status: jobStatus } = status;
    let msg = `Status: ${jobStatus}`;
    if (processed_rows !== null) msg += ` • Processed: ${processed_rows}/${total_rows || '?'} • Success: ${success_count || 0}`;
    if (eta_seconds) msg += ` • ETA: ${Math.round(eta_seconds)}s`;
    setStatus(msg);
  }

  async function downloadResults(jobId) {
    try {
      const a = document.createElement('a');
      a.href = `${API}/batch/${jobId}/results.csv`;
      a.download = `batch-${jobId}-results.csv`;
      a.click();
    } catch (e) {
      console.error('Download error:', e);
    }
  }

  function showResultsTable() {
    progressContentEl.innerHTML = `
      <table id="results-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Label</th>
            <th>Rule</th>
            <th>Low %</th>
            <th>Medium %</th>
            <th>High %</th>
            <th>Why</th>
          </tr>
        </thead>
        <tbody id="results-tbody">
          ${resultsData.map(row => `
            <tr>
              <td>${row.id || ''}</td>
              <td>${row.label}</td>
              <td>${row.rule}</td>
              <td>${Math.round(row.probs.low * 100)}%</td>
              <td>${Math.round(row.probs.medium * 100)}%</td>
              <td>${Math.round(row.probs.high * 100)}%</td>
              <td>${row.why ? row.why.join('; ') : ''}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
      <div class="btn-group" style="margin-top:1rem;">
        <button class="btn btn-primary" onclick="downloadBatchResults('${fileName}', ${JSON.stringify(resultsData).replace(/"/g, '"')})">Download CSV</button>
      </div>
    `;
    progressEl.style.display = 'block';
  }

  function showBatchProgress(state) {
    progressEl.style.display = 'block';
    if (typeof state === 'string') {
      progressContentEl.innerHTML = `<div class="muted">${state}</div>`;
      return;
    }

    // State is results array
    const total = state.length;
    const byLabel = state.reduce((acc, it) => {
      const l = (it.label || 'unknown').toLowerCase();
      acc[l] = (acc[l] || 0) + 1;
      return acc;
    }, {});

    progressContentEl.innerHTML = `
      <div class="batch-stat-chip success">
        <strong>Total Processed:</strong> ${total}
      </div>
      ${Object.entries(byLabel).map(([label, count]) => `
        <div class="batch-stat-chip ${label === 'low' ? 'success' : label === 'medium' ? 'warning' : 'error'}">
          <strong>${label.toUpperCase()}:</strong> ${count}
        </div>
      `).join('')}
      <div class="btn-group" style="margin-top:1rem;">
        <button class="btn btn-primary" onclick="downloadBatchResults('${fileName}', ${JSON.stringify(state).replace(/"/g, '"')})">Download CSV Results</button>
      </div>
    `;
  }

  // Override toggle
  overrideToggle.addEventListener('click', () => {
    overrideToggle.classList.toggle('active');
  });

  // Cancel
  cancelBtn.addEventListener('click', () => {
    parsedData = [];
    fileName = '';
    previewEl.style.display = 'none';
    progressEl.style.display = 'none';
    setStatus('');
  });
}

function downloadBatchResults(filename, results) {
  const headers = ['id', 'label', 'rule', 'low', 'medium', 'high', 'why'];
  const rows = [headers.join(',')];

  results.forEach((it, idx) => {
    const low = Math.round((it.probs?.low || 0) * 100);
    const med = Math.round((it.probs?.medium || 0) * 100);
    const high = Math.round((it.probs?.high || 0) * 100);
    const why = (it.why || []).join('; ');
    rows.push(`${idx},${it.label},${it.rule},${low}%,${med}%,${high}%,"${why}"`);
  });

  const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = (filename ? filename.replace(/\.[^/.]+$/, '') : 'batch-results') + '.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

  populateSearchChips();
  loadRecentAudit();
// Init batch feature
  initBatchAssess();

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

// ---------- Dashboard charts ----------
(function () {
  const radarEl = document.getElementById('radarChart');
  if (radarEl) {
    new Chart(radarEl, {
      type: 'radar',
      data: {
        labels: ['Sanctions','Adverse Media','PEP','Geo / FATF','Anomalies','KYC Gaps'],
        datasets: [
          {
            label: 'This period',
            data: [68, 55, 42, 60, 35, 48],   // demo values, 0-100
            fill: true,
            borderWidth: 2,
            pointRadius: 2,
            backgroundColor: 'rgba(155, 143, 255, .15)',
            borderColor: '#9b8fff'
          },
          {
            label: 'Last period',
            data: [52, 39, 30, 46, 28, 36],
            fill: true,
            borderWidth: 2,
            pointRadius: 2,
            backgroundColor: 'rgba(107,111,245, .12)',
            borderColor: '#6b6ff5'
          }
        ]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          r: {
            angleLines: { color: 'rgba(0,0,0,.05)' },
            grid: { color: 'rgba(0,0,0,.05)' },
            suggestedMin: 0,
            suggestedMax: 100,
            pointLabels: { color: '#5a5f7a', font: { size: 12 } },
            ticks: { display: false }
          }
        }
      }
    });
  }

  const donutEl = document.getElementById('donutChart');
  if (donutEl) {
    new Chart(donutEl, {
      type: 'doughnut',
      data: {
        labels: ['LOW','MEDIUM','HIGH'],
        datasets: [{
          data: [62, 23, 15],     // demo distribution in %
          borderWidth: 0,
          hoverOffset: 4,
          backgroundColor: ['#27c084','#ffcf5c','#ff708b']
        }]
      },
      options: {
        cutout: '70%',
        plugins: { legend: { display: false } }
      }
    });
  }

  // Range pill toggle
  const pills = document.getElementById('range-pills');
  if (pills) {
    pills.addEventListener('click', (e) => {
      const btn = e.target.closest('.pill');
      if (!btn) return;
      pills.querySelectorAll('.pill').forEach(b => b.removeAttribute('aria-pressed'));
      btn.setAttribute('aria-pressed', 'true');
      // (Optional) trigger data reload here using btn.dataset.range
    });
    pills.addEventListener('keydown', (e) => {
      const buttons = Array.from(pills.querySelectorAll('.pill'));
      const activeIdx = buttons.findIndex(b => b.getAttribute('aria-pressed') === 'true');
      if (activeIdx === -1) return;
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        const next = e.key === 'ArrowRight'
          ? (activeIdx + 1) % buttons.length
          : (activeIdx - 1 + buttons.length) % buttons.length;
        buttons.forEach(b => b.removeAttribute('aria-pressed'));
        buttons[next].setAttribute('aria-pressed', 'true');
        buttons[next].focus();
        // (Optional) trigger data reload using buttons[next].dataset.range
      }
    });
  }
})();
