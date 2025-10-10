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

// ---- AUDIT SEARCH ----
const sBtn = $("s-run");
if (sBtn) {
  sBtn.addEventListener('click', async () => {
    const q = {
      label: $("s-label").value.trim() || undefined,
      rule: $("s-rule").value.trim() || undefined,
      band: $("s-band").value.trim() || undefined,
      country: $("s-country").value.trim() || undefined,
      owner: $("s-owner").value.trim() || undefined,
      lang: $("s-lang").value.trim() || undefined,
      since: $("s-since").value.trim() || undefined,
      until: $("s-until").value.trim() || undefined,
      limit: parseInt($("s-limit").value || '50', 10) || 50,
    };
    const params = new URLSearchParams();
    Object.entries(q).forEach(([k,v]) => { if (v !== undefined && v !== '') params.set(k,String(v)); });
    const url = `${API}/audit_search?${params.toString()}`;
    const err = $("s-error"); const out = $("s-results");
    err.hidden = true; err.textContent = '';
    out.textContent = 'Searching...';
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(await r.text());
      const items = await r.json();
      if (!Array.isArray(items)) throw new Error('Bad response');
      out.innerHTML = renderAudit(items);
    } catch(e) {
      err.hidden = false; err.textContent = String(e.message || e);
      out.textContent = '';
    }
  });
}

function renderAudit(items) {
  if (!items.length) return '<div class="muted">No matches</div>';
  const rows = items.map(it => {
    const dt = it.ts ? new Date(it.ts*1000).toISOString() : '';
    const band = it.band_label ? ` • band=${it.band_label}` : '';
    const ctry = it.country ? ` • country=${it.country}` : '';
    const mt = it.meta_tags || {}; const owner = mt.owner ? ` • owner=${mt.owner}` : '';
    const lang = mt.lang ? ` • lang=${mt.lang}` : '';
    return `<div class="tiny">${dt} • ${it.rule} • ${it.label}${band}${ctry}${owner}${lang}</div>`;
  });
  return rows.join('');
}
