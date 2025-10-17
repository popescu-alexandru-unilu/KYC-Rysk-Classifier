/* Lightweight SVG donut for LOW/MED/HIGH (and CRITICAL overlay) */
(function(){
  const COLORS = {
    LOW: '#10B981', MEDIUM: '#F59E0B', HIGH: '#EF4444', CRITICAL: '#B91C1C',
    TRACK: '#EEF2F7'
  };

  function topLabel(p){
    const entries = Object.entries({LOW:p.LOW||0, MEDIUM:p.MEDIUM||0, HIGH:p.HIGH||0, CRITICAL:p.CRITICAL||0});
    entries.sort((a,b)=>b[1]-a[1]);
    const [label,val] = entries[0];
    return {label, val: Math.round((val||0)*100)};
  }

  function donutSVG(p){
    const size = 220, cx=110, cy=110, r=84, sw=16; // stroke width
    const C = 2*Math.PI*r;
    const pct = {
      LOW: Math.max(0, Math.min(100, Math.round((p.LOW||0)*100))),
      MEDIUM: Math.max(0, Math.min(100, Math.round((p.MEDIUM||0)*100))),
      HIGH: Math.max(0, Math.min(100, Math.round((p.HIGH||0)*100))),
      CRITICAL: Math.max(0, Math.min(100, Math.round((p.CRITICAL||0)*100)))
    };

    const segs = [ ['LOW', pct.LOW], ['MEDIUM', pct.MEDIUM], ['HIGH', pct.HIGH] ];
    const total = segs.reduce((s, [,v])=>s+v, 0) || 1;

    let offset = 0; // in percent of full circle
    const arcs = segs.map(([key,v])=>{
      const frac = v/100; // already percent
      const dash = (v/100)*C;
      const gap = C - dash;
      const dashoffset = -(offset/100)*C; // rotate to start position
      offset += v; // next segment starts after this
      return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${COLORS[key]}" stroke-width="${sw}" stroke-linecap="round" stroke-dasharray="${dash} ${gap}" stroke-dashoffset="${dashoffset}" />`;
    }).join('');

    // critical overlay inner ring if present
    let critical = '';
    if (pct.CRITICAL > 0){
      const r2 = 64, C2 = 2*Math.PI*r2, dash = (pct.CRITICAL/100)*C2, gap = C2-dash;
      critical = `<circle cx="${cx}" cy="${cy}" r="${r2}" fill="none" stroke="${COLORS.CRITICAL}" stroke-width="14" stroke-linecap="round" stroke-dasharray="${dash} ${gap}" />`;
    }

    const {label, val} = topLabel(p);
    return `
      <svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">
        <defs>
          <filter id="softShadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#11182722" />
          </filter>
        </defs>
        <g filter="url(#softShadow)">
          <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${COLORS.TRACK}" stroke-width="${sw}" />
          ${arcs}
          ${critical}
        </g>
        <g dominant-baseline="middle" text-anchor="middle" font-family="Inter,ui-sans-serif,system-ui" fill="#1F2937">
          <text x="${cx}" y="${cy-4}" font-size="26" font-weight="700">${val}%</text>
          <text x="${cx}" y="${cy+18}" font-size="12" fill="#6B7280">${label}</text>
        </g>
      </svg>`;
  }

  window.renderDonut = function(probs, elId='viz-donut'){
    try{
      const el = document.getElementById(elId);
      if(!el) return;
      el.innerHTML = donutSVG(probs||{});
    } catch(_e){ /* no-op */ }
  };
  
  // Simple 3-axis radar (LOW, MEDIUM, HIGH)
  function radarSVG(p){
    const size=240, cx=120, cy=110, r=90;
    const vals = [p.LOW||0, p.MEDIUM||0, p.HIGH||0];
    const axes = 3;
    const angleFor = (i)=> -Math.PI/2 + (i*(2*Math.PI/axes));
    // grid rings
    const rings = [0.33, 0.66, 1.0].map(f=>`<circle cx="${cx}" cy="${cy}" r="${r*f}" fill="none" stroke="#E5E7EB" stroke-width="1" />`).join('');
    const spokes = Array.from({length:axes},(_,i)=>{
      const a=angleFor(i); const x=cx+Math.cos(a)*r; const y=cy+Math.sin(a)*r;
      return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#E5E7EB" stroke-width="1" />`;
    }).join('');
    // polygon from values
    const pts = vals.map((v,i)=>{
      const a=angleFor(i); const rr=r*(Math.max(0,Math.min(1,v)));
      return `${cx+Math.cos(a)*rr},${cy+Math.sin(a)*rr}`;
    }).join(' ');
    return `
      <svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">
        <g stroke="#E5E7EB">${rings}${spokes}</g>
        <polygon points="${pts}" fill="rgba(108,99,255,0.18)" stroke="#6C63FF" stroke-width="2" />
        <g font-family="Inter,ui-sans-serif,system-ui" font-size="11" fill="#6B7280" text-anchor="middle">
          <text x="${cx}" y="${cy-r-6}">HIGH</text>
          <text x="${cx+r*Math.cos(angleFor(1))}" y="${cy+r*Math.sin(angleFor(1))+14}">LOW</text>
          <text x="${cx+r*Math.cos(angleFor(2))}" y="${cy+r*Math.sin(angleFor(2))+14}">MED</text>
        </g>
      </svg>`;
  }

  window.renderRadar = function(probs, elId='viz-radar'){
    try{
      const el=document.getElementById(elId); if(!el) return;
      el.innerHTML = radarSVG(probs||{});
    } catch(_e){}
  }
})();
