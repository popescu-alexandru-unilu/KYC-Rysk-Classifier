// Dynamic API base:
// - On local dev servers (commonly port 3000), talk directly to FastAPI on 8000
// - Otherwise (e.g., docker nginx), use '/api' so nginx proxies to the backend
window.API_BASE = (function(current){
  if (typeof current === 'string' && current.trim()) return current; // honor pre-set value
  var port = String(window.location.port || '');
  if (port === '3000') return 'http://localhost:8000';
  return '/api';
})(window.API_BASE);

// Feature flags
window.FEATURE_FLAGS = {
  enableBatchAssess: true,  // Default disabled for gradual rollout
};
