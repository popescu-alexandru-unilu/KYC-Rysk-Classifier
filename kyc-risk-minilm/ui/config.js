// Dynamic API base with environment flag:
// - If IS_LOCAL = true, talk directly to FastAPI on 8000
// - Otherwise, use the configured API URL from env.js
window.API_BASE = (function(current){
  if (typeof current === 'string' && current.trim()) return current; // honor pre-set value
  if (window.IS_LOCAL) return 'http://localhost:8000';
  return window.__ENV__?.API_BASE || '/api';
})(window.API_BASE);

// Feature flags
window.FEATURE_FLAGS = {
  enableBatchAssess: true,  // Default disabled for gradual rollout
};
