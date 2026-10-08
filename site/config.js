// Settings for the live demo.
// Gemini keys are not stored here: build_rag_demo.py writes them to keys.js
// (git-ignored) from the GEMINI_API_KEYS environment variable / GitHub secret.
export const CONFIG = {
  // Optional alternative: URL of a Cloudflare Worker that holds the keys
  // server-side (see worker/README.md). Used only when keys.js has no keys.
  PROXY_URL: "",
  // Tried in order; the next model is used if one is unavailable.
  MODELS: ["gemini-2.5-flash", "gemini-2.0-flash"],
  DATA_URL: "data/corpus.json",
  REPO_URL: "https://github.com/hamad470/caredocs-ai",
};
