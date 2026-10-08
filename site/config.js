// Settings for the live demo.
export const CONFIG = {
  // Optional: URL of your deployed Cloudflare Worker (see worker/README.md).
  // When set, visitors get Gemini answers without entering their own key.
  PROXY_URL: "",
  DEFAULT_MODEL: "gemini-2.5-flash",
  MODELS: ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"],
  DATA_URL: "data/corpus.json",
  REPO_URL: "https://github.com/hamad470/caredocs-ai",
};
