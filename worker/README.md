# Optional: shared Gemini connection for the live demo

You do not need this. By default the demo uses the keys in the
`GEMINI_API_KEYS` repository secret, which are added to the published page and
so are visible to anyone who inspects it. Deploy this Cloudflare Worker instead
if you want the keys kept server-side: they stay in Cloudflare and never reach
the browser. The page uses the Worker only when the secret is empty.

Cloudflare Workers' free plan is enough.

```bash
cd worker
npx wrangler login
npx wrangler secret put GEMINI_API_KEYS      # paste one or more keys, comma-separated
npx wrangler deploy
```

`wrangler deploy` prints a URL such as
`https://caredocs-gemini-proxy.<your-subdomain>.workers.dev`. Put it in
`site/config.js`:

```js
PROXY_URL: "https://caredocs-gemini-proxy.<your-subdomain>.workers.dev",
```

then commit and push; the Pages workflow republishes the site.

What the Worker does: accepts requests only from `ALLOWED_ORIGIN` (your
GitHub Pages address, set in `wrangler.toml`), allows at most 8 requests per
visitor per minute, rejects oversized prompts, only allows the three models the
demo offers, and tries the next key when one hits its free-tier limit.

Create each key in a different Google Cloud project; keys in one project share
a single quota.
