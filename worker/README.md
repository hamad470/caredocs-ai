# Optional: shared Gemini connection for the live demo

Without this, the live demo still works: quoted answers need no AI, and
visitors can paste their own free Gemini key to see generated answers. Deploy
this Cloudflare Worker if you want **Write with Gemini** to work for every
visitor with no setup. The key stays in Cloudflare and never reaches the page.

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
