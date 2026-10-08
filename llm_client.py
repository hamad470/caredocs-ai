"""
llm_client.py — Multi-turn chat client for Gemini
`ai_service.py` performs single-shot narrative generation with a fixed system
prompt and a 400-token ceiling. Conversational RAG needs something different:
multi-turn message histories, per-call system prompts, larger output budgets,
and a JSON mode for the planner step. Rather than complicate the existing
module (which the care-note, incident and care-plan pages depend on), this file
adds a parallel client reading the same API keys from the same environment
variables written by `ai_config.py`.

One provider, several keys
An earlier version of this module spoke to three vendors — Anthropic, Google
and Groq — and fell through from one to the next when a call failed. That was
convenient for a demo and wrong for a dissertation. Silent failover between
vendors means the narrative on a care note might come from any of three models
with different training, different refusal behaviour and different prose, with
nothing in the record saying which. Two runs of the same evaluation could then
differ because a quota reset, not because anything under test had changed.

The client now speaks to Gemini only. What remains variable is which *key*
answers, not which model family, and that is a quota-management concern rather
than a scientific one: every key reaches the same models, so key rotation
cannot change what the system says, only whether it can say it right now.

    Env var             Purpose
    ──────────────────  ─────────────────────────────────────────────────────
    GEMINI_API_KEYS     comma-separated list, tried in order
    GEMINI_API_KEY      the first key; kept so single-key setups still work
    GOOGLE_API_KEY      accepted as an alias for GEMINI_API_KEY
    CHAT_GEMINI_MODEL   pins one model name, overriding discovery

Design notes:
  • urllib only — no SDKs, so the dependency list stays short.
  • Free-tier keys carry small per-minute and per-day quotas. A key that
    reports quota exhaustion goes into a short cooldown and the next key takes
    over; a key that is rejected outright is dropped for the session.
  • Model names are discovered per key rather than hard-coded, because Google
    retires names and two keys on different projects expose different lists.
  • `complete_json` never trusts the model to emit clean JSON — it strips code
    fences and extracts the outermost balanced object.
  • When no key works, callers get (None, None) and must fall back to the
    deterministic template path. Care documentation is a legal record, so
    "the API was down, so no note exists" is never an acceptable outcome.
"""

from __future__ import annotations

import os
import re
import json
import time
import urllib.request
import urllib.error

# Some providers sit behind Cloudflare, which blocks the default urllib
# User-Agent outright (HTTP 403, "error code: 1010" — a browser-integrity
# rejection that happens before the API key is even looked at). Sending a
# normal product User-Agent is what makes those requests pass.
USER_AGENT = os.environ.get("CAREHOME_USER_AGENT",
                            "CareHomeMVP/1.0 (+dissertation-research; python-urllib)")

GEMINI_HOST   = "https://generativelanguage.googleapis.com"
GEMINI_API_VERSIONS = ["v1beta", "v1"]
GEMINI_BASE   = f"{GEMINI_HOST}/v1beta/models"   # legacy constant, kept for callers

GEMINI_MODELS = [m for m in [os.environ.get("CHAT_GEMINI_MODEL", "")] if m] or [
    "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash",
]

PROVIDER = "gemini"          # the only backend; kept as a name for the audit log
DEFAULT_TIMEOUT = 60

_LAST = {"provider": None, "model": None, "latency_ms": 0, "error": None}

# Providers whose key has been rejected (HTTP 401/403) during this process.
# Without this, a single invalid key costs a full timeout on EVERY request for
# the lifetime of the app — which is what makes an app with one bad key feel
# "stuck" rather than "misconfigured". Cleared by reset_failures(), which
# ai_config calls whenever keys are saved.
_DEAD: dict[str, str] = {}
# Everything that went wrong this process, for the diagnostics panel.
_ERRORS: dict[str, str] = {}


def _record_error(provider: str, message: str, fatal: bool = False) -> None:
    _LAST["error"] = message
    _ERRORS[provider] = message
    if fatal:
        _DEAD[provider] = message


def reset_failures() -> None:
    """Forget rejected-key state — call after new keys are saved."""
    _DEAD.clear()
    _ERRORS.clear()
    _GEMINI_BAD.clear()
    _GEMINI_MODELS_BY_KEY.clear()
    _GEMINI_KEY_STATE.clear()


# Provider calls.  Each takes (messages, system, max_tokens, temperature,
# json_mode) and returns text or None.

def _post(url: str, payload: dict, headers: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": USER_AGENT,
                 "Accept": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, headers: dict, timeout: int = 20) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ── Gemini: multiple keys, model discovery, version walking ───────────────
# Two independent problems are solved here.
#
# 1. WHICH MODEL. Hard-coding model names is the usual cause of
#    "404 models/... is not found for API version v1beta": Google retires
#    names, and which names a key may call depends on tier and API version.
#    So the client asks ListModels what THIS key can call, ranks the answer,
#    and walks candidates across both API versions until one replies.
#
# 2. WHICH KEY. Free-tier keys have small per-minute and per-day quotas, so a
#    demo can die mid-sentence on HTTP 429. Several keys can therefore be
#    configured and are tried in order; a key that reports quota exhaustion is
#    put in a short cooldown and the next one takes over. Each key gets its own
#    model-discovery cache, because two keys on different projects genuinely
#    expose different model lists.

_GEMINI_PREFER = [
    "gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash",
    "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-1.5-flash",
]
_GEMINI_REJECT = ("embedding", "aqa", "vision", "image", "audio", "tts",
                  "imagen", "veo", "gemma", "learnlm", "live", "native-audio",
                  "computer-use", "robotics")
_GEMINI_DEMOTE = ("preview", "exp", "thinking", "pro")

# "keyid|model@version" pairs that returned 404. Keyed by the triple, not by
# model name: the same model routinely 404s under one API version and works
# under the other, and two keys can differ.
_GEMINI_BAD: set[str] = set()

# Per-key model discovery: {keyid: {models, version, chosen, error}}
_GEMINI_MODELS_BY_KEY: dict[str, dict] = {}

# Per-key health: {keyid: {status, error, cooldown_until, last_ok}}
_GEMINI_KEY_STATE: dict[str, dict] = {}

QUOTA_COOLDOWN_SECONDS = 90     # a 429 is usually per-minute, not per-day


def gemini_keys() -> list[str]:
    """
    All configured Gemini keys, in priority order.

    GEMINI_API_KEYS holds the full comma-separated list; GEMINI_API_KEY holds
    the primary one and is kept in sync so that any older code reading just
    that variable still works.
    """
    keys = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
    primary = (os.environ.get("GEMINI_API_KEY", "").strip()
               or os.environ.get("GOOGLE_API_KEY", "").strip())
    if primary and primary not in keys:
        keys.insert(0, primary)
    return keys


def key_id(key: str) -> str:
    """Short, non-secret label for logs and the diagnostics table."""
    if not key:
        return "—"
    return f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "short-key"


def _key_state(key: str) -> dict:
    return _GEMINI_KEY_STATE.setdefault(
        key_id(key), {"status": "unknown", "error": None,
                      "cooldown_until": 0.0, "last_ok": None})


def _key_usable(key: str) -> bool:
    st = _key_state(key)
    if st["status"] == "invalid":
        return False
    if st["status"] == "quota" and time.time() < st["cooldown_until"]:
        return False
    return True


def _gemini_rank(name: str) -> tuple:
    """Lower sorts first."""
    low = name.lower()
    demoted = 1 if any(d in low for d in _GEMINI_DEMOTE) else 0
    for i, pref in enumerate(_GEMINI_PREFER):
        if low == pref:
            return (demoted, 0, i, name)
    for i, pref in enumerate(_GEMINI_PREFER):
        if low.startswith(pref):
            return (demoted, 1, i, name)
    return (demoted, 2, 0 if "flash" in low else 1, name)


def discover_gemini_models(force: bool = False, timeout: int = 20,
                           key: str | None = None) -> dict:
    """
    Ask the Gemini API which models a key may call with generateContent.
    Returns {"models": [...], "version": "v1beta"|"v1", "chosen": name,
             "error": str|None, "key_id": str}.
    """
    if key is None:
        keys = gemini_keys()
        key = keys[0] if keys else ""
    if not key:
        return {"models": [], "version": None, "chosen": None,
                "error": "No Gemini API key configured", "key_id": "—"}

    kid = key_id(key)
    cached = _GEMINI_MODELS_BY_KEY.get(kid)
    if cached and cached.get("models") is not None and not force:
        return dict(cached)

    last_error = None
    for version in GEMINI_API_VERSIONS:
        try:
            data = _get(f"{GEMINI_HOST}/{version}/models?pageSize=200",
                        {"x-goog-api-key": key}, timeout)
            names = []
            for m in data.get("models", []):
                methods = m.get("supportedGenerationMethods") or []
                name = str(m.get("name", "")).split("/", 1)[-1]
                if "generateContent" not in methods:
                    continue
                if any(bad in name.lower() for bad in _GEMINI_REJECT):
                    continue
                names.append(name)
            if names:
                names.sort(key=_gemini_rank)
                info = {"models": names, "version": version, "chosen": None,
                        "error": None, "key_id": kid}
                _GEMINI_MODELS_BY_KEY[kid] = info
                return dict(info)
            last_error = f"{version}: key is valid but exposes no generateContent models"
        except urllib.error.HTTPError as e:
            body = e.read()[:180].decode(errors="ignore")
            last_error = f"{version}: HTTP {e.code}: {body}"
            if e.code in (400, 401, 403):
                st = _key_state(key)
                st.update({"status": "invalid", "error": last_error})
        except Exception as e:
            last_error = f"{version}: {type(e).__name__}: {e}"

    info = {"models": [], "version": None, "chosen": None,
            "error": last_error, "key_id": kid}
    _GEMINI_MODELS_BY_KEY[kid] = info
    return dict(info)


def _gemini_plan(key: str) -> tuple[list[str], list[str], str]:
    """(models to try, API versions to try, key id) for one key."""
    kid = key_id(key)
    pinned = os.environ.get("CHAT_GEMINI_MODEL", "").strip()
    if pinned:
        return [pinned], list(GEMINI_API_VERSIONS), kid

    info = discover_gemini_models(key=key)
    discovered = list(info["models"] or [])
    # Merge in the static names: a key can call a model ListModels omits.
    merged = discovered + [m for m in GEMINI_MODELS if m not in discovered]
    merged.sort(key=_gemini_rank)

    chosen = info.get("chosen")
    ordered = ([chosen] if chosen else []) + [m for m in merged if m != chosen]

    primary = info["version"] or GEMINI_API_VERSIONS[0]
    versions = [primary] + [v for v in GEMINI_API_VERSIONS if v != primary]
    return ordered[:8], versions, kid


def _gemini_once(key: str, payload: dict, timeout: int) -> tuple[str | None, str | None, str]:
    """
    Try one key: walk its candidate models across both API versions.
    Returns (text, model, outcome) where outcome is
    "ok" | "quota" | "invalid" | "no_model" | "error".
    """
    models, versions, kid = _gemini_plan(key)
    tried, attempts, MAX_ATTEMPTS = [], 0, 10
    outcome = "no_model"

    for version in versions:
        for model in models:
            if attempts >= MAX_ATTEMPTS:
                break
            pair = f"{kid}|{model}@{version}"
            if pair in _GEMINI_BAD:
                continue                     # already proven 404 — no HTTP call
            attempts += 1
            tried.append(f"{model}@{version}")
            try:
                data = _post(f"{GEMINI_HOST}/{version}/models/{model}:generateContent",
                             payload, {"x-goog-api-key": key}, timeout)
                cand = (data.get("candidates") or [{}])[0]
                parts = (cand.get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts).strip()
                if text:
                    info = _GEMINI_MODELS_BY_KEY.setdefault(
                        kid, {"models": [], "version": version, "chosen": None,
                              "error": None, "key_id": kid})
                    info["chosen"] = model
                    info["version"] = version
                    return text, model, "ok"
                # Empty reply = safety filter or token ceiling, not a bad name.
                _record_error("gemini", f"[{kid}] {model} returned no text "
                                        f"(finishReason={cand.get('finishReason')})")
                outcome = "error"
            except urllib.error.HTTPError as e:
                body = e.read()[:220].decode(errors="ignore")
                if e.code == 404:
                    _GEMINI_BAD.add(pair)
                    continue
                if e.code == 429 or "RESOURCE_EXHAUSTED" in body or "quota" in body.lower():
                    _record_error("gemini", f"[{kid}] quota/rate limit: {body[:160]}")
                    return None, None, "quota"
                if e.code in (400, 401, 403) and (
                        "API_KEY" in body or "api key" in body.lower()
                        or "permission" in body.lower() or "SERVICE_DISABLED" in body):
                    _record_error("gemini", f"[{kid}] key rejected: {body[:160]}")
                    return None, None, "invalid"
                _record_error("gemini", f"[{kid}] HTTP {e.code} ({model}): {body}")
                outcome = "error"
                continue
            except Exception as e:
                _record_error("gemini", f"[{kid}] {type(e).__name__} ({model}): {e}")
                outcome = "error"
                continue

        if version == versions[0] and attempts:
            # Nothing worked on the discovered version — the cached list may
            # predate a key change, so re-ask once before trying the other one.
            discover_gemini_models(force=True, key=key)

    if outcome == "no_model":
        listed = (_GEMINI_MODELS_BY_KEY.get(kid, {}).get("models") or [])[:6]
        _record_error("gemini",
                      f"[{kid}] no callable model — tried {', '.join(tried[:8]) or 'nothing'}. "
                      f"ListModels reports: {', '.join(listed) if listed else 'nothing usable'}. "
                      f"Run  python check_ai.py  for the full list.")
    return None, None, outcome


def _gemini(messages, system, max_tokens, temperature, json_mode, timeout):
    keys = gemini_keys()
    if not keys:
        return None, None

    contents = [{"role": ("model" if m["role"] == "assistant" else "user"),
                 "parts": [{"text": m["content"]}]} for m in messages]
    gen_cfg = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"
    payload = {"contents": contents,
               "systemInstruction": {"parts": [{"text": system}]},
               "generationConfig": gen_cfg}

    skipped = []
    for idx, key in enumerate(keys, start=1):
        if not _key_usable(key):
            skipped.append(f"key {idx} ({_key_state(key)['status']})")
            continue
        text, model, outcome = _gemini_once(key, payload, timeout)
        st = _key_state(key)
        if outcome == "ok":
            st.update({"status": "ok", "error": None, "cooldown_until": 0.0,
                       "last_ok": time.time()})
            _LAST["gemini_key"] = f"{idx}/{len(keys)} {key_id(key)}"
            return text, model
        if outcome == "quota":
            # Rotate to the next key and retry this one after a short rest.
            st.update({"status": "quota", "error": _ERRORS.get("gemini"),
                       "cooldown_until": time.time() + QUOTA_COOLDOWN_SECONDS})
            continue
        if outcome == "invalid":
            st.update({"status": "invalid", "error": _ERRORS.get("gemini")})
            continue
        st.update({"status": "error", "error": _ERRORS.get("gemini")})

    if skipped:
        _record_error("gemini",
                      f"{_ERRORS.get('gemini', 'all keys failed')} "
                      f"(skipped: {', '.join(skipped)})")
    return None, None


def gemini_key_report() -> list[dict]:
    """
    Per-key health, for the AI Settings page and the diagnostics panel.

    Three states matter to a user and they call for different actions, so they
    are reported separately rather than as one "working / not working" flag:
    a usable key needs nothing, a cooling-down key needs only time, and a
    rejected key needs replacing.
    """
    out = []
    now = time.time()
    for idx, key in enumerate(gemini_keys(), start=1):
        st = _key_state(key)
        info = _GEMINI_MODELS_BY_KEY.get(key_id(key), {})
        cooling = max(0, int(st["cooldown_until"] - now))
        invalid = st["status"] == "invalid"

        if invalid:
            note = st["error"] or "Key rejected by the API — replace it."
        elif cooling:
            note = f"Free-tier quota reached; retrying in {cooling}s."
        elif st["status"] == "ok":
            note = (f"Answering with {info.get('chosen')}"
                    if info.get("chosen") else "Answering.")
        elif st["status"] == "error":
            # Reached but failed for a reason that is neither the key's fault
            # nor a quota — a network block, a timeout, a model name that no
            # longer resolves. Worth distinguishing from "never tried", because
            # replacing the key would not help.
            note = (st["error"] or "Call failed for a reason other than the key.")[:160]
        else:
            note = "Not called yet this session."

        out.append({
            "index": idx,
            "id": f"#{idx} {key_id(key)}",
            "key_id": key_id(key),
            "status": st["status"] + (f" ({cooling}s)" if cooling else ""),
            "usable": (not invalid) and cooling == 0 and st["status"] != "error",
            "errored": st["status"] == "error",
            "cooling_down": bool(cooling) and not invalid,
            "cooldown_seconds": cooling,
            "note": note,
            "error": st["error"],
            "model": info.get("chosen"),
            "models_available": len(info.get("models") or []),
        })
    return out


_PROVIDERS = {"gemini": _gemini}


# Public API

def available_providers() -> dict:
    """Kept as a dict so existing call sites and templates need no change."""
    return {"gemini": bool(gemini_keys())}


def auto_order() -> list[str]:
    """
    Retained for callers that still ask which backends will be tried. With one
    provider the answer is either ["gemini"] or [], and the second case is the
    interesting one: it means every screen falls to the deterministic template
    path rather than to a different vendor.
    """
    return ["gemini"] if gemini_keys() else []


def resolve_order(provider: str | None = None) -> list[str]:
    """
    Which backends to try. "template" pins the offline path, which the
    evaluation harness uses to establish what the system produces with no model
    at all; anything else resolves to Gemini when a key exists.
    """
    pref = (provider or os.environ.get("AI_PROVIDER", "auto") or "auto").strip().lower()
    if pref == "template":
        return []
    return auto_order()


def complete(messages: list[dict],
             system: str = "You are a helpful assistant.",
             max_tokens: int = 1200,
             temperature: float = 0.2,
             json_mode: bool = False,
             provider: str | None = None,
             timeout: int = DEFAULT_TIMEOUT) -> tuple[str | None, str | None]:
    """
    Run a chat completion. `messages` is [{"role": "user"|"assistant", "content": str}].
    Returns (text, provider_name); (None, None) when every backend is unavailable,
    which callers must handle by falling back to a deterministic response.
    """
    available = available_providers()
    for name in resolve_order(provider):
        if not available.get(name):
            continue
        if name in _DEAD:
            continue          # key already rejected this session — don't pay the timeout again
        start = time.time()
        text, model = _PROVIDERS[name](messages, system, max_tokens,
                                       temperature, json_mode, timeout)
        if text:
            _LAST.update({"provider": name, "model": model,
                          "latency_ms": int((time.time() - start) * 1000),
                          "error": None})
            return text, name
    return None, None


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict | None:
    """
    Recover a JSON object from model output that may be fenced or narrated.
    Tries, in order: the whole string, any fenced block, then the outermost
    balanced {...} span.
    """
    if not text:
        return None
    for candidate in [text.strip()] + _FENCE_RE.findall(text):
        try:
            obj = json.loads(candidate.strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    start = None
    return None


def complete_json(messages, system, max_tokens=900, temperature=0.0,
                  provider=None, timeout=45) -> tuple[dict | None, str | None]:
    """Chat completion constrained to a JSON object. Returns (obj, provider)."""
    text, prov = complete(messages, system=system, max_tokens=max_tokens,
                          temperature=temperature, json_mode=True,
                          provider=provider, timeout=timeout)
    if not text:
        return None, None
    return extract_json(text), prov


def diagnose(timeout: int = 20) -> dict:
    """
    Ping Gemini with a one-word prompt and report exactly what happened, key by
    key. This is what turns "the AI is offline" — an unhelpful symptom — into
    "key #2 returns HTTP 429: quota exhausted, retry in 90 seconds".

    The return shape keeps the `providers` dict from the multi-provider version
    so existing callers and templates continue to work unchanged.
    """
    reset_failures()                      # a diagnostic run always re-tests
    keys = gemini_keys()
    out = {}

    if not keys:
        out["gemini"] = {"configured": False, "ok": False, "latency_ms": 0,
                         "model": None, "keys": [],
                         "error": "No Gemini API key saved (Settings → AI Settings)"}
    else:
        t0 = time.time()
        text, model = _gemini(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            "You reply with exactly one word.", 12, 0.0, False, timeout)
        out["gemini"] = {
            "configured": True,
            "ok": bool(text),
            "latency_ms": int((time.time() - t0) * 1000),
            "model": model,
            "error": None if text else (_ERRORS.get("gemini")
                                        or "No response (timeout or empty reply)"),
        }

    gem = discover_gemini_models(force=True)
    key_report = gemini_key_report()
    out["gemini"]["available_models"] = gem["models"][:8]
    out["gemini"]["api_version"] = gem["version"]
    out["gemini"]["keys"] = key_report
    if not out["gemini"]["ok"] and gem["error"] and out["gemini"]["configured"]:
        out["gemini"]["error"] = f"{out['gemini']['error']} | discovery: {gem['error']}"

    working = [n for n, v in out.items() if v["ok"]]
    usable = sum(1 for k in key_report if k.get("usable"))
    if working:
        verdict = (f"Answers will be AI-generated. {usable} of {len(key_report)} "
                   f"key(s) currently usable.")
    elif keys:
        verdict = ("No Gemini key is answering, so every screen falls back to the "
                   "deterministic offline template. The records stay complete; the "
                   "prose is plainer.")
    else:
        verdict = ("No key is configured. The system runs entirely on offline "
                   "templates, which is a supported mode rather than a failure.")

    return {
        "providers": out,
        "working": working,
        "active": "gemini" if working else None,
        "preference": (os.environ.get("AI_PROVIDER") or "auto"),
        "order": auto_order(),
        "gemini_models": gem["models"][:8],
        "gemini_keys": key_report,
        "verdict": verdict,
    }


def gemini_generate(prompt: str, system: str, max_tokens: int = 400,
                    temperature: float = 0.4, timeout: int = 30) -> str | None:
    """
    Single-shot Gemini call sharing this module's model discovery, version
    walking and dead-pair memory. Exposed so ai_service.py (care notes, care
    plans, the RAG evaluation) resolves model names the same way the chat does,
    instead of maintaining a second, separately-rotting hard-coded list.
    """
    text, _model = _gemini([{"role": "user", "content": prompt}],
                           system, max_tokens, temperature, False, timeout)
    return text


def last_call() -> dict:
    """Diagnostics for the last completion — provider, model, latency, error."""
    return dict(_LAST)


def status() -> dict:
    av = available_providers()
    order = resolve_order()
    return {"providers": av, "order": order,
            "active": "gemini" if (av["gemini"] and order) else None,
            "models": {"gemini": (discover_gemini_models().get("chosen")
                                  or GEMINI_MODELS[0])},
            "keys": gemini_key_report(),
            "rejected": dict(_DEAD), "errors": dict(_ERRORS),
            "last": dict(_LAST)}
