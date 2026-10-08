"""
ai_config.py — Persistent Gemini key storage for CareDocs AI.

Keys are saved to ai_config.json alongside app.py. On import, saved keys are
loaded into os.environ so ai_service.py and llm_client.py pick them up
transparently — those modules only ever read environment variables and do not
know or care whether a key arrived from run.bat or from the AI Settings page.

Precedence: an environment variable set BEFORE the app starts (i.e. by
run.bat) wins, because _load() uses setdefault. Saving through the UI overrides
it for the running process. This is deliberate — a key baked into the launch
script should not be silently replaced by a stale saved one, but a user
actively typing a new key clearly means it.

Several keys, one provider
Gemini free-tier keys carry small per-minute and per-day quotas, and a full
evaluation run makes hundreds of calls. Up to MAX_GEMINI_KEYS may therefore be
stored and are tried in order; llm_client puts a quota-exhausted key into a
short cooldown and moves to the next.

This is quota management, not model selection. Every key reaches the same
models, so which key answered cannot change what the system said — only
whether it could answer at all. That distinction is why the earlier
multi-vendor arrangement was removed and this one kept: failing over between
Anthropic, Google and Groq silently changed which model wrote a clinical
record, and left nothing in the record to say so.

`gemini_key` remains in the file as the first key, so a configuration written
by an earlier version still loads.

SECURITY NOTE: ai_config.json stores keys in plaintext. It is git-ignored, but
clear the keys (AI Settings → Clear all keys) before submitting or sharing
this project. A production system would use a secrets manager or OS keyring.
"""
import os
import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "ai_config.json"

# config-file field  ->  environment variable
KEY_FIELDS = {
    "gemini_key": "GEMINI_API_KEY",
}

# "auto" and "gemini" are the same thing now and both are accepted, because a
# saved config from an earlier version may hold either. "template" pins the
# offline path, which the evaluation harness uses to measure the floor.
VALID_PROVIDERS = ("auto", "gemini", "template")

# Gemini free-tier keys have small per-minute and per-day quotas, so several
# can be stored and rotated. gemini_key stays the primary (first) one for
# backwards compatibility; gemini_keys holds the full ordered list.
MAX_GEMINI_KEYS = 4
GEMINI_KEYS_ENV = "GEMINI_API_KEYS"


def normalise_gemini_keys(keys) -> list[str]:
    """Trim, drop blanks and duplicates, cap at MAX_GEMINI_KEYS, keep order."""
    if isinstance(keys, str):
        keys = keys.split(",")
    out = []
    for k in (keys or []):
        k = str(k).strip()
        if k and k not in out:
            out.append(k)
    return out[:MAX_GEMINI_KEYS]

# One backend. These are kept so that call sites, saved configuration files and
# the diagnostics endpoint written against the multi-provider version keep
# working without a migration step.
REAL_PROVIDERS = ("gemini",)
DEFAULT_ORDER = ["gemini"]
ORDER_ENV = "AI_PROVIDER_ORDER"


def normalise_order(order=None) -> list[str]:
    """
    There is one provider, so the order is always ["gemini"]. The function is
    retained rather than deleted because an ai_config.json written by an
    earlier version stores a three-element order, and silently normalising it
    is friendlier than refusing to load the file.
    """
    return list(DEFAULT_ORDER)


def get_provider_order() -> list[str]:
    """The live provider order. One element, by construction."""
    return list(DEFAULT_ORDER)


def _load():
    """Load saved keys, provider preference and order into env at startup."""
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            for field, env_var in KEY_FIELDS.items():
                if data.get(field):
                    os.environ.setdefault(env_var, data[field])
            if data.get("provider") in VALID_PROVIDERS:
                os.environ.setdefault("AI_PROVIDER", data["provider"])
            if data.get("provider_order"):
                os.environ.setdefault(ORDER_ENV,
                                      ",".join(normalise_order(data["provider_order"])))
            gkeys = normalise_gemini_keys(
                data.get("gemini_keys") or ([data["gemini_key"]] if data.get("gemini_key") else []))
            if gkeys:
                os.environ.setdefault(GEMINI_KEYS_ENV, ",".join(gkeys))
                os.environ.setdefault("GEMINI_API_KEY", gkeys[0])
        except Exception:
            pass


def save_keys(gemini_keys=None, gemini_key: str = "", provider: str = "auto",
              provider_order=None, **_ignored):
    """
    Persist the Gemini keys plus the provider preference, and update the live
    process immediately so no restart is needed.

    `gemini_keys` (a list) is authoritative; `gemini_key` is accepted as a
    one-element convenience. Any other keyword — including the `anthropic_key`
    and `groq_key` arguments the multi-provider version took — is swallowed by
    **_ignored, so an older call site fails quietly rather than raising, and
    the removed keys simply stop being written.
    """
    gkeys = normalise_gemini_keys(
        gemini_keys if gemini_keys is not None else [gemini_key])

    values = {"gemini_key": (gkeys[0] if gkeys else "")}
    provider = (provider or "auto").strip().lower()
    if provider not in VALID_PROVIDERS:
        provider = "auto"
    order = normalise_order()

    CONFIG_FILE.write_text(json.dumps(
        {**values, "gemini_keys": gkeys,
         "provider": provider, "provider_order": order}, indent=2))

    # Apply immediately to the running process.
    for field, env_var in KEY_FIELDS.items():
        if values[field]:
            os.environ[env_var] = values[field]
        elif env_var in os.environ:
            del os.environ[env_var]

    if gkeys:
        os.environ[GEMINI_KEYS_ENV] = ",".join(gkeys)
    elif GEMINI_KEYS_ENV in os.environ:
        del os.environ[GEMINI_KEYS_ENV]

    os.environ["AI_PROVIDER"] = provider
    os.environ[ORDER_ENV] = ",".join(order)

    # Forget any "this key was rejected" state recorded against the old keys,
    # otherwise a corrected key keeps being skipped until the app restarts.
    try:
        import llm_client
        llm_client.reset_failures()
    except Exception:
        pass


def get_stored_keys() -> dict:
    """Return the saved keys (unmasked), the provider preference and the order."""
    blank = {field: "" for field in KEY_FIELDS}
    blank["provider"] = "auto"
    blank["provider_order"] = get_provider_order()
    blank["gemini_keys"] = []
    if not CONFIG_FILE.exists():
        return blank
    try:
        data = json.loads(CONFIG_FILE.read_text())
        out = {field: data.get(field, "") for field in KEY_FIELDS}
        prov = data.get("provider", "auto")
        out["provider"] = prov if prov in VALID_PROVIDERS else "auto"
        out["provider_order"] = normalise_order(
            data.get("provider_order") or get_provider_order())
        out["gemini_keys"] = normalise_gemini_keys(
            data.get("gemini_keys") or ([out["gemini_key"]] if out["gemini_key"] else []))
        return out
    except Exception:
        return blank


def mask(key: str) -> str:
    """
    Masked display, e.g. AIzaSyDx…4f7B·39 — enough to tell two keys apart.

    Gemini keys share a long fixed prefix, so masking the first eight and last
    four characters produces collisions more often than intuition suggests: two
    keys differing only in the middle mask identically, and the settings page
    then cannot tell which slot a returned mask belongs to. The length suffix
    breaks most of those ties cheaply, and the caller resolves the remainder by
    slot position.
    """
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return f"{key[:8]}...{key[-4:]}\u00b7{len(key)}"


# Load on import
_load()
