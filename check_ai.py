"""
check_ai.py — Is the AI backend reachable, and if not, exactly why?

Run this before a demonstration or a long evaluation. It pings Gemini on every
configured key and prints what happened key by key, so "the AI is offline"
turns into something actionable: a rejected key to replace, a quota to wait
out, or no key at all.

    python check_ai.py            # test every configured key
    python check_ai.py --quiet    # exit code only, for scripts

Exit codes:
    0  at least one key answered
    1  keys are configured but none answered
    2  no key is configured (the app still runs on offline templates)
"""
import argparse
import sys

import ai_config          # noqa: F401  — loads saved keys into the environment
import llm_client


BAR = "─" * 72


def main() -> int:
    ap = argparse.ArgumentParser(description="Check the Gemini backend.")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress output; return an exit code only")
    ap.add_argument("--timeout", type=int, default=20)
    args = ap.parse_args()

    report = llm_client.diagnose(timeout=args.timeout)
    gem = report["providers"]["gemini"]
    keys = report["gemini_keys"]

    def say(*a):
        if not args.quiet:
            print(*a)

    say(BAR)
    say(" CareHome Docs — AI backend check")
    say(BAR)

    if not gem["configured"]:
        say(" No Gemini API key is configured.")
        say("")
        say(" This is not a broken state. Every narrative field falls back to a")
        say(" deterministic offline template, so records stay complete — the prose")
        say(" is simply plainer, and nothing is sent to any third party.")
        say("")
        say(" To enable generation: Settings → AI Settings, or set GEMINI_API_KEYS")
        say(" in run.bat. Free keys: https://aistudio.google.com/apikey")
        say(BAR)
        return 2

    say(f" Keys configured : {len(keys)}")
    say(f" Model in use    : {gem.get('model') or '(none answered)'}")
    say(f" API version     : {gem.get('api_version') or '—'}")
    say(f" Round trip      : {gem['latency_ms']} ms")
    say("")

    for k in keys:
        if k["status"].startswith("ok"):
            mark = "  OK   "
        elif k["cooling_down"]:
            mark = " QUOTA "
        elif k.get("errored"):
            mark = " ERROR "
        elif not k["usable"]:
            mark = " DEAD  "
        else:
            mark = " IDLE  "
        say(f" [{mark}] {k['id']:<24} {k['note']}")

    say("")
    if gem["ok"]:
        say(" Result: working.")
    else:
        say(f" Result: NOT working — {gem['error']}")
        say("")
        say(" Common causes:")
        say("   • Key copied with a trailing space, or from the wrong project.")
        say("   • All keys share one Google Cloud project, so they share one")
        say("     quota. Create each key in a separate project to multiply it.")
        say("   • Free-tier daily limit reached; it resets at midnight Pacific.")

    models = report.get("gemini_models") or []
    if models and not args.quiet:
        say("")
        say(" Models this key can call: " + ", ".join(models[:6]))

    say(BAR)
    return 0 if gem["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
