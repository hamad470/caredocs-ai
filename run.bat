@echo off
REM  run.bat — start CareHome Docs.
REM
REM  AI KEYS (optional)
REM  The app calls Gemini and nothing else. Without a key it still runs: every
REM  narrative field falls back to a deterministic offline template, so records
REM  stay complete and nothing leaves the machine.
REM
REM  To supply keys here rather than through Settings - AI Settings, uncomment
REM  the line below and paste them comma-separated. Keys set here take
REM  precedence over saved ones at startup.
REM
REM  Several keys are tried in order; when one hits its free-tier quota the
REM  next takes over. Create each key in a DIFFERENT Google Cloud project —
REM  keys inside one project share a single quota.
REM
REM      Free keys: https://aistudio.google.com/apikey
REM
REM  set GEMINI_API_KEYS=AIza_first_key,AIza_second_key,AIza_third_key
REM
REM  Pin one model name instead of letting the client discover it:
REM  set CHAT_GEMINI_MODEL=gemini-2.5-flash
REM
REM  Force the offline template path (the evaluation control arm):
REM  set AI_PROVIDER=template
setlocal
cd /d "%~dp0"

echo.
echo ========================================================================
echo   CareHome Docs
echo ========================================================================
echo.

REM ── Python present? ───────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  Python was not found on the PATH.
    echo  Install Python 3.11 or newer from https://www.python.org/downloads/
    echo  and tick "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM ── Dependencies ──────────────────────────────────────────────────────────
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo  Installing dependencies ^(first run only^)...
    python -m pip install --quiet -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  Dependency installation failed. Try:  python -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    echo  Done.
    echo.
)

REM ── Database ──────────────────────────────────────────────────────────────
if not exist carehome.db (
    echo  No database found. Generating the synthetic cohort ^(about 2 minutes^)...
    python synthetic_cohort.py --yes --validate
    if errorlevel 1 (
        echo.
        echo  Cohort generation failed. See the message above.
        echo.
        pause
        exit /b 1
    )
    echo.
)

REM ── AI backend status, reported but never fatal ───────────────────────────
python check_ai.py --quiet
if errorlevel 2 (
    echo  AI: no Gemini key configured - running on offline templates.
    echo      Add keys at Settings - AI Settings, or run setup_ai.bat.
) else if errorlevel 1 (
    echo  AI: keys are configured but none answered. Run check_ai.py to see why.
    echo      The app will use offline templates until one works.
) else (
    echo  AI: Gemini reachable.
)

echo.
echo  Starting the server. Open http://127.0.0.1:5000 in a browser.
echo  Sign in as   manager1 / manager123
echo  Press Ctrl+C to stop.
echo.

python app.py

endlocal
