@echo off
REM  setup_ai.bat — save one or more Gemini API keys, then test them.
REM
REM  The app calls Gemini and nothing else. Several keys may be configured;
REM  they are tried in order, and when one hits its free-tier quota the next
REM  takes over. Every key reaches the same models, so which one answers
REM  changes whether the system can reply, never what it replies.
REM
REM  Get free keys at https://aistudio.google.com/apikey — no card needed.
REM  Create each key in a DIFFERENT Google Cloud project: keys inside one
REM  project share a single quota, so four keys from one project multiply
REM  nothing.
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ========================================================================
echo   CareHome Docs - Gemini key setup
echo ========================================================================
echo.
echo  Paste up to 4 keys. Press Enter on an empty line to stop.
echo  Leave the first one blank to skip and run on offline templates.
echo.

set KEYS=
for /L %%i in (1,1,4) do (
    set "K="
    set /p K=  Key %%i (Enter to finish):
    if "!K!"=="" goto :done
    if "!KEYS!"=="" ( set "KEYS=!K!" ) else ( set "KEYS=!KEYS!,!K!" )
)
:done

if "%KEYS%"=="" (
    echo.
    echo  No key entered. The app will use offline templates, which is a
    echo  supported mode: every record is still complete, just plainer.
    echo.
    goto :end
)

REM Persist through ai_config so the running app and future launches agree.
python -c "import ai_config,sys; ks=[k.strip() for k in sys.argv[1].split(',') if k.strip()]; ai_config.save_keys(gemini_keys=ks, provider='auto'); print('  Saved %%d key(s) to ai_config.json' %% len(ks))" "%KEYS%"

echo.
echo  Testing...
echo.
python check_ai.py
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
    echo  Ready. Start the app with run.bat
) else (
    echo  Keys saved but nothing answered. Re-run check_ai.py after checking
    echo  the notes above, or clear them at Settings - AI Settings.
)

:end
echo.
pause
endlocal
