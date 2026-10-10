@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [start.bat] No .venv found. Create one with: python -m venv .venv
    echo [start.bat] Then: pip install -r requirements-dev.txt
    exit /b 1
)

echo [start.bat] This project runs one local server: the Streamlit app.
echo [start.bat] remote_inference/server.py (GPU inference) is a separate, optional service that runs in
echo [start.bat] a Colab/Kaggle notebook, not locally - see docs/REMOTE_API.md. It is not started here.
echo.

rem Launch Streamlit in its own window so its logs stay visible and it keeps running independently of
rem this script (.streamlit/config.toml sets headless=true, so Streamlit will not open a browser itself).
rem Bound explicitly to 127.0.0.1 (not "localhost"): some browsers auto-upgrade http://localhost to
rem https:// ("Always use secure connections"), and Streamlit's dev server only speaks plain HTTP, so that
rem upgrade fails with a connection error. 127.0.0.1 does not get the same auto-upgrade treatment.
start "RoboTrace Server" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && streamlit run app.py --server.address=127.0.0.1"

echo [start.bat] Waiting for the server at http://127.0.0.1:8501 ...
powershell -NoProfile -Command ^
    "$deadline = (Get-Date).AddSeconds(60); " ^
    "while ((Get-Date) -lt $deadline) { " ^
    "  try { " ^
    "    $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8501/_stcore/health' -UseBasicParsing -TimeoutSec 2; " ^
    "    if ($r.StatusCode -eq 200) { exit 0 } " ^
    "  } catch {} " ^
    "  Start-Sleep -Milliseconds 500 " ^
    "} " ^
    "exit 1"

if errorlevel 1 (
    echo [start.bat] Server did not respond within 60s. Check the "RoboTrace Server" window for errors.
    echo [start.bat] Opening the browser anyway - refresh the page once the server finishes starting.
) else (
    echo [start.bat] Server is up.
)

start "" "http://127.0.0.1:8501"

endlocal
