@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [start.bat] No .venv found. Create one with: python -m venv .venv
    echo [start.bat] Then: pip install -r requirements-dev.txt
    exit /b 1
)

streamlit run app.py

endlocal
