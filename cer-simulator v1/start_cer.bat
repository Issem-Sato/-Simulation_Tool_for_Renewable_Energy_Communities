@echo off
REM =========================
REM  Start CER Simulator (Windows .bat)
REM  - Activates conda env
REM  - Sets PYTHONPATH so 'cer_core' is importable
REM  - Runs Streamlit app
REM =========================

setlocal ENABLEDELAYEDEXPANSION

REM --- move to repo root (this .bat location) ---
cd /d "%~dp0"

REM --- configuration: conda env name ---
if "%CER_ENV%"=="" set CER_ENV=cer-gpu

echo.
echo [CER] Using conda env: %CER_ENV%
echo [CER] Repo root: %CD%

REM --- try to activate conda ---
set "_CONDA_ACT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if exist "%_CONDA_ACT%" (
    call "%_CONDA_ACT%" "%CER_ENV%"
) else (
    set "_CONDA_ACT=%USERPROFILE%\miniconda3\Scripts\activate.bat"
    if exist "%_CONDA_ACT%" (
        call "%_CONDA_ACT%" "%CER_ENV%"
    ) else (
        REM fallback (if 'conda' is on PATH)
        call conda activate "%CER_ENV%"
    )
)

REM --- add src to PYTHONPATH (so 'from cer_core import ...' works) ---
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
echo [CER] PYTHONPATH=%PYTHONPATH%

REM --- quick sanity check ---
python -c "import cer_core, sys; print('[CER] cer_core OK - sys.path entries:', len(sys.path))" 1>nul 2>nul
if errorlevel 1 (
    echo [CER][WARN] 'cer_core' non importabile. Continuo lo stesso...
)

REM --- launch Streamlit ---
echo.
echo [CER] Avvio Streamlit...
streamlit run src\cer_app\app.py

endlocal
