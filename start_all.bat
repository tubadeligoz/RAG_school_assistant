@echo off
color 0F
title AI Agent Orchestra - Control Panel

:menu
cls
echo ===================================================
echo             AI AGENT ORCHESTRA CONTROL             
echo ===================================================
echo.
echo [1] Asistani Baslat (Ollama + Streamlit UI)
echo [2] Bilgi Bankasini Guncelle (RAG Pipeline Run)
echo [3] Cikis
echo.
echo ===================================================

set /p choice="Islem numarasini secin (1/2/3): "

if "%choice%"=="1" goto start_system
if "%choice%"=="2" goto run_pipeline
if "%choice%"=="3" goto end
goto menu

:start_system
echo.
cd /d %~dp0

echo [+] Ollama kontrol ediliyor...
curl http://localhost:11434 >nul 2>&1
if errorlevel 1 (
    echo [+] Ollama baslatiliyor...
    start cmd /k "ollama serve"
    timeout /t 3 >nul
)

echo [+] Streamlit arayuzu baslatiliyor...
start cmd /k ".venv\Scripts\activate && python -m streamlit run src/app/streamlit_app.py --server.port 8501 --server.headless true"

goto end

:run_pipeline
echo.
cd /d %~dp0

echo [+] Pipeline baslatiliyor...
start cmd /k ".venv\Scripts\activate && python scripts\run_pipeline.py"

goto end

:end
exit
