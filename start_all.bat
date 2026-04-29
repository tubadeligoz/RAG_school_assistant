@echo off
REM Terminal rengini minimalist siyah-beyaz yap
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

REM Yanlis tusa basilirsa basa don
goto menu


:start_system
echo.
echo [+] Ollama motoru arka planda baslatiliyor...
cd /d %~dp0
start cmd /k "ollama serve"

echo [+] Streamlit arayuzu baslatiliyor...
start cmd /k "set PYTHONPATH=src && .venv\Scripts\activate && streamlit run src/app/streamlit_app.py"
goto end


:run_pipeline
echo.
echo [+] Veritabanini guncelleme boru hatti (Pipeline) baslatiliyor...
cd /d %~dp0
REM 🔥 Hata düzeltildi: Dosya yolu "scripts\run_pipeline.py" olarak güncellendi
start cmd /k "set PYTHONPATH=src && .venv\Scripts\activate && python scripts\run_pipeline.py"
goto end


:end
exit