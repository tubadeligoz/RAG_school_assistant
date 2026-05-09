@echo off
cd /d "%~dp0"

set LOG_FILE=%~dp0streamlit_tailscale.log

echo [+] DORA Streamlit Tailscale modunda baslatiliyor...
echo [+] Yerel adres: http://localhost:8501
echo [+] Tailscale adresi icin: tailscale ip -4

echo [%date% %time%] DORA Streamlit baslatiliyor... > "%LOG_FILE%"
call ".venv\Scripts\activate.bat" >> "%LOG_FILE%" 2>&1
where streamlit >> "%LOG_FILE%" 2>&1
streamlit run src/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true >> "%LOG_FILE%" 2>&1
