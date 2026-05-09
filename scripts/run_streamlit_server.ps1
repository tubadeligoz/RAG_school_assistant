Set-Location "C:\Users\tubad\Desktop\RAG_school_assistant"
& ".\.venv\Scripts\python.exe" -m streamlit run "src/app/streamlit_app.py" --server.port=8501 --server.address=0.0.0.0 --server.headless=true
