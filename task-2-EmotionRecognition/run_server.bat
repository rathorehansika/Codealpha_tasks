@echo off
REM Start the NEUROVOX emotion-recognition backend using the project venv.
cd /d "%~dp0"
echo Starting NEUROVOX backend on http://localhost:8000 ...
".\venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
