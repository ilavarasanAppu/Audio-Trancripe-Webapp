@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo    VaniScript AI - Indic Realtime Dubbing & Audio RAG Studio
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/4] Running System Doctor & Environment Diagnostics...
python backend/doctor.py
if %ERRORLEVEL% NEQ 0 (
    echo [!] Doctor found missing dependencies. Attempting auto-fix...
    python -c "from backend.doctor import auto_fix_missing; auto_fix_missing()"
)

echo [2/4] Checking Frontend dependencies...
if not exist "node_modules" (
    echo Installing node dependencies...
    call npm install
)

echo [3/4] Starting Python AI Backend Server on port 8000...
start "VaniScript Backend Core" /B cmd /c "python backend/server.py > backend/server.log 2>&1"

echo [4/4] Starting Vite UI Studio on port 5173...
start "VaniScript Frontend Studio" /B cmd /c "npm run dev > frontend.log 2>&1"

timeout /t 3 >nul

echo.
echo ======================================================================
echo    VaniScript AI is RUNNING!
echo    URL: http://localhost:5173
echo ======================================================================
echo Opening your default browser...
start http://localhost:5173

echo.
echo Press any key to keep this window open or close it (servers run in background).
pause
