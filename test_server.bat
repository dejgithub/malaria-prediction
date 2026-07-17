@echo off
echo Testing Flask Server...
echo.

set PYTHON=C:\Users\User\AppData\Local\Programs\Python\Python314\python.exe

REM Start server in background
start /b "" "%PYTHON%" app.py > server.log 2>&1

REM Wait for server to start
timeout /t 3 /nobreak >nul

REM Test endpoints
echo Testing / endpoint...
curl -s http://127.0.0.1:5000/ > nul
if %errorlevel% neq 0 (
    echo FAILED: Cannot connect to http://127.0.0.1:5000
    echo Check server.log for errors
    type server.log
) else (
    echo SUCCESS: Server is running!
    echo.
    echo Open http://127.0.0.1:5000 in your browser
)

REM Kill the server
taskkill /f /im python.exe > nul 2>&1

pause