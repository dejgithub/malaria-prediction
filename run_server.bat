@echo off
echo ========================================
echo Malaria Prediction Server
echo ========================================
echo.

set PYTHON=C:\Users\User\AppData\Local\Programs\Python\Python314\python.exe

REM Change to project directory
cd /d "C:\Users\User\Videos\Malaria"

echo Starting Flask server at http://127.0.0.1:5000
echo Open this URL in your browser
echo.

"%PYTHON%" app.py