@echo off
echo ========================================
echo Starting Malaria Prediction Server
echo ========================================
echo.

REM Set Python path - adjust if your Python is in different location
set PYTHON=C:\Users\User\AppData\Local\Programs\Python\Python314\python.exe

echo Starting Flask server at http://127.0.0.1:5000
echo Open this URL in your browser to use the app
echo.

"%PYTHON%" app.py

pause