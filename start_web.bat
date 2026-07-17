@echo off
echo ========================================
echo Malaria Prediction Web App
echo ========================================
echo.
echo This will start the web server and open the app in your browser.
echo.
echo Make sure you have installed the requirements:
echo   pip install flask flask-cors torch pandas numpy
echo.

set PYTHON=C:\Users\User\AppData\Local\Programs\Python\Python314\python.exe

echo Starting server at http://127.0.0.1:5000 ...
echo.
start "" "%PYTHON%" app.py

timeout /t 3 /nobreak >nul

echo Opening browser...
start http://127.0.0.1:5000

echo.
echo If browser doesn't open, manually go to: http://127.0.0.1:5000
echo.
pause