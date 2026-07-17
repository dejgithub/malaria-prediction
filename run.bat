@echo off
REM Malaria Prediction System - Run Script

echo ========================================
echo Malaria Disease Prediction System
echo ========================================
echo.

REM Set Python path
set PYTHON=C:\Users\User\AppData\Local\Programs\Python\Python314\python.exe

REM Check if Python is installed
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Python found: 
"%PYTHON%" --version
echo.

REM Install dependencies
echo Installing dependencies...
"%PYTHON%" -m pip install numpy pandas scikit-learn matplotlib torch --quiet

echo.
echo ========================================
echo Step 1: Preprocess data
echo ========================================
"%PYTHON%" src\data_preprocessing.py

if errorlevel 1 (
    echo ERROR: Data preprocessing failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 2: Train model
echo ========================================
"%PYTHON%" src\train_model.py

if errorlevel 1 (
    echo ERROR: Model training failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 3: Generate predictions
echo ========================================
"%PYTHON%" src\make_predictions.py

if errorlevel 1 (
    echo ERROR: Prediction generation failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo DONE!
echo ========================================
echo Open dashboard\index.html in your browser to view results
echo.
pause