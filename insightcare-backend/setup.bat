@echo off
echo ========================================
echo InsightCare Backend - Quick Setup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate

echo [3/5] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [4/5] Setting up environment file...
if not exist .env (
    copy .env.example .env
    echo Created .env file - PLEASE EDIT IT with your database credentials!
) else (
    echo .env file already exists, skipping...
)

echo [5/5] Setup complete!
echo.
echo ========================================
echo IMPORTANT: Next Steps
echo ========================================
echo.
echo 1. Edit .env file with your PostgreSQL credentials
echo 2. Ensure PostgreSQL is running
echo 3. Create database: psql -U postgres -c "CREATE DATABASE insightcare;"
echo 4. Run the server: uvicorn app.main:app --reload
echo.
echo API will be available at: http://localhost:8000
echo Interactive docs at: http://localhost:8000/docs
echo.
echo ========================================
pause
