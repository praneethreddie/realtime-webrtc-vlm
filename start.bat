@echo off
setlocal enabledelayedexpansion

echo 🚀 Starting WebRTC Object Detection Demo
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python first.
    pause
    exit /b 1
)

echo 📦 Setting up backend dependencies...
cd backend

REM Install Python dependencies
echo Installing Python packages...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install Python dependencies
    pause
    exit /b 1
)

REM Create models directory if it doesn't exist
if not exist "models" mkdir models

REM Download models if they don't exist
echo 🤖 Checking and downloading ML models...
if not exist "models\yolov5s.onnx" (
    echo Downloading required models...
    python download_models.py
) else (
    echo ✅ Models already downloaded
)

REM Start backend server
echo 🔧 Starting Flask backend server...
start "Backend Server" python app.py

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Go back to project root
cd ..

REM Start frontend server
echo 🌐 Starting frontend server...
cd frontend

start "Frontend Server" python -m http.server 8080

REM Wait for frontend to start
timeout /t 2 /nobreak >nul

REM Check if cloudflared is installed and start tunnel
echo ☁️ Setting up Cloudflare tunnel...
cloudflared version >nul 2>&1
if errorlevel 1 (
    echo 📥 Cloudflared not found. Please install from:
    echo https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
    echo.
    echo ⚠️ Continuing without tunnel - app accessible locally only
) else (
    echo 🌐 Creating Cloudflare tunnel...
    start "Cloudflare Tunnel" cloudflared tunnel --url http://localhost:8080
    timeout /t 3 /nobreak >nul
    echo ✅ Cloudflare tunnel started!
)

echo.
echo ================================================
echo 🎉 Demo is now running!
echo.
echo 📱 Local Access:
echo    Main App: http://localhost:8080
echo    Mobile Camera: http://localhost:8080/mobile.html
echo    Backend API: http://localhost:5000
echo.
echo 🌍 Public Access:
echo    Check the Cloudflare tunnel URL in the tunnel window
echo.
echo 📋 Instructions:
echo    1. Open the main app in your browser
echo    2. Scan the QR code with your mobile device
echo    3. Allow camera access on mobile
echo    4. Enjoy real-time object detection!
echo.
echo 🛑 Close all terminal windows to stop the demo
echo ================================================

REM Open the app in default browser
start http://localhost:8080

pause