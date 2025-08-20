@echo off
echo Real-time WebRTC Object Detection Benchmark
echo ===========================================

REM Check if Python backend is running
echo Checking backend status...
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Backend is running
) else (
    echo ⚠ Backend not detected. Starting backend...
    cd backend
    start /b python app.py
    timeout /t 5 /nobreak >nul
    cd ..
)

REM Install benchmark dependencies
echo Installing benchmark dependencies...
pip install psutil matplotlib seaborn pandas requests opencv-python

REM Run benchmark
echo Starting benchmark...
python benchmark.py --backends python javascript --duration 30 --output metrics.json

REM Generate analysis
echo Generating analysis...
python analyze_metrics.py --input metrics.json --report benchmark_report.md --charts

echo Benchmark completed!
echo Results saved to:
echo   - metrics.json (raw data)
echo   - benchmark_report.md (analysis report)
echo   - benchmark_charts.png (visualizations)
pause