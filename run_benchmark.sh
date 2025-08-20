#!/bin/bash

# Quick benchmark runner script

echo "Real-time WebRTC Object Detection Benchmark"
echo "==========================================="

# Check if Python backend is running
echo "Checking backend status..."
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✓ Backend is running"
else
    echo "⚠ Backend not detected. Starting backend..."
    cd backend
    python app.py &
    BACKEND_PID=$!
    sleep 5
    cd ..
fi

# Install benchmark dependencies
echo "Installing benchmark dependencies..."
pip install psutil matplotlib seaborn pandas requests opencv-python

# Run benchmark
echo "Starting benchmark..."
python benchmark.py --backends python javascript --duration 30 --output metrics.json

# Generate analysis
echo "Generating analysis..."
python analyze_metrics.py --input metrics.json --report benchmark_report.md --charts

# Cleanup
if [ ! -z "$BACKEND_PID" ]; then
    echo "Stopping backend..."
    kill $BACKEND_PID
fi

echo "Benchmark completed!"
echo "Results saved to:"
echo "  - metrics.json (raw data)"
echo "  - benchmark_report.md (analysis report)"
echo "  - benchmark_charts.png (visualizations)"