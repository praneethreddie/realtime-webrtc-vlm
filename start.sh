#!/bin/bash

# Real-time WebRTC Object Detection Startup Script

set -e

# Default values
MODE="python"
USE_TUNNEL=false
DEV_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --tunnel)
            USE_TUNNEL=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --mode MODE     Detection backend mode (python|javascript|wasm)"
            echo "  --tunnel        Enable Cloudflare tunnel for external access"
            echo "  --dev          Enable development mode with hot reload"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Real-time WebRTC Object Detection System"
echo "======================================="
echo "Mode: $MODE"
echo "Tunnel: $USE_TUNNEL"
echo "Development: $DEV_MODE"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    echo "Please install Docker and try again"
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "Error: docker-compose is not available"
    echo "Please install docker-compose and try again"
    exit 1
fi

# Create frontend directory if it doesn't exist
if [ ! -d "frontend" ]; then
    echo "Creating frontend directory..."
    mkdir -p frontend
    
    # Move HTML files if they exist in root
    if [ -f "index.html" ]; then
        mv index.html frontend/
        echo "Moved index.html to frontend/"
    fi
    
    if [ -f "mobile.html" ]; then
        mv mobile.html frontend/
        echo "Moved mobile.html to frontend/"
    fi
fi

# Stop any existing containers
echo "Stopping existing containers..."
$COMPOSE_CMD down --remove-orphans

# Build and start services
echo "Building and starting services..."

if [ "$USE_TUNNEL" = true ]; then
    if [ -z "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
        echo "Warning: CLOUDFLARE_TUNNEL_TOKEN not set. Tunnel service will not start."
        echo "Set the environment variable and restart to enable tunnel."
    fi
    $COMPOSE_CMD --profile tunnel up --build -d
else
    $COMPOSE_CMD up --build -d
fi

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Health check
echo "Performing health checks..."

# Check backend
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✓ Backend is healthy"
else
    echo "⚠ Backend health check failed"
fi

# Check frontend
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✓ Frontend is healthy"
else
    echo "⚠ Frontend health check failed"
fi

echo ""
echo "🚀 System is ready!"
echo ""
echo "Access URLs:"
echo "  Desktop Browser: http://localhost:8080"
echo "  Mobile Interface: http://localhost:8080/mobile.html"
echo "  Backend API: http://localhost:5000"

if [ "$USE_TUNNEL" = true ] && [ ! -z "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
    echo "  External Access: Check Cloudflare dashboard for tunnel URL"
fi

echo ""
echo "To stop the system: $COMPOSE_CMD down"
echo "To view logs: $COMPOSE_CMD logs -f"
echo "To restart: $0 $*"