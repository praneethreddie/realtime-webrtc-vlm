# Real-time WebRTC Object Detection

A real-time object detection system that streams video from your phone to a browser and overlays detection results using WebRTC technology.

## Features

- **Real-time Video Streaming**: Stream video from your phone camera to any browser
- **Multiple Detection Backends**: Choose between Python (ONNX/YOLO) or JavaScript (TensorFlow.js) processing
- **WebRTC Technology**: Low-latency peer-to-peer video streaming
- **Containerized Deployment**: Docker support for easy deployment
- **Real-time Metrics**: Monitor latency, FPS, and bandwidth
- **Mobile Optimized**: Dedicated mobile interface for phone cameras

## Quick Start

### One-Command Start (Docker)

```bash
# Start with default Python backend
./start.sh

# Start with specific mode
./start.sh --mode python    # Python + ONNX/YOLO backend
./start.sh --mode js        # JavaScript + TensorFlow.js backend
./start.sh --mode wasm      # WASM client-side detection

# Start with Cloudflare tunnel (for external access)
./start.sh --tunnel
```

### Manual Setup (Development)

1. **Install Dependencies**
   ```bash
   # Backend dependencies
   cd backend
   pip install -r requirements.txt
   python download_models.py
   
   # Start backend
   python app.py
   ```

2. **Serve Frontend**
   ```bash
   # Simple HTTP server
   python -m http.server 8080
   # Or use any web server to serve index.html
   ```

## Usage

### Desktop Browser (Receiver)

1. Open your browser and navigate to `http://localhost:8080`
2. You'll see a QR code for connection
3. Select your preferred detection backend:
   - **Python + ONNX**: Server-side processing with ONNX models
   - **Python + YOLO**: Server-side processing with YOLO models
   - **JavaScript + TensorFlow.js**: Client-side processing
   - **WASM**: WebAssembly-based client-side detection
4. Adjust confidence threshold as needed

### Mobile Phone (Sender)

1. Scan the QR code with your phone
2. Allow camera permissions
3. Your phone camera will stream to the desktop browser
4. Object detection results will overlay on the video in real-time

## Detection Modes

### Python Backend Modes

- **ONNX Models**: Optimized for CPU inference
  - EfficientDet D0
  - SSD MobileNet V2
- **YOLO Models**: High accuracy detection
  - YOLOv5s
  - YOLOv5nu

### Client-side Modes

- **TensorFlow.js**: COCO-SSD model running in browser
- **WASM**: WebAssembly-optimized detection (coming soon)

## Configuration

### Environment Variables

```bash
# Backend configuration
PORT=5000                    # Backend server port
MODEL_PATH=./models         # Path to model files
DEFAULT_CONFIDENCE=0.5      # Default confidence threshold

# Frontend configuration
FRONTEND_PORT=8080          # Frontend server port
BACKEND_URL=localhost:5000  # Backend URL for API calls
```

### Docker Configuration

The `docker-compose.yml` supports multiple deployment scenarios:

```yaml
# Standard deployment
docker-compose up

# With Cloudflare tunnel
docker-compose --profile tunnel up

# Development mode with hot reload
docker-compose -f docker-compose.dev.yml up
```

## Performance Metrics

The system automatically collects and displays:

- **End-to-end Latency**: Time from frame capture to detection display
- **Processing FPS**: Frames processed per second
- **Network Bandwidth**: Video stream bandwidth usage
- **Detection Accuracy**: Confidence scores and object counts

## Architecture