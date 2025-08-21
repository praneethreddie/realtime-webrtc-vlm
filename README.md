# Real-time WebRTC Object Detection

A real-time object detection system that streams video from your phone to a browser and overlays detection results using WebRTC technology.

## DEMO

https://www.loom.com/share/229129e104304297b7e1c12c9b277f70?sid=2474581e-5724-4160-a604-d04be4db8810

## Features

- **Real-time Video Streaming**: Stream video from your phone camera to any browser
- **Multiple Detection Backends**: Choose between Python (ONNX/YOLO) or JavaScript (TensorFlow.js) processing
- **WebRTC Technology**: Low-latency peer-to-peer video streaming
- **Containerized Deployment**: Docker support for easy deployment
- **Real-time Metrics**: Monitor latency, FPS, and bandwidth
- **Mobile Optimized**: Dedicated mobile interface for phone cameras

## Quick Start

```bash
run start.bat file in windows platform.
```
it runs every thing use cloudflare tunneled url to acces the output

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

┌─────────────┐    WebRTC     ┌─────────────┐
│   Mobile    │◄─────────────►│   Browser   │
│   Camera    │               │  (Desktop)  │
└─────────────┘               └─────────────┘
│
Socket.IO/HTTP
│
┌─────────────┐
│   Python    │
│   Backend   │
│ (Detection) │
└─────────────┘
│
RESTful API
│
┌─────────────┐
│   Flask     │
│   App       │
└─────────────┘
│
WebSocket
│
│   SocketIO  │



## Troubleshooting

### Common Issues

1. **Camera not working on mobile**
   - Ensure HTTPS connection (required for camera access)
   - Check camera permissions in browser settings

2. **High latency**
   - Use Python backend for faster processing
   - Reduce video resolution on mobile
   - Check network connection quality

3. **Detection accuracy issues**
   - Adjust confidence threshold
   - Ensure good lighting conditions
   - Try different detection models

### Performance Optimization

- **Low-resource mode**: Automatically reduces processing when system resources are limited
- **Backpressure handling**: Skips frames when processing can't keep up
- **Model selection**: Choose appropriate model based on hardware capabilities

## Development

### Project Structure

realtime-webrtc-vlm/
├── frontend/
│   ├── index.html          # Desktop browser interface
│   └── mobile.html         # Mobile camera interface
├── backend/
│   ├── app.py              # Flask backend
│   ├── requirements.txt    # Python dependencies
│   ├── download_models.py  # Model download script
│   └── models/            # Pre-trained models
├── Dockerfile.backend      # Backend container
├── Dockerfile.frontend     # Frontend container
├── docker-compose.yml      # Container orchestration
├── nginx.conf             # Nginx configuration
├── start.sh               # Linux/Mac startup script
├── start.bat              # Windows startup script
├── benchmark.py           # Benchmarking script
├── analyze_metrics.py     # Metrics analysis
└── README.md              # Documentation


### Adding New Models

1. Add model download logic to `backend/download_models.py`
2. Implement detection function in `backend/app.py`
3. Update model selection in frontend interface

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- TensorFlow.js team for client-side ML capabilities
- ONNX Runtime for optimized model inference
- PeerJS for simplified WebRTC implementation
- Ultralytics for YOLO model implementations
