# Object Detection Benchmark Report

Generated: 2025-08-21 07:49:57

## Executive Summary

## Performance Comparison

| Backend | Avg Latency (ms) | Processing FPS | Frame Drop Rate | Bandwidth (Mbps) | CPU Usage (%) |
|---------|------------------|----------------|-----------------|------------------|---------------|
| Python | 0.00 | 0.48 | 100.00% | 1.41 | 23.2 |
| Javascript | 53.77 | 10.21 | 0.00% | 0.20 | 12.9 |

## Detailed Metrics

### Python Backend

#### Latency Metrics
- Average: 0.00ms
- Minimum: 0.00ms
- Maximum: 0.00ms
- 95th Percentile: 0.00ms
- 99th Percentile: 0.00ms

#### Throughput Metrics
- Processing FPS: 0.48
- Display FPS: 0.47
- Frame Drop Rate: 100.00%

#### Resource Usage
- CPU Usage: 23.2%
- Memory Usage: 13188.4 MB
- GPU Usage: 0.0%

#### Detection Quality
- Total Detections: 0
- Average Confidence: 0.00
- Detection Accuracy: 0.00%
- Objects per Frame: 0.00

### Javascript Backend

#### Latency Metrics
- Average: 53.77ms
- Minimum: 50.16ms
- Maximum: 72.81ms
- 95th Percentile: 58.87ms
- 99th Percentile: 60.38ms

#### Throughput Metrics
- Processing FPS: 10.21
- Display FPS: 10.16
- Frame Drop Rate: 0.00%

#### Resource Usage
- CPU Usage: 12.9%
- Memory Usage: 13314.9 MB
- GPU Usage: 0.0%

#### Detection Quality
- Total Detections: 600
- Average Confidence: 0.78
- Detection Accuracy: 100.00%
- Objects per Frame: 2.00

## Recommendations

- **Lowest Latency**: Python (0.00ms)
- **Highest Throughput**: Javascript (10.21 FPS)
- **Most Efficient**: Javascript (12.9% CPU)

### Use Case Recommendations

- **Real-time Applications**: Choose the backend with lowest latency
- **High-volume Processing**: Choose the backend with highest FPS
- **Resource-constrained Environments**: Choose the most CPU-efficient backend
- **Mobile/Edge Deployment**: Consider JavaScript/WASM for client-side processing