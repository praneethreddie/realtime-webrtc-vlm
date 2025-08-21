# Object Detection Benchmark Report

Generated: 2025-08-21 09:10:09

## Executive Summary

## Performance Comparison

| Backend | Avg Latency (ms) | Processing FPS | Frame Drop Rate | Bandwidth (Mbps) | CPU Usage (%) |
|---------|------------------|----------------|-----------------|------------------|---------------|
| Python | 0.00 | 0.48 | 100.00% | 0.63 | 36.0 |
| Javascript | 51.11 | 10.15 | 0.00% | 0.16 | 15.6 |

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
- CPU Usage: 36.0%
- Memory Usage: 14093.5 MB
- GPU Usage: 0.0%

#### Detection Quality
- Total Detections: 0
- Average Confidence: 0.00
- Detection Accuracy: 0.00%
- Objects per Frame: 0.00

### Javascript Backend

#### Latency Metrics
- Average: 51.11ms
- Minimum: 50.04ms
- Maximum: 62.26ms
- 95th Percentile: 52.82ms
- 99th Percentile: 58.55ms

#### Throughput Metrics
- Processing FPS: 10.15
- Display FPS: 9.97
- Frame Drop Rate: 0.00%

#### Resource Usage
- CPU Usage: 15.6%
- Memory Usage: 12846.8 MB
- GPU Usage: 0.0%

#### Detection Quality
- Total Detections: 600
- Average Confidence: 0.78
- Detection Accuracy: 100.00%
- Objects per Frame: 2.00

## Recommendations

- **Lowest Latency**: Python (0.00ms)
- **Highest Throughput**: Javascript (10.15 FPS)
- **Most Efficient**: Javascript (15.6% CPU)

### Use Case Recommendations

- **Real-time Applications**: Choose the backend with lowest latency
- **High-volume Processing**: Choose the backend with highest FPS
- **Resource-constrained Environments**: Choose the most CPU-efficient backend
- **Mobile/Edge Deployment**: Consider JavaScript/WASM for client-side processing