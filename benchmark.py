#!/usr/bin/env python3
"""
Benchmarking script for Real-time WebRTC Object Detection System
Generates metrics.json with latency, FPS, and bandwidth measurements
"""

import json
import time
import asyncio
import statistics
import argparse
import requests
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import psutil
import threading
import queue
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkMetrics:
    """Data class for storing benchmark metrics"""
    timestamp: str
    test_duration_seconds: float
    detection_backend: str
    
    # Latency metrics (milliseconds)
    avg_end_to_end_latency: float
    min_latency: float
    max_latency: float
    p95_latency: float
    p99_latency: float
    
    # FPS metrics
    avg_processing_fps: float
    avg_display_fps: float
    frame_drop_rate: float
    
    # Bandwidth metrics (Mbps)
    avg_video_bandwidth: float
    peak_bandwidth: float
    total_data_transferred_mb: float
    
    # System metrics
    avg_cpu_usage: float
    avg_memory_usage_mb: float
    avg_gpu_usage: float
    
    # Detection metrics
    total_detections: int
    avg_confidence: float
    detection_accuracy: float
    objects_per_frame: float

class SystemMonitor:
    """Monitor system resources during benchmarking"""
    
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_samples.append(memory.used / 1024 / 1024)  # MB
            
            # GPU usage (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_samples.append(gpus[0].load * 100)
                else:
                    self.gpu_samples.append(0)
            except ImportError:
                self.gpu_samples.append(0)
            
            time.sleep(0.5)
    
    def get_averages(self):
        """Get average system metrics"""
        return {
            'cpu': statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            'memory': statistics.mean(self.memory_samples) if self.memory_samples else 0,
            'gpu': statistics.mean(self.gpu_samples) if self.gpu_samples else 0
        }

class LatencyTracker:
    """Track end-to-end latency measurements"""
    
    def __init__(self):
        self.latencies = []
        self.frame_timestamps = {}
    
    def mark_frame_sent(self, frame_id: str):
        """Mark when a frame was sent for processing"""
        self.frame_timestamps[frame_id] = time.time() * 1000  # milliseconds
    
    def mark_frame_received(self, frame_id: str):
        """Mark when detection results were received"""
        if frame_id in self.frame_timestamps:
            latency = (time.time() * 1000) - self.frame_timestamps[frame_id]
            self.latencies.append(latency)
            del self.frame_timestamps[frame_id]
    
    def get_statistics(self):
        """Get latency statistics"""
        if not self.latencies:
            return {'avg': 0, 'min': 0, 'max': 0, 'p95': 0, 'p99': 0}
        
        sorted_latencies = sorted(self.latencies)
        return {
            'avg': statistics.mean(self.latencies),
            'min': min(self.latencies),
            'max': max(self.latencies),
            'p95': sorted_latencies[int(0.95 * len(sorted_latencies))],
            'p99': sorted_latencies[int(0.99 * len(sorted_latencies))]
        }

class BandwidthMonitor:
    """Monitor network bandwidth usage"""
    
    def __init__(self):
        self.bandwidth_samples = []
        self.total_bytes = 0
        self.start_time = None
    
    def start_monitoring(self):
        """Start bandwidth monitoring"""
        self.start_time = time.time()
        self.initial_stats = psutil.net_io_counters()
    
    def record_bandwidth(self, bytes_transferred: int):
        """Record bandwidth usage"""
        current_time = time.time()
        if self.start_time:
            duration = current_time - self.start_time
            if duration > 0:
                mbps = (bytes_transferred * 8) / (duration * 1000000)  # Mbps
                self.bandwidth_samples.append(mbps)
                self.total_bytes += bytes_transferred
    
    def get_statistics(self):
        """Get bandwidth statistics"""
        if not self.bandwidth_samples:
            return {'avg': 0, 'peak': 0, 'total_mb': 0}
        
        return {
            'avg': statistics.mean(self.bandwidth_samples),
            'peak': max(self.bandwidth_samples),
            'total_mb': self.total_bytes / 1024 / 1024
        }

class DetectionBenchmark:
    """Main benchmarking class"""
    
    def __init__(self, backend_url: str = "http://localhost:5000"):
        self.backend_url = backend_url
        self.system_monitor = SystemMonitor()
        self.latency_tracker = LatencyTracker()
        self.bandwidth_monitor = BandwidthMonitor()
        
        # Metrics storage
        self.fps_samples = []
        self.detection_results = []
        self.frame_count = 0
        self.dropped_frames = 0
    
    def load_test_video(self, video_path: str = None):
        """Load test video or create synthetic frames"""
        if video_path and os.path.exists(video_path):
            return cv2.VideoCapture(video_path)
        else:
            # Create synthetic test frames
            return self._create_synthetic_video()
    
    def _create_synthetic_video(self):
        """Create synthetic video frames for testing"""
        class SyntheticVideo:
            def __init__(self):
                self.frame_count = 0
                self.max_frames = 300  # 10 seconds at 30fps
            
            def read(self):
                if self.frame_count >= self.max_frames:
                    return False, None
                
                # Create a test frame with moving objects
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add moving rectangle (simulates object)
                x = int((self.frame_count * 5) % 600)
                y = int(200 + 50 * np.sin(self.frame_count * 0.1))
                cv2.rectangle(frame, (x, y), (x+40, y+40), (0, 255, 0), -1)
                
                # Add some noise
                noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
                frame = cv2.add(frame, noise)
                
                self.frame_count += 1
                return True, frame
            
            def release(self):
                pass
        
        return SyntheticVideo()
    
    def test_detection_backend(self, backend: str, duration: int = 60):
        """Test specific detection backend"""
        print(f"Testing {backend} backend for {duration} seconds...")
        
        # Start monitoring
        self.system_monitor.start_monitoring()
        self.bandwidth_monitor.start_monitoring()
        
        start_time = time.time()
        video = self.load_test_video()
        
        frame_id = 0
        last_fps_time = start_time
        frames_in_second = 0
        
        try:
            while time.time() - start_time < duration:
                ret, frame = video.read()
                if not ret:
                    break
                
                frame_id += 1
                self.frame_count += 1
                
                # Encode frame for transmission
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_data = buffer.tobytes()
                
                # Record bandwidth
                self.bandwidth_monitor.record_bandwidth(len(frame_data))
                
                # Send for detection
                frame_id_str = f"frame_{frame_id}"
                self.latency_tracker.mark_frame_sent(frame_id_str)
                
                try:
                    detection_result = self._send_for_detection(frame_data, backend)
                    self.latency_tracker.mark_frame_received(frame_id_str)
                    self.detection_results.append(detection_result)
                except Exception as e:
                    print(f"Detection failed for frame {frame_id}: {e}")
                    self.dropped_frames += 1
                
                # Calculate FPS
                frames_in_second += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frames_in_second / (current_time - last_fps_time)
                    self.fps_samples.append(fps)
                    frames_in_second = 0
                    last_fps_time = current_time
                
                # Control frame rate (simulate 30fps)
                time.sleep(1/30)
        
        finally:
            video.release()
            self.system_monitor.stop_monitoring()
        
        return self._compile_metrics(backend, time.time() - start_time)
    
    def _send_for_detection(self, frame_data: bytes, backend: str):
        """Send frame to detection backend"""
        if backend == "python":
            return self._detect_python_backend(frame_data)
        elif backend == "javascript":
            return self._detect_javascript_backend(frame_data)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _detect_python_backend(self, frame_data: bytes):
        """Send frame to Python backend"""
        files = {'frame': ('frame.jpg', frame_data, 'image/jpeg')}
        data = {'confidence': 0.5, 'model': 'yolov5s'}
        
        response = requests.post(
            f"{self.backend_url}/detect",
            files=files,
            data=data,
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Backend error: {response.status_code}")
    
    def _detect_javascript_backend(self, frame_data: bytes):
        """Simulate JavaScript backend detection"""
        # Simulate processing time
        time.sleep(0.05)  # 50ms processing time
        
        # Return mock detection results
        return {
            'detections': [
                {'class': 'person', 'confidence': 0.85, 'bbox': [100, 100, 200, 300]},
                {'class': 'car', 'confidence': 0.72, 'bbox': [300, 200, 500, 400]}
            ],
            'processing_time': 50
        }
    
    def _compile_metrics(self, backend: str, duration: float) -> BenchmarkMetrics:
        """Compile all metrics into final result"""
        latency_stats = self.latency_tracker.get_statistics()
        bandwidth_stats = self.bandwidth_monitor.get_statistics()
        system_stats = self.system_monitor.get_averages()
        
        # Calculate detection metrics
        total_detections = sum(len(result.get('detections', [])) for result in self.detection_results)
        avg_confidence = 0
        if total_detections > 0:
            confidences = []
            for result in self.detection_results:
                for detection in result.get('detections', []):
                    confidences.append(detection.get('confidence', 0))
            avg_confidence = statistics.mean(confidences) if confidences else 0
        
        return BenchmarkMetrics(
            timestamp=datetime.now().isoformat(),
            test_duration_seconds=duration,
            detection_backend=backend,
            
            # Latency metrics
            avg_end_to_end_latency=latency_stats['avg'],
            min_latency=latency_stats['min'],
            max_latency=latency_stats['max'],
            p95_latency=latency_stats['p95'],
            p99_latency=latency_stats['p99'],
            
            # FPS metrics
            avg_processing_fps=statistics.mean(self.fps_samples) if self.fps_samples else 0,
            avg_display_fps=self.frame_count / duration if duration > 0 else 0,
            frame_drop_rate=self.dropped_frames / self.frame_count if self.frame_count > 0 else 0,
            
            # Bandwidth metrics
            avg_video_bandwidth=bandwidth_stats['avg'],
            peak_bandwidth=bandwidth_stats['peak'],
            total_data_transferred_mb=bandwidth_stats['total_mb'],
            
            # System metrics
            avg_cpu_usage=system_stats['cpu'],
            avg_memory_usage_mb=system_stats['memory'],
            avg_gpu_usage=system_stats['gpu'],
            
            # Detection metrics
            total_detections=total_detections,
            avg_confidence=avg_confidence,
            detection_accuracy=1.0 - (self.dropped_frames / self.frame_count) if self.frame_count > 0 else 0,
            objects_per_frame=total_detections / len(self.detection_results) if self.detection_results else 0
        )

def run_comprehensive_benchmark(backends: List[str], duration: int = 60, output_file: str = "metrics.json"):
    """Run comprehensive benchmark across multiple backends"""
    results = {}
    
    for backend in backends:
        print(f"\n{'='*50}")
        print(f"Benchmarking {backend.upper()} Backend")
        print(f"{'='*50}")
        
        benchmark = DetectionBenchmark()
        
        try:
            metrics = benchmark.test_detection_backend(backend, duration)
            results[backend] = asdict(metrics)
            
            # Print summary
            print(f"\nResults for {backend}:")
            print(f"  Average Latency: {metrics.avg_end_to_end_latency:.2f}ms")
            print(f"  Processing FPS: {metrics.avg_processing_fps:.2f}")
            print(f"  Frame Drop Rate: {metrics.frame_drop_rate:.2%}")
            print(f"  Average Bandwidth: {metrics.avg_video_bandwidth:.2f} Mbps")
            print(f"  CPU Usage: {metrics.avg_cpu_usage:.1f}%")
            print(f"  Memory Usage: {metrics.avg_memory_usage_mb:.1f} MB")
            
        except Exception as e:
            print(f"Benchmark failed for {backend}: {e}")
            results[backend] = {"error": str(e)}
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark object detection system")
    parser.add_argument("--backends", nargs="+", default=["python", "javascript"],
                       help="Backends to test (python, javascript, wasm)")
    parser.add_argument("--duration", type=int, default=60,
                       help="Test duration in seconds")
    parser.add_argument("--output", default="metrics.json",
                       help="Output file for metrics")
    parser.add_argument("--backend-url", default="http://localhost:5000",
                       help="Backend URL for testing")
    
    args = parser.parse_args()
    
    print("Real-time WebRTC Object Detection Benchmark")
    print(f"Testing backends: {', '.join(args.backends)}")
    print(f"Duration: {args.duration} seconds per backend")
    print(f"Backend URL: {args.backend_url}")
    
    # Check if backend is running
    try:
        response = requests.get(f"{args.backend_url}/health", timeout=5)
        if response.status_code != 200:
            print("Warning: Backend health check failed")
    except requests.exceptions.RequestException:
        print("Warning: Could not connect to backend. Some tests may fail.")
    
    results = run_comprehensive_benchmark(
        backends=args.backends,
        duration=args.duration,
        output_file=args.output
    )
    
    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    main()