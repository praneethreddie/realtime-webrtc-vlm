#!/usr/bin/env python3
"""
Metrics analysis script for benchmark results
Generates reports and visualizations from metrics.json
"""

import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns
from typing import Dict, Any

def load_metrics(file_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_comparison_report(metrics: Dict[str, Any], output_file: str = "benchmark_report.md"):
    """Generate markdown comparison report"""
    
    report = []
    report.append("# Object Detection Benchmark Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Executive Summary")
    
    # Find best performing backend for each metric
    backends = [k for k in metrics.keys() if 'error' not in metrics[k]]
    
    if not backends:
        report.append("\nNo successful benchmark results found.")
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        return
    
    # Performance comparison table
    report.append("\n## Performance Comparison")
    report.append("\n| Backend | Avg Latency (ms) | Processing FPS | Frame Drop Rate | Bandwidth (Mbps) | CPU Usage (%) |")
    report.append("|---------|------------------|----------------|-----------------|------------------|---------------|")
    
    for backend in backends:
        data = metrics[backend]
        report.append(f"| {backend.title()} | {data['avg_end_to_end_latency']:.2f} | {data['avg_processing_fps']:.2f} | {data['frame_drop_rate']:.2%} | {data['avg_video_bandwidth']:.2f} | {data['avg_cpu_usage']:.1f} |")
    
    # Detailed metrics
    report.append("\n## Detailed Metrics")
    
    for backend in backends:
        data = metrics[backend]
        report.append(f"\n### {backend.title()} Backend")
        
        report.append("\n#### Latency Metrics")
        report.append(f"- Average: {data['avg_end_to_end_latency']:.2f}ms")
        report.append(f"- Minimum: {data['min_latency']:.2f}ms")
        report.append(f"- Maximum: {data['max_latency']:.2f}ms")
        report.append(f"- 95th Percentile: {data['p95_latency']:.2f}ms")
        report.append(f"- 99th Percentile: {data['p99_latency']:.2f}ms")
        
        report.append("\n#### Throughput Metrics")
        report.append(f"- Processing FPS: {data['avg_processing_fps']:.2f}")
        report.append(f"- Display FPS: {data['avg_display_fps']:.2f}")
        report.append(f"- Frame Drop Rate: {data['frame_drop_rate']:.2%}")
        
        report.append("\n#### Resource Usage")
        report.append(f"- CPU Usage: {data['avg_cpu_usage']:.1f}%")
        report.append(f"- Memory Usage: {data['avg_memory_usage_mb']:.1f} MB")
        report.append(f"- GPU Usage: {data['avg_gpu_usage']:.1f}%")
        
        report.append("\n#### Detection Quality")
        report.append(f"- Total Detections: {data['total_detections']}")
        report.append(f"- Average Confidence: {data['avg_confidence']:.2f}")
        report.append(f"- Detection Accuracy: {data['detection_accuracy']:.2%}")
        report.append(f"- Objects per Frame: {data['objects_per_frame']:.2f}")
    
    # Recommendations
    report.append("\n## Recommendations")
    
    # Find best backend for different use cases
    latency_winner = min(backends, key=lambda b: metrics[b]['avg_end_to_end_latency'])
    fps_winner = max(backends, key=lambda b: metrics[b]['avg_processing_fps'])
    efficiency_winner = min(backends, key=lambda b: metrics[b]['avg_cpu_usage'])
    
    report.append(f"\n- **Lowest Latency**: {latency_winner.title()} ({metrics[latency_winner]['avg_end_to_end_latency']:.2f}ms)")
    report.append(f"- **Highest Throughput**: {fps_winner.title()} ({metrics[fps_winner]['avg_processing_fps']:.2f} FPS)")
    report.append(f"- **Most Efficient**: {efficiency_winner.title()} ({metrics[efficiency_winner]['avg_cpu_usage']:.1f}% CPU)")
    
    # Use case recommendations
    report.append("\n### Use Case Recommendations")
    report.append("\n- **Real-time Applications**: Choose the backend with lowest latency")
    report.append("- **High-volume Processing**: Choose the backend with highest FPS")
    report.append("- **Resource-constrained Environments**: Choose the most CPU-efficient backend")
    report.append("- **Mobile/Edge Deployment**: Consider JavaScript/WASM for client-side processing")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Benchmark report saved to {output_file}")

def create_visualizations(metrics: Dict[str, Any], output_dir: str = "."):
    """Create visualization charts"""
    
    backends = [k for k in metrics.keys() if 'error' not in metrics[k]]
    if not backends:
        print("No data available for visualization")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Object Detection Benchmark Results', fontsize=16)
    
    # Latency comparison
    latencies = [metrics[b]['avg_end_to_end_latency'] for b in backends]
    axes[0, 0].bar(backends, latencies)
    axes[0, 0].set_title('Average End-to-End Latency')
    axes[0, 0].set_ylabel('Latency (ms)')
    
    # FPS comparison
    fps_values = [metrics[b]['avg_processing_fps'] for b in backends]
    axes[0, 1].bar(backends, fps_values)
    axes[0, 1].set_title('Processing FPS')
    axes[0, 1].set_ylabel('Frames per Second')
    
    # Resource usage
    cpu_usage = [metrics[b]['avg_cpu_usage'] for b in backends]
    memory_usage = [metrics[b]['avg_memory_usage_mb'] for b in backends]
    
    x = range(len(backends))
    width = 0.35
    axes[1, 0].bar([i - width/2 for i in x], cpu_usage, width, label='CPU %')
    axes[1, 0].bar([i + width/2 for i in x], [m/10 for m in memory_usage], width, label='Memory (MB/10)')
    axes[1, 0].set_title('Resource Usage')
    axes[1, 0].set_ylabel('Usage')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(backends)
    axes[1, 0].legend()
    
    # Detection quality
    accuracy = [metrics[b]['detection_accuracy'] * 100 for b in backends]
    confidence = [metrics[b]['avg_confidence'] * 100 for b in backends]
    
    axes[1, 1].bar([i - width/2 for i in x], accuracy, width, label='Accuracy %')
    axes[1, 1].bar([i + width/2 for i in x], confidence, width, label='Avg Confidence %')
    axes[1, 1].set_title('Detection Quality')
    axes[1, 1].set_ylabel('Percentage')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(backends)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_charts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_dir}/benchmark_charts.png")

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark metrics")
    parser.add_argument("--input", default="metrics.json", help="Input metrics file")
    parser.add_argument("--report", default="benchmark_report.md", help="Output report file")
    parser.add_argument("--charts", action="store_true", help="Generate visualization charts")
    
    args = parser.parse_args()
    
    try:
        metrics = load_metrics(args.input)
        
        # Generate report
        generate_comparison_report(metrics, args.report)
        
        # Generate charts if requested
        if args.charts:
            create_visualizations(metrics)
        
        print("Analysis completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Metrics file '{args.input}' not found.")
        print("Run the benchmark script first to generate metrics.")
    except Exception as e:
        print(f"Error analyzing metrics: {e}")

if __name__ == "__main__":
    main()