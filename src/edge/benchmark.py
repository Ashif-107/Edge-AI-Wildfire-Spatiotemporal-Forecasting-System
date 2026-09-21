"""
Edge-AI Efficiency Benchmarking Harness.

Compares Base FP32 ONNX model vs Quantized INT8 ONNX model across:
  1. Disk Storage Size (MB)
  2. CPU Latency per Frame (ms)
  3. Peak RAM Footprint (MB)
  4. Accuracy Retention (IoU & Delta IoU)
"""

import os
import time
import json
import tracemalloc
import numpy as np
import onnxruntime as ort
from src.data.dataset_generator import WildfireDataCubeGenerator
from src.evaluation.metrics import compute_iou


def benchmark_onnx_model(model_path: str, test_inputs: np.ndarray, test_targets: np.ndarray, warmup_runs: int = 5, num_runs: int = 20):
    """
    Runs empirical benchmarks on an ONNX model file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    disk_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Initialize ONNX Runtime Session (CPU Provider)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Warmup
    dummy = test_inputs[:1]
    for _ in range(warmup_runs):
        session.run([output_name], {input_name: dummy})

    # Measure RAM Footprint & Latency
    tracemalloc.start()
    latencies = []
    predictions = []

    for i in range(min(num_runs, len(test_inputs))):
        sample_in = test_inputs[i:i+1]
        t0 = time.perf_counter()
        out = session.run([output_name], {input_name: sample_in})[0]
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)  # Convert to ms
        predictions.append(out[0])

    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_ram_mb = peak_ram / (1024 * 1024)
    avg_latency_ms = float(np.mean(latencies))

    predictions = np.array(predictions)
    targets_subset = test_targets[:len(predictions)]
    iou = compute_iou(predictions, targets_subset, threshold=0.35)

    return {
        'disk_size_mb': round(disk_size_mb, 2),
        'avg_latency_ms': round(avg_latency_ms, 2),
        'peak_ram_mb': round(peak_ram_mb, 2),
        'iou': round(iou, 4)
    }


def run_full_benchmark(model_name: str = "convlstm"):
    fp32_path = f"models/{model_name}.onnx"
    int8_path = f"models/{model_name}_int8.onnx"

    print(f"\n=======================================================")
    print(f"       EDGE-AI EFFICIENCY BENCHMARK: {model_name.upper()}")
    print(f"=======================================================")

    generator = WildfireDataCubeGenerator(seed=123)
    X, Y = generator.create_dataset_samples(sequence_length=5, num_days=25)
    X_np = X.numpy()
    Y_np = Y.numpy()

    fp32_metrics = benchmark_onnx_model(fp32_path, X_np, Y_np)
    int8_metrics = benchmark_onnx_model(int8_path, X_np, Y_np)

    speedup = fp32_metrics['avg_latency_ms'] / max(int8_metrics['avg_latency_ms'], 1e-5)
    size_reduction = (1.0 - int8_metrics['disk_size_mb'] / fp32_metrics['disk_size_mb']) * 100
    iou_retention = (int8_metrics['iou'] / max(fp32_metrics['iou'], 1e-5)) * 100

    report = {
        'model_name': model_name,
        'FP32_Base': fp32_metrics,
        'INT8_Quantized': int8_metrics,
        'Comparison': {
            'Latency_Speedup': f"{speedup:.2f}x",
            'Disk_Size_Reduction': f"{size_reduction:.1f}%",
            'IoU_Accuracy_Retention': f"{iou_retention:.1f}%"
        }
    }

    print("\n--- Benchmark Summary Table ---")
    print(f"Metric                   Base FP32 Model    Quantized INT8    Gain / Improvement")
    print(f"--------------------------------------------------------------------------------")
    print(f"Model Disk Size (MB)     {fp32_metrics['disk_size_mb']:<18} {int8_metrics['disk_size_mb']:<17} {size_reduction:.1f}% smaller")
    print(f"CPU Latency per Frame    {fp32_metrics['avg_latency_ms']} ms             {int8_metrics['avg_latency_ms']} ms            {speedup:.2f}x faster")
    print(f"Peak RAM Footprint (MB)  {fp32_metrics['peak_ram_mb']:<18} {int8_metrics['peak_ram_mb']:<17} Edge ready")
    print(f"Prediction Accuracy IoU  {fp32_metrics['iou']:<18} {int8_metrics['iou']:<17} {iou_retention:.1f}% retained")
    print(f"--------------------------------------------------------------------------------")

    os.makedirs('models', exist_ok=True)
    with open(f"models/{model_name}_benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    if os.path.exists("models/convlstm.onnx") and os.path.exists("models/convlstm_int8.onnx"):
        run_full_benchmark("convlstm")
    if os.path.exists("models/st_gnn.onnx") and os.path.exists("models/st_gnn_int8.onnx"):
        run_full_benchmark("st_gnn")
