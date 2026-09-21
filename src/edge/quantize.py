"""
INT8 Quantization Engine for Edge AI Deployment.

Uses ONNX Runtime quantization utilities to convert FP32 ONNX models into INT8 models,
enabling CPU latency speedups and disk/RAM size reductions.
"""

import os
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx_model(fp32_model_path: str, int8_output_path: str):
    """
    Applies dynamic INT8 quantization to an ONNX FP32 model file.
    """
    os.makedirs(os.path.dirname(int8_output_path), exist_ok=True)
    if not os.path.exists(fp32_model_path):
        raise FileNotFoundError(f"FP32 model not found at: {fp32_model_path}")

    print(f"[INT8 Quantization] Quantizing {fp32_model_path} -> {int8_output_path}...")

    try:
        quantize_dynamic(
            model_input=fp32_model_path,
            model_output=int8_output_path,
            weight_type=QuantType.QUInt8
        )
    except Exception as e:
        print(f"[INT8 Quantization Warning] Standard dynamic quantization encountered: {e}. Falling back to MatMul quantization...")
        quantize_dynamic(
            model_input=fp32_model_path,
            model_output=int8_output_path,
            op_types_to_quantize=['MatMul', 'Gemm'],
            weight_type=QuantType.QUInt8
        )

    fp32_size_mb = os.path.getsize(fp32_model_path) / (1024 * 1024)
    int8_size_mb = os.path.getsize(int8_output_path) / (1024 * 1024)
    reduction = (1 - int8_size_mb / fp32_size_mb) * 100

    print(f"[INT8 Quantization] Completed!")
    print(f"  - FP32 Disk Size: {fp32_size_mb:.2f} MB")
    print(f"  - INT8 Disk Size: {int8_size_mb:.2f} MB")
    print(f"  - Disk Size Reduction: {reduction:.1f}%")

    return int8_output_path, fp32_size_mb, int8_size_mb


def main():
    # Quantize ConvLSTM
    if os.path.exists('models/convlstm.onnx'):
        quantize_onnx_model('models/convlstm.onnx', 'models/convlstm_int8.onnx')

    # Quantize ST-GNN
    if os.path.exists('models/st_gnn.onnx'):
        quantize_onnx_model('models/st_gnn.onnx', 'models/st_gnn_int8.onnx')


if __name__ == "__main__":
    main()
