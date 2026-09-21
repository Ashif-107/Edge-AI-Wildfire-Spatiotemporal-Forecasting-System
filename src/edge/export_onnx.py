"""
PyTorch to ONNX Exporter.

Exports trained ConvLSTM2D and ST-GNN PyTorch models into portable ONNX FP32 format
with dynamic batch dimension support.
"""

import os
import sys
import torch

# Fix Windows console UTF-8 output encoding for PyTorch ONNX printouts
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.models.convlstm import ConvLSTM2D
from src.models.st_gnn import SpatiotemporalGNN


def export_model_to_onnx(model, dummy_input: torch.Tensor, onnx_output_path: str):
    """
    Exports a PyTorch model to ONNX format.
    """
    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_datacube'],
        output_names=['predicted_risk_maps'],
        dynamic_axes={
            'input_datacube': {0: 'batch_size'},
            'predicted_risk_maps': {0: 'batch_size'}
        },
        dynamo=False
    )
    print(f"[ONNX Export] Successfully exported FP32 ONNX model to: {onnx_output_path}")
    return onnx_output_path


def main():
    dummy_input = torch.randn(1, 5, 7, 128, 128)

    # 1. Export ConvLSTM2D
    convlstm = ConvLSTM2D(in_channels=7, hidden_dims=[32, 16], out_horizons=3)
    if os.path.exists('checkpoints/convlstm_best.pt'):
        convlstm.load_state_dict(torch.load('checkpoints/convlstm_best.pt', map_location='cpu'))
    export_model_to_onnx(convlstm, dummy_input, 'models/convlstm.onnx')

    # 2. Export ST-GNN
    st_gnn = SpatiotemporalGNN(in_channels=7, patch_size=4, hidden_dim=32, out_horizons=3)
    if os.path.exists('checkpoints/st_gnn_best.pt'):
        st_gnn.load_state_dict(torch.load('checkpoints/st_gnn_best.pt', map_location='cpu'))
    export_model_to_onnx(st_gnn, dummy_input, 'models/st_gnn.onnx')


if __name__ == "__main__":
    main()
