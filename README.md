# Edge-AI Wildfire Spatiotemporal Forecasting & Risk Mapping System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8-00599C.svg)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![UN SDGs](https://img.shields.io/badge/UN_SDG-13_%7C_15-green.svg)](https://sdgs.un.org/goals)

A Spatiotemporal Deep Learning Framework for Proactive Wildfire Risk Forecasting (24h, 48h, 72h lead time) with Edge-Deployable ONNX INT8 Quantized Inference and an Interactive Decision-Support Dashboard.

---

## 🌟 Key Highlights & Architecture

1. **Multi-Modal Spatiotemporal Data Fusion**: Fuses 5-day rolling sequence data cubes `[Batch, T=5, C=7, H=128, W=128]` combining active fire labels, wind vectors ($u, v$), relative humidity, surface temperature, NDVI fuel density, and slope topography.
2. **Dual Model Benchmark**:
   - **Primary Model**: Stacked `ConvLSTM2D` capturing spatial topology and temporal weather state recurrence.
   - **Comparison Model**: `Spatiotemporal Graph Neural Network (ST-GNN)` modeling directional wind-weighted graph edge propagation.
3. **Severe Class Imbalance Loss**: Custom `CombinedFocalDiceLoss` ($L = \alpha L_{\text{Focal}} + \beta L_{\text{Dice}}$) weighting rare active fire pixels (<1%).
4. **Edge-AI Optimization**: PyTorch FP32 $\rightarrow$ ONNX INT8 quantization via `onnxruntime.quantization` with benchmarked size reduction and latency speedup.
5. **Interactive Streamlit Dashboard**: Side-by-side ground truth vs predicted hazard maps, live environmental perturbation sliders (wind, temperature, humidity), architecture toggle, edge efficiency cards, and per-horizon confusion matrices.

---

## 📂 Repository Structure

```
wildfire/
├── app/
│   └── main.py                  # Interactive Streamlit Decision-Support Dashboard
├── src/
│   ├── data/
│   │   ├── dataset_generator.py  # Realistic 5-day spatiotemporal data cube & label synthesizer
│   │   ├── wildfire_dataset.py   # PyTorch Dataset & DataLoader with chronological split
│   │   └── gee_pipeline.py       # Production Google Earth Engine raw data ingestion reference
│   ├── models/
│   │   ├── convlstm.py           # Stacked ConvLSTM2D network architecture
│   │   ├── st_gnn.py             # Spatiotemporal Graph Neural Network (ST-GNN) architecture
│   │   └── losses.py             # Focal Loss + Dice Loss for extreme class imbalance
│   ├── edge/
│   │   ├── export_onnx.py        # FP32 model to ONNX export
│   │   ├── quantize.py           # INT8 ONNX Quantization engine
│   │   └── benchmark.py          # Latency, disk size, RAM, and IoU benchmark suite
│   └── evaluation/
│       └── metrics.py            # PR-AUC, IoU, and per-horizon confusion matrix calculators
├── checkpoints/                  # Trained PyTorch state dicts (.pt)
├── models/                       # Exported FP32 and INT8 ONNX models (.onnx)
├── train.py                      # Main model training script
├── requirements.txt              # Project dependencies
└── README.md                     # System documentation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Train Models (ConvLSTM & ST-GNN)
```powershell
.\venv\Scripts\python.exe train.py
```

### 3. Quantize to INT8 ONNX & Benchmark
```powershell
.\venv\Scripts\python.exe -m src.edge.export_onnx
.\venv\Scripts\python.exe -m src.edge.quantize
.\venv\Scripts\python.exe -m src.edge.benchmark
```

### 4. Launch Interactive Streamlit Dashboard
```powershell
.\venv\Scripts\streamlit.exe run app/main.py
```

---

## 🌍 Alignment with UN Sustainable Development Goals

- **SDG 13 (Climate Action)**: Provides proactive 24–72 hour early warning risk forecasting to enable disaster-risk reduction and prevent massive greenhouse gas releases from forest megafires.
- **SDG 15 (Life on Land)**: Protects forest ecosystems, biodiversity, and prevents post-fire soil degradation by enabling preemptive firefighting resource positioning.
