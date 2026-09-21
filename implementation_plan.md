# Implementation Plan - Edge-AI Wildfire Spatiotemporal Forecasting & Risk Mapping System

A comprehensive spatiotemporal deep learning framework for proactive wildfire risk forecasting (24h, 48h, 72h lead time) with edge-deployable ONNX INT8 quantization and an interactive decision-support dashboard.

## User Review Required

> [!IMPORTANT]
> - **Data Source**: We provide both a **Google Earth Engine (GEE) script** for fetching real satellite/weather datasets (NASA FIRMS, ERA5-Land, MODIS NDVI, USGS DEM) and a **realistic Synthetic Data Cube Generator**. The synthetic generator allows full offline execution, training, and immediate demonstration without requiring GEE authentication tokens.
> - **Architectures**: We will implement the primary **Stacked ConvLSTM2D** model and a **Spatiotemporal Graph Neural Network (ST-GNN)** benchmark model with wind-directional edge weighting.

## Proposed Steps & Architecture

We will implement the system step-by-step in logical modules:

---

### Step 1: Project Setup & Data Pipeline
- `requirements.txt`: Python packages (`torch`, `onnx`, `onnxruntime`, `streamlit`, `plotly`, `folium`, `streamlit-folium`, `scikit-learn`, `numpy`, `scipy`).
- [NEW] `src/data/dataset_generator.py`: Generates realistic 5-day spatiotemporal data cubes (`[B, T=5, C=7, H=128, W=128]`) with channels: `[fire_mask, wind_u, wind_v, humidity, temperature, NDVI, slope]`, and ground truth labels for 24h, 48h, 72h horizons.
- [NEW] `src/data/wildfire_dataset.py`: PyTorch `Dataset` and `DataLoader` with chronological train/val/test splits (eliminating temporal leakage).
- [NEW] `src/data/gee_pipeline.py`: Production reference script for Google Earth Engine raw data acquisition.

---

### Step 2: Spatiotemporal Models & Focal/Dice Loss
- [NEW] `src/models/convlstm.py`: Stacked `ConvLSTM2D` architecture with 2D spatial convolutions and temporal LSTM cells, outputting multi-horizon hazard risk grids.
- [NEW] `src/models/st_gnn.py`: Spatiotemporal Graph Neural Network representing grid spatial neighborhoods as graph nodes with wind-vector directional edge weights.
- [NEW] `src/models/losses.py`: Custom `FocalDiceLoss` addressing extreme pixel class imbalance (<1% fire pixels).

---

### Step 3: Evaluation Metrics & Training Pipeline
- [NEW] `src/evaluation/metrics.py`: PR-AUC, Intersection over Union (IoU/Jaccard), and per-horizon confusion matrix calculators.
- [NEW] `train.py`: Training script for ConvLSTM and ST-GNN models with checkpointing and loss/metric tracking.

---

### Step 4: Edge-AI Optimization & Quantization
- [NEW] `src/edge/export_onnx.py`: PyTorch model exporter to ONNX FP32 format.
- [NEW] `src/edge/quantize.py`: INT8 Dynamic/Static quantization engine via `onnxruntime.quantization`.
- [NEW] `src/edge/benchmark.py`: Benchmarking suite comparing FP32 vs INT8 in disk size (MB), CPU latency (ms), RAM footprint (MB), and IoU accuracy retention.

---

### Step 5: Interactive Streamlit Decision-Support Dashboard
- [NEW] `app/main.py`: Interactive Streamlit application featuring:
  1. **Side-by-Side Geo-Map Visualization**: Historical actual fire vs model predicted hazard grid overlay.
  2. **Horizon Time-Slider**: Interactive 24h, 48h, and 72h forecast horizon selection.
  3. **Live Parameter Perturbation Sliders**: Adjust wind speed, wind direction, temperature, and humidity in real-time to observe dynamic expansion/contraction of high-risk fire zones.
  4. **Model Architecture Toggle**: Instant comparison between ConvLSTM and ST-GNN risk maps.
  5. **Edge Efficiency Gauge Cards**: Live performance stats showing FP32 vs INT8 speedups (>3.5x), size reduction (~75%), and memory footprint.
  6. **Evaluation Dashboard & Export**: Precision-Recall metrics, Confusion Matrix, and PNG/GeoTIFF download options.

---

## Verification Plan

### Automated Verification
1. **Model Pipeline & Tensor Tests**: Verify input tensor shape `[Batch, 5, 7, 128, 128]` matches output shape `[Batch, 3, 128, 128]`.
2. **Loss Function Test**: Verify `FocalDiceLoss` converges on imbalanced synthetic grids.
3. **Training & Checkpoint Verification**: Run 5 epochs of training to produce trained FP32 weights for ConvLSTM and ST-GNN.
4. **ONNX Export & Quantization Test**: Run ONNX conversion and INT8 quantization, verifying output `.onnx` models run without error.
5. **Benchmark Verification**: Execute `src/edge/benchmark.py` to produce FP32 vs INT8 performance comparison metrics.

### Manual Verification
1. Launch Streamlit application (`streamlit run app/main.py`).
2. Verify interactive map, parameter sliders, model toggle, and edge efficiency gauge cards.
