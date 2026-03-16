# AirPen

**AirPen** is an end-to-end embedded machine learning system that recognizes handwritten letters traced in the air. A custom IMU-equipped pen captures motion, streams it over serial to a host, and a trained 1D ResNet classifies the gesture in real time.

---

## Demo

Write a letter in the air → see the prediction:

```
[INFERENCE] Waiting for gesture...
[SAMPLE]    500 samples received
[RESULT]    Predicted: A  (confidence: 97.3%)
```

Supported letters: **A, C, E, I, L, N, O, R, S, T**

---

## How It Works

```
Wio Terminal + MPU6050
        │  USB serial @ 115200 baud
        │  CSV: ax, ay, az, gx, gy, gz @ ~100 Hz
        ▼
  Signal Processing
  ├── Butterworth lowpass filter (10 Hz cutoff)
  ├── PCA transform on accelerometer axes
  ├── Align to first-movement direction
  └── Gravity component extraction
        │
        ▼
   ResNet1D (PyTorch)
   Input: (3, 498) — 3 PCA channels, ~5 seconds
        │
        ▼
  Predicted Letter + Confidence
```

### Signal Processing Pipeline

Raw 6-DOF IMU data is transformed before feeding the model:

1. **Lowpass filtering** — Butterworth filter removes high-frequency noise
2. **PCA transform** — Rotates 3D accelerometer readings into principal component coordinates, with handedness enforcement via determinant check
3. **First-movement alignment** — Rotates the PCA plane so the first principal axis aligns with initial pen motion, making predictions orientation-invariant
4. **Gravity estimation** — Extracts and removes the gravity component via a separate lowpass filter

### Model Architecture

A **1D ResNet** adapted for time-series classification:

- **Stem**: 7×1 convolution with stride 2 + MaxPool → 125 features
- **4 residual layers** with progressive downsampling and channel expansion
- **Adaptive average pooling** → Dropout(0.5) → Linear classifier
- BatchNorm + ReLU + skip connections throughout
- ~500 timesteps → 10-class softmax output

### Training

- **Optimizer**: AdamW (lr=0.001, weight_decay=0.05)
- **Scheduler**: OneCycleLR with 30% linear warmup
- **Augmentations**:
  - Gravity-invariant yaw rotation (random 3D rotation around gravity axis)
  - Time scaling with physics-aware acceleration adjustment (a ∝ 1/t²)
  - Temporal warping via cubic spline interpolation
- **Experiment tracking**: Aim — logs loss, accuracy, LR, per-class metrics, and confusion matrices

---

## Project Structure

```
airpen/
├── firmware/
│   ├── pen_fw/          # Main firmware (MPU6050, Wio Terminal)
│   └── pen_fw_mpu9250/  # Alternate firmware for MPU9250 variant
├── ml/
│   ├── net.py           # ResNet1D model definition
│   ├── dataset.py       # Dataset loading and augmentation
│   ├── processing.py    # Signal processing pipeline
│   ├── train.py         # Training script (Hydra-configured)
│   ├── inference_serv.py # Real-time inference over serial
│   ├── test.py          # Evaluation and confusion matrices
│   ├── record.py        # Data recording from device
│   ├── configs/         # Hydra YAML configs (train, inference, test)
│   └── utils.py         # Helpers
├── data/
│   ├── samples/         # Raw CSV recordings by letter
│   ├── csvs_to_npz.py   # Convert CSVs to NPZ
│   ├── preprocess.py    # Preprocessing pipeline
│   └── split.py         # Stratified train/val/test split
├── vis_transform.py     # Visualize augmentation pipeline
├── vis_pca.py           # PCA transform visualization
├── vis_anim.py          # 3D gesture animation
├── to_onnx.py           # Export to ONNX
├── to_tflite.py         # Export to TFLite (for embedded deployment)
└── pyproject.toml
```

---

## Hardware

| Component | Details |
|-----------|---------|
| Microcontroller | Seeed Wio Terminal (ATMEL SAM D51, ARM Cortex-M4 @ 120 MHz) |
| IMU | MPU6050 — 3-axis accelerometer (±8G) + 3-axis gyroscope (±500°/s) |
| Interface | USB-C serial @ 115200 baud |
| Sampling rate | ~100 Hz (6-channel CSV stream) |
| Gesture window | 500 samples (~5 seconds) |

---

## Installation

**Requirements**: Python 3.11+, [uv](https://github.com/astral-sh/uv), PlatformIO

### Python Environment

```bash
git clone https://github.com/your-username/airpen.git
cd airpen

# Install dependencies
uv sync
# or: pip install -e .
```

### Firmware

```bash
cd firmware/pen_fw

# Compile and upload to Wio Terminal
pio run -t upload

# Monitor serial output
pio device monitor
```

---

## Usage

### 1. Record Training Data

Connect the pen and run the recorder to collect labeled samples:

```bash
python ml/record.py
```

Recordings are saved as CSVs under `data/samples/<letter>/`.

### 2. Preprocess Data

```bash
cd data/

# Convert CSVs to NPZ
python csvs_to_npz.py

# Apply signal processing
python preprocess.py

# Create stratified 70/10/20 split
python split.py samples_processed.npz
```

### 3. Train

```bash
python ml/train.py
# Config at ml/configs/train.yaml — override any param via CLI:
python ml/train.py num_epochs=200 optimizer.params.lr=0.0005
```

Training artifacts and metrics are tracked with **Aim**:

```bash
aim up  # Open Aim dashboard at http://localhost:43800
```

### 4. Evaluate

```bash
python ml/test.py
# Prints per-class accuracy and displays confusion matrix
```

### 5. Real-Time Inference

```bash
python ml/inference_serv.py
```

The script listens on serial, waits for the firmware to signal gesture start, collects 500 samples, runs the full processing + inference pipeline, and prints the predicted letter.

### 6. Export Model

```bash
# ONNX
python to_onnx.py

# TFLite (for embedded / mobile deployment)
python to_tflite.py
```

---

## Configuration

All training, inference, and test parameters are managed with **Hydra** YAML configs:

**`ml/configs/train.yaml`** (key fields):
```yaml
model:
  in_channels: 3      # PCA feature channels
  num_classes: 10     # A, C, E, I, L, N, O, R, S, T

batch_size: 128
num_epochs: 100
seed: 42

optimizer:
  name: AdamW
  params:
    lr: 0.001
    weight_decay: 0.05

scheduler:
  name: OneCycleLR
  params:
    max_lr: 0.01
    pct_start: 0.3    # 30% warmup
```

---

## Technologies & Libraries

### Machine Learning
| Library | Purpose |
|---------|---------|
| PyTorch | Model definition, training, inference |
| TensorFlow / tf-keras | Model export |
| ai-edge-torch | TFLite conversion for embedded deployment |
| scikit-learn | Confusion matrices, stratified splitting |
| scipy | Butterworth filter, cubic spline interpolation |
| einops | Readable tensor operations |

### Infrastructure
| Library | Purpose |
|---------|---------|
| Hydra | Hierarchical config management and CLI overrides |
| Aim | Experiment tracking, metrics, artifact storage |
| ONNX | Cross-platform model serialization |

### Data & Visualization
| Library | Purpose |
|---------|---------|
| NumPy | Array operations and NPZ storage |
| matplotlib | Plotting, confusion matrices, 3D animations |
| pyserial | Serial communication with the embedded device |

### Embedded
| Tool | Purpose |
|------|---------|
| PlatformIO | Embedded build system and device management |
| Arduino framework | Firmware runtime |
| Adafruit MPU6050 | IMU sensor driver |

---

## Dataset

- **Classes**: 10 letters — A, C, E, I, L, N, O, R, S, T
- **Input shape**: (3, 498) after PCA transform and interpolation
- **Split**: 70% train / 10% validation / 20% test (stratified)
- **Format**: NPZ with per-split arrays
