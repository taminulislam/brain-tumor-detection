# EfficientMTNet - Quick Start Guide

## Overview

EfficientMTNet is a lightweight multi-task network (~4M parameters) for brain tumor classification and segmentation.

## Directory Structure

```
EfficientMTNet/
├── model/              # EfficientMTNet architecture
├── datasets/           # Dataset loaders
├── losses/             # Loss functions
├── utils/              # Metrics and utilities
├── checkpoints/        # Trained models
├── logs/               # TensorBoard logs
├── results/            # Test results
├── config.py           # Configuration
├── train.py            # Training script
└── test.py             # Testing script
```

## Quick Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure paths in config.py
# Ensure DATA_ROOT points to your dataset

# 3. Train the model
python train.py

# 4. Test the model
python test.py
```

## Key Features

- **Lightweight**: ~4M parameters
- **Fast**: ~45 FPS inference
- **Deep Supervision**: Auxiliary heads for better training
- **Efficient Components**:
  - Depthwise separable convolutions
  - Squeeze-and-Excitation blocks
  - LightHam attention
  - Multi-scale feature aggregation

## Configuration

Edit `config.py` to customize:

```python
# Model
USE_AUX_HEADS = True  # Enable/disable auxiliary heads

# Training
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

# Loss weights
CLASSIFICATION_WEIGHT = 0.4
SEGMENTATION_WEIGHT = 0.6
AUX_WEIGHT = 0.4
```

## Model Architecture

1. **Enhanced Learning-to-Downsample**: Efficient feature extraction with SE blocks
2. **Global Feature Extractor**: Multi-stage inverted residual blocks
3. **Multi-Scale Neck**: FPN-style feature aggregation
4. **LightHam Head**: Efficient global context modeling
5. **Dual Branches**: Classification + Segmentation outputs

## Training Tips

1. **GPU Memory**: Reduce `BATCH_SIZE` if out of memory
2. **Convergence**: Monitor with TensorBoard: `tensorboard --logdir logs/`
3. **Auxiliary Heads**: Keep enabled during training for better features
4. **Early Stopping**: Automatically stops after 15 epochs without improvement

## Citation

```
EfficientMTNet: Efficient Multi-Task Network for Brain Tumor Detection
A Lightweight Architecture with Deep Supervision and Global Context Modeling
```
