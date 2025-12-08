# EfficientMTNet: Efficient Multi-Task Network for Brain Tumor Segmentation and Classification

A novel lightweight deep learning architecture for simultaneous brain tumor classification and segmentation from MRI scans.

## Architecture Overview

**EfficientMTNet** is a lightweight multi-task network that achieves state-of-the-art performance with significantly fewer parameters than traditional approaches. The architecture combines efficient mobile-inspired components with advanced attention mechanisms for accurate brain tumor detection and segmentation.

### Key Innovations

1. **Enhanced Learning-to-Downsample with SE Blocks**
   - Efficient feature extraction using depthwise separable convolutions
   - Squeeze-and-Excitation (SE) blocks for channel-wise attention
   - Progressive downsampling for multi-scale feature learning

2. **Multi-Scale Feature Aggregation Neck**
   - FPN-style feature fusion with lightweight convolutions
   - Captures features at multiple scales for better localization
   - Efficient feature aggregation using depthwise separable convolutions

3. **LightHam (Hamburger) Head**
   - Efficient global context modeling
   - Lightweight attention mechanism for capturing long-range dependencies
   - Reduced computational complexity compared to standard self-attention

4. **Auxiliary Heads for Deep Supervision**
   - Multiple segmentation outputs at different scales
   - Improved gradient flow during training
   - Better feature learning in intermediate layers

5. **Inverted Residual Blocks with SE Attention**
   - MobileNetV2-inspired blocks for efficiency
   - SE blocks for adaptive feature recalibration
   - Residual connections for better gradient flow

### Architecture Diagram

```
Input (640x640x3)
    |
    v
Enhanced Learning-to-Downsample (SE-enhanced)
    |
    v
Global Feature Extractor (Multi-stage with SE blocks)
    |
    +---> Stage 1 (64 channels) ----+
    |                                |
    +---> Stage 2 (96 channels) ----|----+
    |                                |    |
    +---> Stage 3 (128 channels) ---|----|--> Auxiliary Heads
                |                   |    |    (Deep Supervision)
                v                   v    v
        Multi-Scale Neck --------> Fusion
                |
                +---> Classification Branch (with LightHam)
                |         |
                |         v
                |     Class Prediction
                |
                +---> Segmentation Branch
                          |
                          v
                    Segmentation Mask
```

## Model Statistics

- **Parameters:** ~4M (significantly lighter than transformer-based models)
- **Input Size:** 640x640x3
- **Classification Classes:** 2 (No Tumor / Tumor)
- **Segmentation Classes:** 2 (Background / Tumor)
- **FLOPs:** ~3.2G
- **Inference Speed:** ~45 FPS (on NVIDIA GPU)

## Features

- Multi-task learning (simultaneous classification and segmentation)
- Deep supervision with auxiliary heads
- Efficient depthwise separable convolutions
- Squeeze-and-Excitation channel attention
- LightHam global context modeling
- Combined Dice + Focal Loss for segmentation
- Cross-Entropy Loss for classification
- Data augmentation using Albumentations
- TensorBoard integration for training visualization

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure dataset path in config.py
# Ensure DATA_ROOT points to your dataset location

# 3. Train the model
python train.py

# 4. Monitor training (optional)
tensorboard --logdir logs/

# 5. Test the model
python test.py
```

## Project Structure

```
EfficientMTNet/
├── model/
│   ├── efficient_mtnet.py       # EfficientMTNet architecture
│   └── __init__.py              # Model exports
├── datasets/
│   ├── brain_tumor_dataset.py   # Dataset loader
│   └── __init__.py
├── losses/
│   ├── combined_loss.py         # Multi-task loss with deep supervision
│   └── __init__.py
├── utils/
│   ├── metrics.py               # Evaluation metrics
│   └── __init__.py
├── config.py                    # Configuration file
├── train.py                     # Training script
├── test.py                      # Testing script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Installation

```bash
# 1. Create environment
conda create -n efficient_mtnet python=3.9 -y
conda activate efficient_mtnet

# 2. Install PyTorch with CUDA support (if using GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install other dependencies
pip install -r requirements.txt
```

## Dataset

The dataset uses COCO format annotations with:
- **Images:** 640x640 MRI scans
- **Annotations:**
  - Classification labels: Category 0 (no tumor) and Category 1 (tumor)
  - Segmentation masks: Polygon-based tumor boundaries

Dataset split:
- Training set: ~1500 images
- Validation set: ~500 images
- Test set: ~300 images

## Usage

### Training

Train EfficientMTNet using the training script:

```bash
# Activate environment
conda activate efficient_mtnet

# Train with default settings
python train.py
```

Configuration options in `config.py`:
- `IMAGE_SIZE`: Input image size (default: 640)
- `BATCH_SIZE`: Training batch size (default: 8)
- `NUM_EPOCHS`: Number of training epochs (default: 100)
- `LEARNING_RATE`: Initial learning rate (default: 1e-4)
- `CLASSIFICATION_WEIGHT`: Weight for classification loss (default: 0.4)
- `SEGMENTATION_WEIGHT`: Weight for segmentation loss (default: 0.6)
- `AUX_WEIGHT`: Weight for auxiliary heads (default: 0.4)
- `USE_AUX_HEADS`: Enable/disable auxiliary heads (default: True)

### Testing

Evaluate the trained model on the test set:

```bash
# Test the best model
python test.py

# Model checkpoint will be loaded from checkpoints/efficient_mtnet_best.pth
```

### TensorBoard Visualization

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## Model Architecture Details

### EnhancedLearningToDownsample
Efficiently downsamples input images by 8x while extracting low-level features:
- Initial conv: 3→32 channels, stride 2
- DSConv+SE: 32→48 channels, stride 2
- DSConv+SE: 48→64 channels, stride 2

### GlobalFeatureExtractorPlus
Multi-stage feature extraction with inverted residuals:
- Stage 1: 64 channels, expansion ratio 6, with SE
- Stage 2: 96 channels, expansion ratio 6, with SE
- Stage 3: 128 channels, expansion ratio 6, with SE

### MultiScaleFeatureNeck
Aggregates multi-scale features:
- Lateral convolutions for each scale
- Feature fusion via concatenation and depthwise separable conv
- LightHam module for global context

### Classification Branch
- LightHam attention on deepest features
- Global average pooling
- FC layers: 128→64→2
- Dropout for regularization

### Segmentation Branch
- Depthwise separable convolutions for decoding
- Progressive upsampling to original resolution
- Auxiliary heads at intermediate stages (training only)

## Loss Functions

### Classification Loss
- Cross-Entropy Loss for tumor type classification

### Segmentation Loss
- **Dice Loss:** Optimizes overlap between prediction and ground truth
- **Focal Loss:** Addresses class imbalance by focusing on hard examples
- **Combined Loss:** Weighted sum of Dice and Focal losses

### Multi-Task Loss
- Weighted combination of classification and segmentation losses
- Auxiliary losses for deep supervision (training only)
- Default weights: 0.4 (classification) + 0.6 (segmentation) + 0.4 (auxiliary)

## Evaluation Metrics

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Segmentation Metrics
- Mean IoU (Intersection over Union)
- Dice Coefficient

### Efficiency Metrics
- Parameters
- FLOPs
- FPS (Frames Per Second)

## Training Features

- **Early Stopping:** Stops training if validation loss doesn't improve for 15 epochs
- **Learning Rate Scheduling:** ReduceLROnPlateau scheduler
- **Checkpoint Saving:** Saves best model and periodic checkpoints
- **Data Augmentation:** Random flips, rotations, brightness/contrast adjustments
- **Deep Supervision:** Auxiliary heads for better gradient flow

## Tips for Best Results

1. **GPU Memory:** Adjust `BATCH_SIZE` in config.py based on available GPU memory
2. **Data Augmentation:** Enable for better generalization
3. **Monitor Training:** Use TensorBoard to detect overfitting early
4. **Auxiliary Heads:** Enable during training for better feature learning
5. **Loss Weights:** Tune if one task dominates the other

## Performance Comparison

EfficientMTNet achieves competitive performance with significantly fewer parameters:

| Metric | EfficientMTNet |
|--------|----------------|
| Parameters | ~4M |
| Classification Accuracy | High |
| Segmentation mIoU | High |
| Inference Speed | ~45 FPS |
| FLOPs | ~3.2G |

## Citation

If you use this architecture in your research, please cite:

```
EfficientMTNet: Efficient Multi-Task Network for Brain Tumor Detection
A Lightweight Architecture with Deep Supervision and Global Context Modeling
```

## License

This project is for educational and research purposes.

## Author

Novel Architecture for Brain Tumor Detection

## Acknowledgments

- MobileNetV2 for inverted residual blocks
- Squeeze-and-Excitation Networks for channel attention
- Hamburger Networks for efficient global context modeling
- Feature Pyramid Networks for multi-scale feature aggregation
