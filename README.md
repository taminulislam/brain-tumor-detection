# Brain Tumor Detection and Segmentation

Deep learning project for brain tumor classification and segmentation from MRI scans using multi-task learning.

## Project Overview

This project implements various deep learning architectures for simultaneous brain tumor classification and segmentation:

1. **Baseline Models:**
   - U-Net
   - ResNet-based Encoder-Decoder

2. **Transformer-based Models:**
   - Vision Transformer (ViT)
   - Swin Transformer

3. **Lightweight Transformer:**
   - Custom model based on SegFormer-B0
   - Efficient self-attention with reduction ratio
   - Depthwise separable convolutions
   - Window-based attention for reduced computational complexity

## Features

- Multi-task learning (simultaneous classification and segmentation)
- Data augmentation using Albumentations
- Combined Dice + Focal Loss for segmentation
- Cross-Entropy Loss for classification
- Comprehensive evaluation metrics:
  - Classification: Accuracy, Precision, Recall, F1-Score
  - Segmentation: Mean IoU, Dice Coefficient
  - Efficiency: FPS, Model Parameters, FLOPs
- TensorBoard integration for training visualization

## Project Structure

```
deep_learning_project/
├── brain_tumer_dataset/         # Dataset directory
│   ├── train/                   # Training images and annotations
│   ├── valid/                   # Validation images and annotations
│   └── test/                    # Test images and annotations
├── src/
│   ├── models/                  # Model implementations
│   │   ├── unet.py             # U-Net model
│   │   ├── resnet_unet.py      # ResNet-based encoder-decoder
│   │   ├── vit_model.py        # Vision Transformer
│   │   ├── swin_transformer.py # Swin Transformer
│   │   └── lightweight_transformer.py  # Lightweight model
│   ├── datasets/                # Dataset loaders
│   │   └── brain_tumor_dataset.py
│   ├── losses/                  # Loss functions
│   │   └── combined_loss.py
│   └── utils/                   # Utility functions
│       └── metrics.py
├── checkpoints/                 # Saved model checkpoints
├── logs/                        # TensorBoard logs
├── results/                     # Test results and visualizations
├── config.py                    # Configuration file
├── train.py                     # Training script
├── test.py                      # Testing script
└── README.md                    # This file
```

## Installation

### 1. Create Conda Environment

```bash
cd deep_learning_project
conda create -n brain_tumor python=3.9 -y
conda activate brain_tumor
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers timm segmentation-models-pytorch albumentations \
    opencv-python matplotlib seaborn scikit-learn pycocotools tqdm tensorboard
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

Train a model using the training script:

```bash
# Activate environment
conda activate brain_tumor

# Train with default settings (Lightweight Transformer)
python train.py

# Train specific model by editing config.py
# Change MODEL_NAME to: 'unet', 'resnet_unet', 'vit', 'swin', or 'lightweight_transformer'
```

Configuration options in `config.py`:
- `MODEL_NAME`: Model architecture to use
- `BATCH_SIZE`: Training batch size (default: 8)
- `NUM_EPOCHS`: Number of training epochs (default: 100)
- `LEARNING_RATE`: Initial learning rate (default: 1e-4)
- `CLASSIFICATION_WEIGHT`: Weight for classification loss (default: 0.4)
- `SEGMENTATION_WEIGHT`: Weight for segmentation loss (default: 0.6)

### Testing

Evaluate a trained model on the test set:

```bash
# Test the best model
python test.py --model lightweight_transformer --checkpoint checkpoints/lightweight_transformer_best.pth

# Test specific checkpoint
python test.py --model unet --checkpoint checkpoints/unet_epoch_50.pth
```

### TensorBoard Visualization

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## Model Architectures

### 1. U-Net
- Classic encoder-decoder architecture
- Skip connections for better localization
- ~31M parameters

### 2. ResNet-UNet
- ResNet-50 pretrained encoder
- U-Net style decoder
- ~45M parameters

### 3. Vision Transformer (ViT)
- Transformer-based architecture
- Patch-based image processing
- ~86M parameters

### 4. Swin Transformer
- Hierarchical vision transformer
- Shifted window attention
- ~88M parameters

### 5. Lightweight Transformer (Proposed)
- Based on SegFormer-B0 architecture
- Efficient self-attention with spatial reduction
- Depthwise separable convolutions
- Mix-FFN for positional encoding
- ~4M parameters (most efficient)

## Evaluation Metrics

### Classification Metrics
- **Accuracy:** Overall classification accuracy
- **Precision:** Ratio of true positives to predicted positives
- **Recall:** Ratio of true positives to actual positives
- **F1-Score:** Harmonic mean of precision and recall

### Segmentation Metrics
- **Mean IoU (mIoU):** Intersection over Union averaged across classes
- **Dice Coefficient:** Similarity metric for segmentation masks

### Efficiency Metrics
- **Parameters:** Total number of trainable parameters
- **FPS:** Frames per second (inference speed)
- **FLOPs:** Floating point operations (computational cost)

## Results

After training, results are saved in the `results/` directory:
- `{model_name}_test_results.json`: Comprehensive test metrics
- `{model_name}_confusion_matrix.png`: Classification confusion matrix
- `{model_name}_history.json`: Training history

## Loss Functions

### Classification Loss
- Cross-Entropy Loss for tumor type classification

### Segmentation Loss
- **Dice Loss:** Optimizes overlap between prediction and ground truth
- **Focal Loss:** Addresses class imbalance by focusing on hard examples
- **Combined Loss:** Weighted sum of Dice and Focal losses

### Multi-Task Loss
- Weighted combination of classification and segmentation losses
- Default weights: 0.4 (classification) + 0.6 (segmentation)

## Training Features

- **Early Stopping:** Stops training if validation loss doesn't improve for 15 epochs
- **Learning Rate Scheduling:** ReduceLROnPlateau scheduler
- **Checkpoint Saving:** Saves best model and periodic checkpoints
- **Data Augmentation:** Random flips, rotations, brightness/contrast adjustments
- **Mixed Precision Training:** Optional for faster training

## Tips for Best Results

1. **Adjust batch size** based on GPU memory (reduce if OOM errors occur)
2. **Use data augmentation** for better generalization
3. **Monitor TensorBoard** to detect overfitting early
4. **Try different models** to find the best architecture for your use case
5. **Tune loss weights** if one task dominates the other

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `config.py`
- Use smaller model (e.g., lightweight_transformer instead of vit)

### Slow Training
- Increase `NUM_WORKERS` for faster data loading
- Use mixed precision training
- Reduce image size if acceptable

### Poor Performance
- Increase `NUM_EPOCHS`
- Adjust learning rate
- Try different loss weights
- Use data augmentation

## Citation

If you use this code, please cite:

```
Brain Tumor Detection and Segmentation using Deep Learning
Multi-task Learning with Transformer Architectures
```

## License

This project is for educational purposes.
