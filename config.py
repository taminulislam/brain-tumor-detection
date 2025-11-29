"""
Configuration file for Brain Tumor Segmentation and Classification
"""

import torch

# Dataset paths
DATASET_ROOT = 'brain_tumer_dataset'
TRAIN_DIR = f'{DATASET_ROOT}/train'
VAL_DIR = f'{DATASET_ROOT}/valid'
TEST_DIR = f'{DATASET_ROOT}/test'

TRAIN_ANNOTATIONS = f'{TRAIN_DIR}/_annotations.coco.json'
VAL_ANNOTATIONS = f'{VAL_DIR}/_annotations.coco.json'
TEST_ANNOTATIONS = f'{TEST_DIR}/_annotations.coco.json'

# Model configurations
IMAGE_SIZE = 640
NUM_CLASSES = 3  # Tumor categories: 0, 1, 2
SEGMENTATION_CLASSES = 2  # Background and Tumor

# Training hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Checkpoint and logging
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
RESULTS_DIR = 'results'
SAVE_FREQ = 5  # Save every N epochs

# Loss weights for multi-task learning
CLASSIFICATION_WEIGHT = 0.4
SEGMENTATION_WEIGHT = 0.6

# Model selection
# Options: 'unet', 'resnet_unet', 'vit', 'swin', 'lightweight_transformer'
MODEL_NAME = 'lightweight_transformer'

# Early stopping
EARLY_STOPPING_PATIENCE = 15

# Data augmentation
USE_AUGMENTATION = True
