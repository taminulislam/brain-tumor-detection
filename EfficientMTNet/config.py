"""
Configuration file for EfficientMTNet training and testing
"""

import torch
import os

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configuration
MODEL_NAME = 'efficient_mtnet'
USE_AUX_HEADS = True  # Use auxiliary heads for deep supervision

# Dataset paths
DATA_ROOT = os.path.join('..', 'brain_tumer_dataset')
TRAIN_DIR = os.path.join(DATA_ROOT, 'train', 'images')
VAL_DIR = os.path.join(DATA_ROOT, 'valid', 'images')
TEST_DIR = os.path.join(DATA_ROOT, 'test', 'images')
TRAIN_ANNOTATIONS = os.path.join(DATA_ROOT, 'train', 'annotations.json')
VAL_ANNOTATIONS = os.path.join(DATA_ROOT, 'valid', 'annotations.json')
TEST_ANNOTATIONS = os.path.join(DATA_ROOT, 'test', 'annotations.json')

# Image configuration
IMAGE_SIZE = 640
NUM_CLASSES = 3  # Total classes in dataset (including background)
SEGMENTATION_CLASSES = 2  # Binary segmentation (background, tumor)

# Training hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Loss weights
CLASSIFICATION_WEIGHT = 0.4
SEGMENTATION_WEIGHT = 0.6
AUX_WEIGHT = 0.4  # Weight for auxiliary heads

# Data loading
NUM_WORKERS = 4

# Training settings
EARLY_STOPPING_PATIENCE = 15
SAVE_FREQ = 10  # Save checkpoint every N epochs

# Directories
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
RESULTS_DIR = 'results'

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
