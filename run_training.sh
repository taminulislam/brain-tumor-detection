#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate brain_tumor

# Run training
echo "Starting training with configuration:"
echo "Model: See config.py for MODEL_NAME"
echo "Device: CUDA if available, else CPU"
echo ""

python train.py

echo ""
echo "Training completed! Check logs/ directory for TensorBoard logs"
echo "Checkpoints saved in checkpoints/ directory"
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir logs/"
