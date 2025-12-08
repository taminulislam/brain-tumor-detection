"""
Evaluation metrics for classification and segmentation
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def calculate_classification_metrics(predictions, targets):
    """
    Calculate classification metrics: accuracy, precision, recall, F1-score

    Args:
        predictions: predicted class labels
        targets: ground truth labels

    Returns:
        dict: dictionary containing all metrics
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def calculate_iou(pred_mask, true_mask, num_classes=2):
    """
    Calculate Intersection over Union (IoU) for segmentation

    Args:
        pred_mask: predicted segmentation mask [B, H, W]
        true_mask: ground truth mask [B, H, W]
        num_classes: number of classes

    Returns:
        float: mean IoU across all classes
    """
    ious = []
    pred_mask = pred_mask.cpu().numpy()
    true_mask = true_mask.cpu().numpy()

    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)

        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        ious.append(iou)

    return np.mean(ious)


def calculate_dice_coefficient(pred_mask, true_mask, num_classes=2):
    """
    Calculate Dice coefficient for segmentation

    Args:
        pred_mask: predicted segmentation mask [B, H, W]
        true_mask: ground truth mask [B, H, W]
        num_classes: number of classes

    Returns:
        float: mean Dice coefficient
    """
    dice_scores = []
    pred_mask = pred_mask.cpu().numpy()
    true_mask = true_mask.cpu().numpy()

    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)

        intersection = np.logical_and(pred_cls, true_cls).sum()

        if pred_cls.sum() + true_cls.sum() == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / (pred_cls.sum() + true_cls.sum())

        dice_scores.append(dice)

    return np.mean(dice_scores)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        int: number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model, input_size=(1, 3, 640, 640), device='cuda'):
    """
    Estimate FLOPs for a model (simplified calculation)

    Args:
        model: PyTorch model
        input_size: input tensor size
        device: device to run on

    Returns:
        int: estimated FLOPs
    """
    try:
        from thop import profile
        input_tensor = torch.randn(input_size).to(device)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops
    except ImportError:
        print("thop not installed. Install with: pip install thop")
        return 0


def calculate_fps(model, input_size=(1, 3, 640, 640), device='cuda', num_iterations=100):
    """
    Calculate frames per second (FPS) for model inference

    Args:
        model: PyTorch model
        input_size: input tensor size
        device: device to run on
        num_iterations: number of iterations for averaging

    Returns:
        float: average FPS
    """
    import time

    model.eval()
    input_tensor = torch.randn(input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Measure
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)

    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_iterations / total_time

    return fps
