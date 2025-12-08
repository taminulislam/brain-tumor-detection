"""
Loss functions for multi-task learning
"""

from .combined_loss import (
    DiceLoss,
    FocalLoss,
    CombinedSegmentationLoss,
    MultiTaskLoss
)

__all__ = [
    'DiceLoss',
    'FocalLoss',
    'CombinedSegmentationLoss',
    'MultiTaskLoss'
]
