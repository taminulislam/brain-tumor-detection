"""
Loss functions for multi-task learning:
- Cross-Entropy Loss for classification
- Combined Dice + Focal Loss for segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, C, H, W] - predicted logits
            targets: [B, H, W] - ground truth masks
        """
        # Convert targets to one-hot encoding
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Apply softmax to predictions
        predictions = F.softmax(predictions, dim=1)

        # Calculate Dice coefficient for each class
        intersection = (predictions * targets_one_hot).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, C, H, W] - predicted logits
            targets: [B, H, W] - ground truth masks
        """
        # Calculate cross entropy
        ce_loss = F.cross_entropy(predictions, targets.long(), reduction='none')

        # Calculate pt
        pt = torch.exp(-ce_loss)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedSegmentationLoss(nn.Module):
    """
    Combined Dice + Focal Loss for segmentation
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, smooth=1.0, alpha=0.25, gamma=2.0):
        super(CombinedSegmentationLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, C, H, W] - predicted logits
            targets: [B, H, W] - ground truth masks
        """
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)

        combined_loss = self.dice_weight * dice + self.focal_weight * focal
        return combined_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification and segmentation losses
    Supports auxiliary heads for deep supervision
    """
    def __init__(self, classification_weight=0.4, segmentation_weight=0.6,
                 dice_weight=0.5, focal_weight=0.5, aux_weight=0.4):
        super(MultiTaskLoss, self).__init__()
        self.classification_weight = classification_weight
        self.segmentation_weight = segmentation_weight
        self.aux_weight = aux_weight

        self.classification_loss = nn.CrossEntropyLoss()
        self.segmentation_loss = CombinedSegmentationLoss(
            dice_weight=dice_weight,
            focal_weight=focal_weight
        )

    def forward(self, cls_pred, seg_pred, cls_target, seg_target, aux_preds=None):
        """
        Args:
            cls_pred: [B, num_classes] - classification predictions
            seg_pred: [B, 2, H, W] - segmentation predictions
            cls_target: [B] - classification targets
            seg_target: [B, H, W] - segmentation targets
            aux_preds: list of auxiliary segmentation predictions (optional)

        Returns:
            dict: dictionary with total loss and individual losses
        """
        cls_loss = self.classification_loss(cls_pred, cls_target.long())
        seg_loss = self.segmentation_loss(seg_pred, seg_target)

        total_loss = (self.classification_weight * cls_loss +
                     self.segmentation_weight * seg_loss)

        # Add auxiliary losses if provided
        if aux_preds is not None:
            aux_loss = 0
            for aux_pred in aux_preds:
                aux_loss += self.segmentation_loss(aux_pred, seg_target)
            aux_loss = aux_loss / len(aux_preds)
            total_loss = total_loss + self.aux_weight * aux_loss

        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss.item(),
            'segmentation_loss': seg_loss.item()
        }
