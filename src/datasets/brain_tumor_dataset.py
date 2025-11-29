"""
Dataset loader for Brain Tumor Detection and Segmentation
Handles COCO format annotations
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class BrainTumorDataset(Dataset):
    """
    Brain Tumor Dataset for multi-task learning (classification + segmentation)
    """
    def __init__(self, image_dir, annotation_file, image_size=640, augmentation=True, is_train=True):
        """
        Args:
            image_dir: directory containing images
            annotation_file: path to COCO format annotation JSON
            image_size: size to resize images to
            augmentation: whether to apply data augmentation
            is_train: whether this is training set
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.is_train = is_train

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']

        # Create image_id to annotation mapping
        self.image_to_annotation = {}
        for ann in self.annotations:
            self.image_to_annotation[ann['image_id']] = ann

        # Define transforms
        self.transform = self.get_transforms(augmentation, is_train)

    def get_transforms(self, augmentation, is_train):
        """Get albumentations transforms"""
        if augmentation and is_train:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})

    def create_mask_from_segmentation(self, segmentation, height, width):
        """
        Create binary mask from COCO segmentation polygon

        Args:
            segmentation: list of polygon coordinates
            height: image height
            width: image width

        Returns:
            numpy array: binary mask
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        if isinstance(segmentation, list) and len(segmentation) > 0:
            for seg in segmentation:
                if len(seg) >= 6:  # Need at least 3 points (6 coordinates)
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)

        return mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            image: [3, H, W] tensor
            segmentation_mask: [H, W] tensor with values 0 or 1
            classification_label: scalar tensor (category_id: 1 or 2)
        """
        # Load image info
        image_info = self.images[idx]
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_height = image_info['height']
        image_width = image_info['width']

        # Load image
        image_path = os.path.join(self.image_dir, image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotation
        annotation = self.image_to_annotation.get(image_id, None)

        if annotation is not None:
            # Get classification label (category_id: 1 or 2)
            # We'll map to 0-indexed: 0 or 1
            category_id = annotation['category_id']
            if category_id == 1:
                classification_label = 0
            elif category_id == 2:
                classification_label = 1
            else:
                classification_label = 0  # Default

            # Create segmentation mask
            segmentation = annotation.get('segmentation', [])
            mask = self.create_mask_from_segmentation(segmentation, image_height, image_width)
        else:
            # No annotation found - default values
            classification_label = 0
            mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # Convert mask to tensor if needed (albumentations may return tensor with ToTensorV2)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return {
            'image': image,
            'segmentation_mask': mask,
            'classification_label': torch.tensor(classification_label, dtype=torch.long),
            'image_id': image_id,
            'filename': image_filename
        }


def get_dataloaders(train_dir, val_dir, test_dir,
                   train_ann, val_ann, test_ann,
                   batch_size=8, num_workers=4, image_size=640):
    """
    Create dataloaders for train, validation, and test sets

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = BrainTumorDataset(
        image_dir=train_dir,
        annotation_file=train_ann,
        image_size=image_size,
        augmentation=True,
        is_train=True
    )

    val_dataset = BrainTumorDataset(
        image_dir=val_dir,
        annotation_file=val_ann,
        image_size=image_size,
        augmentation=False,
        is_train=False
    )

    test_dataset = BrainTumorDataset(
        image_dir=test_dir,
        annotation_file=test_ann,
        image_size=image_size,
        augmentation=False,
        is_train=False
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
