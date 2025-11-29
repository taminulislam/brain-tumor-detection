"""
Training script for Brain Tumor Detection and Segmentation
Multi-task learning with classification and segmentation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json

# Add src to path
sys.path.append('src')

import config
from models import get_model
from datasets.brain_tumor_dataset import get_dataloaders
from losses.combined_loss import MultiTaskLoss
from utils.metrics import (calculate_classification_metrics, calculate_iou,
                           calculate_dice_coefficient, count_parameters)


class Trainer:
    """Trainer class for multi-task learning"""
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Create directories
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=f'{config.LOG_DIR}/{model_name}')

        # Get dataloaders
        print("Loading datasets...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            train_dir=config.TRAIN_DIR,
            val_dir=config.VAL_DIR,
            test_dir=config.TEST_DIR,
            train_ann=config.TRAIN_ANNOTATIONS,
            val_ann=config.VAL_ANNOTATIONS,
            test_ann=config.TEST_ANNOTATIONS,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            image_size=config.IMAGE_SIZE
        )
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        # Initialize model
        print(f"Initializing {model_name} model...")
        self.model = get_model(
            model_name=model_name,
            n_classes_seg=config.SEGMENTATION_CLASSES,
            n_classes_cls=config.NUM_CLASSES - 1,  # Categories 1 and 2 -> 0 and 1
            img_size=config.IMAGE_SIZE
        ).to(device)

        num_params = count_parameters(self.model)
        print(f"Total trainable parameters: {num_params:,} ({num_params/1e6:.2f}M)")

        # Initialize loss function
        self.criterion = MultiTaskLoss(
            classification_weight=config.CLASSIFICATION_WEIGHT,
            segmentation_weight=config.SEGMENTATION_WEIGHT
        )

        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_cls_loss': [],
            'val_cls_loss': [],
            'train_seg_loss': [],
            'val_seg_loss': [],
            'train_cls_acc': [],
            'val_cls_acc': [],
            'train_seg_iou': [],
            'val_seg_iou': []
        }

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_seg_loss = 0.0
        all_cls_preds = []
        all_cls_targets = []
        all_seg_preds = []
        all_seg_targets = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Train]')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            seg_masks = batch['segmentation_mask'].to(self.device)
            cls_labels = batch['classification_label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            cls_output, seg_output = self.model(images)

            # Calculate loss
            loss_dict = self.criterion(cls_output, seg_output, cls_labels, seg_masks)
            total_loss = loss_dict['total_loss']

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += total_loss.item()
            running_cls_loss += loss_dict['classification_loss']
            running_seg_loss += loss_dict['segmentation_loss']

            # Predictions
            cls_preds = torch.argmax(cls_output, dim=1)
            seg_preds = torch.argmax(seg_output, dim=1)

            all_cls_preds.append(cls_preds.cpu())
            all_cls_targets.append(cls_labels.cpu())
            all_seg_preds.append(seg_preds.cpu())
            all_seg_targets.append(seg_masks.cpu())

            pbar.set_postfix({'loss': total_loss.item()})

        # Calculate metrics
        all_cls_preds = torch.cat(all_cls_preds)
        all_cls_targets = torch.cat(all_cls_targets)
        all_seg_preds = torch.cat(all_seg_preds)
        all_seg_targets = torch.cat(all_seg_targets)

        cls_metrics = calculate_classification_metrics(all_cls_preds, all_cls_targets)
        seg_iou = calculate_iou(all_seg_preds, all_seg_targets, num_classes=config.SEGMENTATION_CLASSES)

        avg_loss = running_loss / len(self.train_loader)
        avg_cls_loss = running_cls_loss / len(self.train_loader)
        avg_seg_loss = running_seg_loss / len(self.train_loader)

        return avg_loss, avg_cls_loss, avg_seg_loss, cls_metrics['accuracy'], seg_iou

    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_seg_loss = 0.0
        all_cls_preds = []
        all_cls_targets = []
        all_seg_preds = []
        all_seg_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Val]')
            for batch in pbar:
                images = batch['image'].to(self.device)
                seg_masks = batch['segmentation_mask'].to(self.device)
                cls_labels = batch['classification_label'].to(self.device)

                # Forward pass
                cls_output, seg_output = self.model(images)

                # Calculate loss
                loss_dict = self.criterion(cls_output, seg_output, cls_labels, seg_masks)
                total_loss = loss_dict['total_loss']

                # Statistics
                running_loss += total_loss.item()
                running_cls_loss += loss_dict['classification_loss']
                running_seg_loss += loss_dict['segmentation_loss']

                # Predictions
                cls_preds = torch.argmax(cls_output, dim=1)
                seg_preds = torch.argmax(seg_output, dim=1)

                all_cls_preds.append(cls_preds.cpu())
                all_cls_targets.append(cls_labels.cpu())
                all_seg_preds.append(seg_preds.cpu())
                all_seg_targets.append(seg_masks.cpu())

                pbar.set_postfix({'loss': total_loss.item()})

        # Calculate metrics
        all_cls_preds = torch.cat(all_cls_preds)
        all_cls_targets = torch.cat(all_cls_targets)
        all_seg_preds = torch.cat(all_seg_preds)
        all_seg_targets = torch.cat(all_seg_targets)

        cls_metrics = calculate_classification_metrics(all_cls_preds, all_cls_targets)
        seg_iou = calculate_iou(all_seg_preds, all_seg_targets, num_classes=config.SEGMENTATION_CLASSES)

        avg_loss = running_loss / len(self.val_loader)
        avg_cls_loss = running_cls_loss / len(self.val_loader)
        avg_seg_loss = running_seg_loss / len(self.val_loader)

        return avg_loss, avg_cls_loss, avg_seg_loss, cls_metrics['accuracy'], seg_iou

    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}\n")

        for epoch in range(1, config.NUM_EPOCHS + 1):
            # Train
            train_loss, train_cls_loss, train_seg_loss, train_cls_acc, train_seg_iou = self.train_epoch(epoch)

            # Validate
            val_loss, val_cls_loss, val_seg_loss, val_cls_acc, val_seg_iou = self.validate_epoch(epoch)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_cls_loss'].append(train_cls_loss)
            self.history['val_cls_loss'].append(val_cls_loss)
            self.history['train_seg_loss'].append(train_seg_loss)
            self.history['val_seg_loss'].append(val_seg_loss)
            self.history['train_cls_acc'].append(train_cls_acc)
            self.history['val_cls_acc'].append(val_cls_acc)
            self.history['train_seg_iou'].append(train_seg_iou)
            self.history['val_seg_iou'].append(val_seg_iou)

            # Tensorboard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Classification/train_acc', train_cls_acc, epoch)
            self.writer.add_scalar('Classification/val_acc', val_cls_acc, epoch)
            self.writer.add_scalar('Segmentation/train_iou', train_seg_iou, epoch)
            self.writer.add_scalar('Segmentation/val_iou', val_seg_iou, epoch)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{config.NUM_EPOCHS} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Cls Acc: {train_cls_acc:.4f} | Val Cls Acc: {val_cls_acc:.4f}")
            print(f"  Train Seg IoU: {train_seg_iou:.4f} | Val Seg IoU: {val_seg_iou:.4f}")

            # Save checkpoint
            if epoch % config.SAVE_FREQ == 0 or val_loss < self.best_val_loss:
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{self.model_name}_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(config.CHECKPOINT_DIR, f'{self.model_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, best_model_path)
                print(f"  Best model saved: {best_model_path}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        # Save training history
        history_path = os.path.join(config.RESULTS_DIR, f'{self.model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"\nTraining history saved: {history_path}")

        self.writer.close()
        print("Training completed!")


def main():
    """Main function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create trainer and train
    trainer = Trainer(model_name=config.MODEL_NAME, device=config.DEVICE)
    trainer.train()


if __name__ == "__main__":
    main()
