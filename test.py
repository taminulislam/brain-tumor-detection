"""
Testing script for Brain Tumor Detection and Segmentation
Evaluates model on test set and calculates comprehensive metrics
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

# Add src to path
sys.path.append('src')

import config
from models import get_model
from datasets.brain_tumor_dataset import get_dataloaders
from losses.combined_loss import MultiTaskLoss
from utils.metrics import (calculate_classification_metrics, calculate_iou,
                           calculate_dice_coefficient, count_parameters,
                           calculate_fps)


class Tester:
    """Tester class for model evaluation"""
    def __init__(self, model_name, checkpoint_path, device):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device

        # Create results directory
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        # Get dataloaders
        print("Loading datasets...")
        _, _, self.test_loader = get_dataloaders(
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
        print(f"Test batches: {len(self.test_loader)}")

        # Initialize model
        print(f"Initializing {model_name} model...")
        self.model = get_model(
            model_name=model_name,
            n_classes_seg=config.SEGMENTATION_CLASSES,
            n_classes_cls=config.NUM_CLASSES - 1,
            img_size=config.IMAGE_SIZE
        ).to(device)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'N/A')})")

        # Model statistics
        num_params = count_parameters(self.model)
        print(f"Total trainable parameters: {num_params:,} ({num_params/1e6:.2f}M)")

        # Initialize loss function
        self.criterion = MultiTaskLoss(
            classification_weight=config.CLASSIFICATION_WEIGHT,
            segmentation_weight=config.SEGMENTATION_WEIGHT
        )

    def test(self):
        """Test the model on test set"""
        self.model.eval()

        running_loss = 0.0
        running_cls_loss = 0.0
        running_seg_loss = 0.0

        all_cls_preds = []
        all_cls_targets = []
        all_seg_preds = []
        all_seg_targets = []

        inference_times = []

        print("\nRunning inference on test set...")
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for batch in pbar:
                images = batch['image'].to(self.device)
                seg_masks = batch['segmentation_mask'].to(self.device)
                cls_labels = batch['classification_label'].to(self.device)

                # Measure inference time
                start_time = time.time()
                cls_output, seg_output = self.model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / images.size(0))  # Per image

                # Calculate loss
                loss_dict = self.criterion(cls_output, seg_output, cls_labels, seg_masks)
                total_loss = loss_dict['total_loss']

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

        # Concatenate all predictions
        all_cls_preds = torch.cat(all_cls_preds)
        all_cls_targets = torch.cat(all_cls_targets)
        all_seg_preds = torch.cat(all_seg_preds)
        all_seg_targets = torch.cat(all_seg_targets)

        # Calculate metrics
        print("\nCalculating metrics...")

        # Classification metrics
        cls_metrics = calculate_classification_metrics(all_cls_preds, all_cls_targets)
        cm = confusion_matrix(all_cls_targets.numpy(), all_cls_preds.numpy())
        cls_report = classification_report(all_cls_targets.numpy(), all_cls_preds.numpy(),
                                          target_names=['Class 0', 'Class 1'], output_dict=True)

        # Segmentation metrics
        seg_iou = calculate_iou(all_seg_preds, all_seg_targets, num_classes=config.SEGMENTATION_CLASSES)
        seg_dice = calculate_dice_coefficient(all_seg_preds, all_seg_targets,
                                              num_classes=config.SEGMENTATION_CLASSES)

        # Loss metrics
        avg_loss = running_loss / len(self.test_loader)
        avg_cls_loss = running_cls_loss / len(self.test_loader)
        avg_seg_loss = running_seg_loss / len(self.test_loader)

        # Performance metrics
        avg_inference_time = np.mean(inference_times)
        fps = 1.0 / avg_inference_time
        num_params = count_parameters(self.model)

        # Compile results
        results = {
            'model_name': self.model_name,
            'checkpoint': self.checkpoint_path,
            'test_loss': avg_loss,
            'classification_loss': avg_cls_loss,
            'segmentation_loss': avg_seg_loss,
            'classification_metrics': {
                'accuracy': cls_metrics['accuracy'],
                'precision': cls_metrics['precision'],
                'recall': cls_metrics['recall'],
                'f1_score': cls_metrics['f1_score'],
                'per_class': cls_report
            },
            'segmentation_metrics': {
                'mean_iou': seg_iou,
                'dice_coefficient': seg_dice
            },
            'efficiency_metrics': {
                'parameters': num_params,
                'parameters_M': num_params / 1e6,
                'avg_inference_time_sec': avg_inference_time,
                'fps': fps
            }
        }

        # Print results
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"\nModel: {self.model_name}")
        print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"\nLoss Metrics:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Classification Loss: {avg_cls_loss:.4f}")
        print(f"  Segmentation Loss: {avg_seg_loss:.4f}")
        print(f"\nClassification Metrics:")
        print(f"  Accuracy: {cls_metrics['accuracy']:.4f}")
        print(f"  Precision: {cls_metrics['precision']:.4f}")
        print(f"  Recall: {cls_metrics['recall']:.4f}")
        print(f"  F1-Score: {cls_metrics['f1_score']:.4f}")
        print(f"\nSegmentation Metrics:")
        print(f"  Mean IoU: {seg_iou:.4f}")
        print(f"  Dice Coefficient: {seg_dice:.4f}")
        print(f"\nEfficiency Metrics:")
        print(f"  Avg Inference Time: {avg_inference_time*1000:.2f} ms/image")
        print(f"  FPS: {fps:.2f}")
        print("="*60)

        # Save results
        results_path = os.path.join(config.RESULTS_DIR, f'{self.model_name}_test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {results_path}")

        # Plot confusion matrix
        self.plot_confusion_matrix(cm, save_path=os.path.join(config.RESULTS_DIR,
                                                              f'{self.model_name}_confusion_matrix.png'))

        return results

    def plot_confusion_matrix(self, cm, save_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Test Brain Tumor Detection Model')
    parser.add_argument('--model', type=str, default=config.MODEL_NAME,
                       help='Model name (unet, resnet_unet, vit, swin, lightweight_transformer)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Create tester and run test
    tester = Tester(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    tester.test()


if __name__ == "__main__":
    main()
