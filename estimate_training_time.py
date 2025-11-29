"""
Quick script to estimate training time
"""

import sys
import torch
import time
sys.path.append('src')

import config
from models import get_model
from datasets.brain_tumor_dataset import get_dataloaders

# Get dataloaders
print("Loading datasets...")
train_loader, _, _ = get_dataloaders(
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

print(f"Training batches: {len(train_loader)}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Total training images: {len(train_loader) * config.BATCH_SIZE}")

# Initialize model
print(f"\nInitializing {config.MODEL_NAME} model...")
model = get_model(
    model_name=config.MODEL_NAME,
    n_classes_seg=config.SEGMENTATION_CLASSES,
    n_classes_cls=config.NUM_CLASSES - 1,
    img_size=config.IMAGE_SIZE
).to(config.DEVICE)

print(f"Device: {config.DEVICE}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Benchmark training
print("\nRunning benchmark on 10 batches...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

start_time = time.time()
for idx, batch in enumerate(train_loader):
    if idx >= 10:
        break

    images = batch['image'].to(config.DEVICE)
    seg_masks = batch['segmentation_mask'].to(config.DEVICE)
    cls_labels = batch['classification_label'].to(config.DEVICE)

    optimizer.zero_grad()
    cls_output, seg_output = model(images)

    # Simple loss for benchmark
    cls_loss = torch.nn.functional.cross_entropy(cls_output, cls_labels)
    seg_loss = torch.nn.functional.cross_entropy(seg_output, seg_masks)
    loss = cls_loss + seg_loss

    loss.backward()
    optimizer.step()

    if idx == 0:
        print(f"First batch completed")

end_time = time.time()
elapsed = end_time - start_time

# Calculate estimates
avg_time_per_batch = elapsed / 10
total_batches_per_epoch = len(train_loader)
time_per_epoch = avg_time_per_batch * total_batches_per_epoch

# Assuming validation takes ~40% of training time
time_per_epoch_with_val = time_per_epoch * 1.4

total_time_100_epochs = time_per_epoch_with_val * 100

print("\n" + "="*60)
print("TRAINING TIME ESTIMATES")
print("="*60)
print(f"Average time per batch: {avg_time_per_batch:.2f} seconds")
print(f"Batches per epoch: {total_batches_per_epoch}")
print(f"Time per epoch (train only): {time_per_epoch/60:.1f} minutes")
print(f"Time per epoch (train + val): {time_per_epoch_with_val/60:.1f} minutes")
print(f"\nFor {config.NUM_EPOCHS} epochs:")
print(f"  Total time: {total_time_100_epochs/3600:.1f} hours ({total_time_100_epochs/60:.1f} minutes)")
print(f"  Per day (24h): ~{100/(total_time_100_epochs/3600*24)*100:.0f} epochs")
print("\nNote: Actual time may vary based on:")
print("  - Early stopping (may finish earlier)")
print("  - System load")
print("  - I/O performance")
print("="*60)
