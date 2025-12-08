"""
Demo script to test EfficientMTNet model
Quick verification that the model can be instantiated
"""

import torch
from model import get_efficient_mtnet_model, EfficientMTNet

def main():
    print("=" * 60)
    print("EfficientMTNet Model Demo")
    print("=" * 60)

    # Model configuration
    n_classes_seg = 2  # Binary segmentation
    n_classes_cls = 2  # Binary classification
    img_size = 640
    use_aux = True

    print(f"\nModel Configuration:")
    print(f"  Segmentation Classes: {n_classes_seg}")
    print(f"  Classification Classes: {n_classes_cls}")
    print(f"  Input Size: {img_size}x{img_size}")
    print(f"  Auxiliary Heads: {use_aux}")

    # Initialize model
    print("\n[1] Initializing model...")
    model = get_efficient_mtnet_model(
        n_classes_seg=n_classes_seg,
        n_classes_cls=n_classes_cls,
        img_size=img_size,
        use_aux=use_aux
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"    ✓ Model initialized successfully")
    print(f"    Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"    Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # Test forward pass - Training mode
    print("\n[2] Testing forward pass (training mode)...")
    model.train()
    x = torch.randn(2, 3, img_size, img_size)
    print(f"    Input shape: {x.shape}")

    with torch.no_grad():
        outputs = model(x)
        cls_out, seg_out, aux1, aux2 = outputs

    print(f"    ✓ Forward pass successful")
    print(f"    Classification output: {cls_out.shape}")
    print(f"    Segmentation output: {seg_out.shape}")
    print(f"    Auxiliary output 1: {aux1.shape}")
    print(f"    Auxiliary output 2: {aux2.shape}")

    # Test forward pass - Inference mode
    print("\n[3] Testing forward pass (inference mode)...")
    model.eval()

    with torch.no_grad():
        outputs = model(x)
        cls_out, seg_out = outputs

    print(f"    ✓ Forward pass successful")
    print(f"    Classification output: {cls_out.shape}")
    print(f"    Segmentation output: {seg_out.shape}")

    # Model components
    print("\n[4] Model Architecture Components:")
    print("    ✓ Enhanced Learning-to-Downsample")
    print("    ✓ Global Feature Extractor (3 stages)")
    print("    ✓ Multi-Scale Feature Neck")
    print("    ✓ LightHam Attention")
    print("    ✓ Classification Head")
    print("    ✓ Segmentation Head")
    print("    ✓ Auxiliary Heads (2x)")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Configure dataset paths in config.py")
    print("  2. Run: python train.py")
    print("  3. Monitor: tensorboard --logdir logs/")
    print("  4. Test: python test.py")

if __name__ == "__main__":
    main()
