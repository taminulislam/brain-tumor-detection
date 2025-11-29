"""
U-Net baseline model for multi-task learning
Includes both segmentation and classification heads
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net with multi-task heads for classification and segmentation
    """
    def __init__(self, n_channels=3, n_classes_seg=2, n_classes_cls=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes_seg = n_classes_seg
        self.n_classes_cls = n_classes_cls

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder for segmentation
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Segmentation head
        self.seg_head = nn.Conv2d(64, n_classes_seg, kernel_size=1)

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes_cls)
        )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Classification from bottleneck
        cls_features = self.pool(x5)
        cls_output = self.classifier(cls_features)

        # Decoder for segmentation
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_output = self.seg_head(x)

        return cls_output, seg_output


def get_unet_model(n_channels=3, n_classes_seg=2, n_classes_cls=2):
    """
    Get U-Net model for multi-task learning

    Args:
        n_channels: number of input channels
        n_classes_seg: number of segmentation classes
        n_classes_cls: number of classification classes

    Returns:
        UNet model
    """
    model = UNet(n_channels=n_channels, n_classes_seg=n_classes_seg, n_classes_cls=n_classes_cls)
    return model


if __name__ == "__main__":
    # Test model
    model = get_unet_model()
    x = torch.randn(2, 3, 640, 640)
    cls_out, seg_out = model(x)
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
