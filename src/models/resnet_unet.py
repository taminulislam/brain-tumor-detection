"""
ResNet-based U-Net encoder-decoder for multi-task learning
Uses pretrained ResNet as encoder
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetUNet(nn.Module):
    """
    ResNet encoder with U-Net decoder for multi-task learning
    """
    def __init__(self, n_channels=3, n_classes_seg=2, n_classes_cls=2, pretrained=True):
        super(ResNetUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes_seg = n_classes_seg
        self.n_classes_cls = n_classes_cls

        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)

        # Encoder layers from ResNet
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1  # 256 channels
        self.encoder2 = resnet.layer2  # 512 channels
        self.encoder3 = resnet.layer3  # 1024 channels
        self.encoder4 = resnet.layer4  # 2048 channels

        # Decoder
        self.decoder4 = self._make_decoder_block(2048, 1024)
        self.decoder3 = self._make_decoder_block(1024, 512)
        self.decoder2 = self._make_decoder_block(512, 256)
        self.decoder1 = self._make_decoder_block(256, 64)

        # Final decoder
        self.final_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Segmentation head
        self.seg_head = nn.Conv2d(32, n_classes_seg, kernel_size=1)

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes_cls)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block with upsampling and convolutions"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x0 = self.initial(x)  # [B, 64, H/2, W/2]
        x_pool = self.maxpool(x0)  # [B, 64, H/4, W/4]

        x1 = self.encoder1(x_pool)  # [B, 256, H/4, W/4]
        x2 = self.encoder2(x1)  # [B, 512, H/8, W/8]
        x3 = self.encoder3(x2)  # [B, 1024, H/16, W/16]
        x4 = self.encoder4(x3)  # [B, 2048, H/32, W/32]

        # Classification from bottleneck
        cls_features = self.pool(x4)
        cls_output = self.classifier(cls_features)

        # Decoder for segmentation
        d4 = self.decoder4(x4)  # [B, 1024, H/16, W/16]
        d3 = self.decoder3(d4)  # [B, 512, H/8, W/8]
        d2 = self.decoder2(d3)  # [B, 256, H/4, W/4]
        d1 = self.decoder1(d2)  # [B, 64, H/2, W/2]

        # Final upsampling
        d0 = self.final_decoder(d1)  # [B, 32, H, W]

        # Segmentation output
        seg_output = self.seg_head(d0)  # [B, n_classes_seg, H, W]

        return cls_output, seg_output


def get_resnet_unet_model(n_channels=3, n_classes_seg=2, n_classes_cls=2, pretrained=True):
    """
    Get ResNet-UNet model for multi-task learning

    Args:
        n_channels: number of input channels
        n_classes_seg: number of segmentation classes
        n_classes_cls: number of classification classes
        pretrained: whether to use pretrained ResNet weights

    Returns:
        ResNetUNet model
    """
    model = ResNetUNet(
        n_channels=n_channels,
        n_classes_seg=n_classes_seg,
        n_classes_cls=n_classes_cls,
        pretrained=pretrained
    )
    return model


if __name__ == "__main__":
    # Test model
    model = get_resnet_unet_model()
    x = torch.randn(2, 3, 640, 640)
    cls_out, seg_out = model(x)
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
