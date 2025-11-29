"""
Swin Transformer based model for multi-task learning
Uses timm library for pretrained Swin Transformer backbone
"""

import torch
import torch.nn as nn
import timm


class SwinMultiTask(nn.Module):
    """
    Swin Transformer with multi-task heads for classification and segmentation
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', n_classes_seg=2, n_classes_cls=2,
                 img_size=640, pretrained=True):
        super(SwinMultiTask, self).__init__()
        self.img_size = img_size
        self.n_classes_seg = n_classes_seg
        self.n_classes_cls = n_classes_cls

        # Load pretrained Swin Transformer backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )

        # Get feature dimensions
        # For Swin, we need to extract features at different scales
        self.feature_info = self.backbone.feature_info

        # Classification head
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        final_feature_dim = self.feature_info[-1]['num_chs']

        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(final_feature_dim),
            nn.Linear(final_feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes_cls)
        )

        # Segmentation decoder with skip connections
        self.seg_decoder = self._build_decoder()

        # Segmentation head
        self.seg_head = nn.Conv2d(64, n_classes_seg, kernel_size=1)

    def _build_decoder(self):
        """Build FPN-like decoder for segmentation"""
        feature_dims = [info['num_chs'] for info in self.feature_info]

        decoder = nn.ModuleDict({
            'up4': nn.Sequential(
                nn.ConvTranspose2d(feature_dims[-1], 256, kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'up3': nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            'up2': nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            'up1': nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            'final': nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        })

        return decoder

    def forward(self, x):
        # Extract multi-scale features from Swin backbone
        features = self.backbone.forward_features(x)  # Returns final features

        # For classification: use global average pooling on final features
        cls_features = self.cls_pool(features)
        cls_output = self.cls_head(cls_features)

        # For segmentation: decode features
        x = features

        x = self.seg_decoder['up4'](x)
        x = self.seg_decoder['up3'](x)
        x = self.seg_decoder['up2'](x)
        x = self.seg_decoder['up1'](x)
        x = self.seg_decoder['final'](x)

        seg_output = self.seg_head(x)

        return cls_output, seg_output


def get_swin_model(model_name='swin_base_patch4_window7_224', n_classes_seg=2, n_classes_cls=2,
                   img_size=640, pretrained=True):
    """
    Get Swin Transformer model for multi-task learning

    Args:
        model_name: timm model name
        n_classes_seg: number of segmentation classes
        n_classes_cls: number of classification classes
        img_size: input image size
        pretrained: whether to use pretrained weights

    Returns:
        SwinMultiTask model
    """
    model = SwinMultiTask(
        model_name=model_name,
        n_classes_seg=n_classes_seg,
        n_classes_cls=n_classes_cls,
        img_size=img_size,
        pretrained=pretrained
    )
    return model


if __name__ == "__main__":
    # Test model
    model = get_swin_model(img_size=640)
    x = torch.randn(2, 3, 640, 640)
    cls_out, seg_out = model(x)
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
