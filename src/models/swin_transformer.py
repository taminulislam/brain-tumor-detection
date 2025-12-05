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
    def __init__(self, model_name='swin_tiny_patch4_window7_224', n_classes_seg=2, n_classes_cls=2,
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
            num_classes=0,
            global_pool=''
        )

        # Get the embedding dimension
        self.embed_dim = self.backbone.num_features
        
        # Calculate patch dimensions
        self.patch_size = 4
        self.num_patches_h = self.num_patches_w = img_size // (self.patch_size * 8)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes_cls)
        )

        # Segmentation decoder
        self.seg_decoder = self._build_decoder()

        # Segmentation head
        self.seg_head = nn.Conv2d(64, n_classes_seg, kernel_size=1)

    def _build_decoder(self):
        """Build decoder for segmentation"""
        decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        return decoder

    def forward(self, x):
        B = x.shape[0]
        
        # Extract features from Swin backbone
        features = self.backbone.forward_features(x)
        
        # Swin returns features in (B, H, W, C) or (B, H*W, C) format
        # Reshape to (B, C, H, W) for convolution layers
        if features.dim() == 3:
            # (B, H*W, C) -> (B, C, H, W)
            features = features.reshape(B, self.num_patches_h, self.num_patches_w, self.embed_dim)
            features = features.permute(0, 3, 1, 2)
        elif features.dim() == 4:
            # (B, H, W, C) -> (B, C, H, W)
            features = features.permute(0, 3, 1, 2)
        
        # For classification: global average pooling
        cls_features = features.view(B, self.embed_dim, -1)
        cls_output = self.cls_head(cls_features)
        
        # For segmentation: decode features
        seg_features = self.seg_decoder(features)
        seg_output = self.seg_head(seg_features)
        
        return cls_output, seg_output


def get_swin_model(model_name='swin_tiny_patch4_window7_224', n_classes_seg=2, n_classes_cls=2,
                   img_size=640, pretrained=True):
    """
    Get Swin Transformer model for multi-task learning

    Args:
        model_name: timm model name (default: swin_tiny for efficiency)
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
