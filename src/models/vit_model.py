"""
Vision Transformer (ViT) based model for multi-task learning
Uses timm library for pretrained ViT backbone
"""

import torch
import torch.nn as nn
import timm


class ViTMultiTask(nn.Module):
    """
    Vision Transformer with multi-task heads for classification and segmentation
    """
    def __init__(self, model_name='vit_base_patch16_224', n_classes_seg=2, n_classes_cls=2,
                 img_size=640, pretrained=True):
        super(ViTMultiTask, self).__init__()
        self.img_size = img_size
        self.n_classes_seg = n_classes_seg
        self.n_classes_cls = n_classes_cls

        # Load pretrained ViT backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )

        # Get feature dimension from backbone
        self.feature_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_patches = (img_size // self.patch_size) ** 2

        # Classification head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
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
        patch_h = patch_w = self.img_size // self.patch_size

        decoder = nn.Sequential(
            # Reshape and initial conv
            nn.ConvTranspose2d(self.feature_dim, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Upsample blocks
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        return decoder

    def forward(self, x):
        B = x.shape[0]

        # Extract features from ViT backbone
        features = self.backbone(x)  # [B, num_patches+1, feature_dim]

        # Classification: use CLS token (first token)
        cls_token = features[:, 0]  # [B, feature_dim]
        cls_output = self.cls_head(cls_token)

        # Segmentation: use patch tokens (excluding CLS token)
        patch_tokens = features[:, 1:]  # [B, num_patches, feature_dim]

        # Reshape to spatial dimensions
        patch_h = patch_w = self.img_size // self.patch_size
        patch_features = patch_tokens.transpose(1, 2).reshape(B, self.feature_dim, patch_h, patch_w)

        # Decode to segmentation mask
        seg_features = self.seg_decoder(patch_features)
        seg_output = self.seg_head(seg_features)

        return cls_output, seg_output


def get_vit_model(model_name='vit_base_patch16_224', n_classes_seg=2, n_classes_cls=2,
                  img_size=640, pretrained=True):
    """
    Get Vision Transformer model for multi-task learning

    Args:
        model_name: timm model name
        n_classes_seg: number of segmentation classes
        n_classes_cls: number of classification classes
        img_size: input image size
        pretrained: whether to use pretrained weights

    Returns:
        ViTMultiTask model
    """
    model = ViTMultiTask(
        model_name=model_name,
        n_classes_seg=n_classes_seg,
        n_classes_cls=n_classes_cls,
        img_size=img_size,
        pretrained=pretrained
    )
    return model


if __name__ == "__main__":
    # Test model
    model = get_vit_model(img_size=640)
    x = torch.randn(2, 3, 640, 640)
    cls_out, seg_out = model(x)
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
