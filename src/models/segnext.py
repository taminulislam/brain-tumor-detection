"""
SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation
Lightweight transformer-based segmentation model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MSCA(nn.Module):
    """Multi-Scale Convolutional Attention"""
    def __init__(self, dim):
        super(MSCA, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SegNeXtBlock(nn.Module):
    """SegNeXt Transformer Block"""
    def __init__(self, dim, mlp_ratio=4., dropout=0.):
        super(SegNeXtBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = MSCA(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, dropout=dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Patch Embedding"""
    def __init__(self, in_channels=3, embed_dim=32, patch_size=4):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class DownSample(nn.Module):
    """Downsampling layer"""
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.norm(self.conv(x))


class SegNeXtEncoder(nn.Module):
    """SegNeXt Encoder"""
    def __init__(self, in_channels=3, embed_dims=[32, 64, 128, 256], depths=[2, 2, 2, 2]):
        super(SegNeXtEncoder, self).__init__()
        
        self.patch_embed = PatchEmbed(in_channels, embed_dims[0], patch_size=4)
        
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            if i == 0:
                stage = nn.Sequential(*[
                    SegNeXtBlock(embed_dims[i], mlp_ratio=4., dropout=0.1)
                    for _ in range(depths[i])
                ])
            else:
                downsample = DownSample(embed_dims[i-1], embed_dims[i])
                blocks = nn.Sequential(*[
                    SegNeXtBlock(embed_dims[i], mlp_ratio=4., dropout=0.1)
                    for _ in range(depths[i])
                ])
                stage = nn.Sequential(downsample, blocks)
            
            self.stages.append(stage)
        
        self.norm = nn.BatchNorm2d(embed_dims[-1])
    
    def forward(self, x):
        x = self.patch_embed(x)
        
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        x = self.norm(x)
        return x, features


class SegNeXtDecoder(nn.Module):
    """Simple decoder for segmentation"""
    def __init__(self, encoder_dims=[32, 64, 128, 256], decoder_dim=128):
        super(SegNeXtDecoder, self).__init__()
        
        self.conv1 = ConvBNReLU(encoder_dims[-1], decoder_dim, kernel_size=1, stride=1, padding=0)
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(decoder_dim, decoder_dim, kernel_size=3, stride=1, padding=1)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(decoder_dim, 128, kernel_size=3, stride=1, padding=1)
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(128, 64, kernel_size=3, stride=1, padding=1)
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        )
        
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        )
        
        self.final_conv = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.final_conv(x)
        return x


class SegNeXtMultiTask(nn.Module):
    """
    SegNeXt with multi-task heads for classification and segmentation
    """
    def __init__(self, n_classes_seg=2, n_classes_cls=2, img_size=640):
        super(SegNeXtMultiTask, self).__init__()
        self.img_size = img_size
        
        embed_dims = [32, 64, 128, 256]
        depths = [2, 2, 2, 2]
        
        self.encoder = SegNeXtEncoder(3, embed_dims, depths)
        self.decoder = SegNeXtDecoder(embed_dims, 128)
        
        # Classification head
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(embed_dims[-1]),
            nn.Linear(embed_dims[-1], 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes_cls)
        )
        
        # Segmentation head
        self.seg_head = nn.Conv2d(64, n_classes_seg, kernel_size=1)
    
    def forward(self, x):
        encoder_out, features = self.encoder(x)
        
        # Classification
        cls_features = self.cls_pool(encoder_out)
        cls_output = self.cls_head(cls_features)
        
        # Segmentation
        seg_features = self.decoder(encoder_out)
        seg_output = self.seg_head(seg_features)
        
        return cls_output, seg_output


def get_segnext_model(n_classes_seg=2, n_classes_cls=2, img_size=640):
    """
    Get SegNeXt model for multi-task learning
    
    Args:
        n_classes_seg: number of segmentation classes
        n_classes_cls: number of classification classes
        img_size: input image size
    
    Returns:
        SegNeXtMultiTask model
    """
    model = SegNeXtMultiTask(
        n_classes_seg=n_classes_seg,
        n_classes_cls=n_classes_cls,
        img_size=img_size
    )
    return model


if __name__ == "__main__":
    model = get_segnext_model(img_size=640)
    x = torch.randn(2, 3, 640, 640)
    cls_out, seg_out = model(x)
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

