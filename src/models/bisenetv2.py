"""
BiSeNetV2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation
Lightweight and efficient model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Basic conv + bn + relu block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class StemBlock(nn.Module):
    """Stem block for detail branch"""
    def __init__(self, in_channels=3, out_channels=16):
        super(StemBlock, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(out_channels, out_channels//2, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(out_channels//2, out_channels, kernel_size=3, stride=2)
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_channels*2, out_channels, kernel_size=3, stride=1)
    
    def forward(self, x):
        x = self.conv1(x)
        left = self.left(x)
        right = self.right(x)
        x = torch.cat([left, right], dim=1)
        x = self.fuse(x)
        return x


class GELayer(nn.Module):
    """Gather-and-Expansion Layer"""
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super(GELayer, self).__init__()
        mid_channels = in_channels * expand_ratio
        self.conv1 = ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=1)
        
        if stride == 1:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, 
                         padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, 
                         padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, 
                         padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        
        x = x + shortcut
        x = self.relu(x)
        return x


class DetailBranch(nn.Module):
    """Detail branch for capturing spatial details"""
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.stem = StemBlock(3, 16)
        self.stage1 = nn.Sequential(
            ConvBNReLU(16, 32, kernel_size=3, stride=2),
            ConvBNReLU(32, 32, kernel_size=3, stride=1),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(32, 64, kernel_size=3, stride=2),
            ConvBNReLU(64, 64, kernel_size=3, stride=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x


class SemanticBranch(nn.Module):
    """Semantic branch for context"""
    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.stem = StemBlock(3, 16)
        self.stage3 = nn.Sequential(
            GELayer(16, 32, expand_ratio=6, stride=2),
            GELayer(32, 32, expand_ratio=6, stride=1)
        )
        self.stage4 = nn.Sequential(
            GELayer(32, 64, expand_ratio=6, stride=2),
            GELayer(64, 64, expand_ratio=6, stride=1)
        )
        self.stage5 = nn.Sequential(
            GELayer(64, 128, expand_ratio=6, stride=2),
            GELayer(128, 128, expand_ratio=6, stride=1),
            GELayer(128, 128, expand_ratio=6, stride=1),
            GELayer(128, 128, expand_ratio=6, stride=1)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x


class AggregationLayer(nn.Module):
    """Bilateral Guided Aggregation"""
    def __init__(self, detail_channels=64, semantic_channels=128, out_channels=128):
        super(AggregationLayer, self).__init__()
        self.detail_branch = nn.Sequential(
            nn.Conv2d(detail_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.semantic_branch = nn.Sequential(
            nn.Conv2d(semantic_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1)
    
    def forward(self, detail, semantic):
        detail = self.detail_branch(detail)
        semantic = F.interpolate(semantic, size=detail.shape[2:], mode='bilinear', align_corners=True)
        semantic = self.semantic_branch(semantic)
        x = detail + semantic
        x = self.conv(x)
        return x


class BiSeNetV2MultiTask(nn.Module):
    """
    BiSeNetV2 with multi-task heads for classification and segmentation
    """
    def __init__(self, n_classes_seg=2, n_classes_cls=2, img_size=640):
        super(BiSeNetV2MultiTask, self).__init__()
        self.img_size = img_size
        
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        self.aggregation = AggregationLayer(64, 128, 128)
        
        # Classification head
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes_cls)
        )
        
        # Segmentation decoder
        self.seg_decoder = nn.Sequential(
            ConvBNReLU(128, 64, kernel_size=3, stride=1),
            nn.Dropout(0.1)
        )
        
        self.seg_head = nn.Conv2d(64, n_classes_seg, kernel_size=1)
    
    def forward(self, x):
        detail = self.detail_branch(x)
        semantic = self.semantic_branch(x)
        
        fused = self.aggregation(detail, semantic)
        
        # Classification
        cls_features = self.cls_pool(semantic)
        cls_output = self.cls_head(cls_features)
        
        # Segmentation
        seg_features = self.seg_decoder(fused)
        seg_output = self.seg_head(seg_features)
        seg_output = F.interpolate(seg_output, size=(self.img_size, self.img_size), 
                                   mode='bilinear', align_corners=True)
        
        return cls_output, seg_output


def get_bisenetv2_model(n_classes_seg=2, n_classes_cls=2, img_size=640):
    """
    Get BiSeNetV2 model for multi-task learning
    
    Args:
        n_classes_seg: number of segmentation classes
        n_classes_cls: number of classification classes
        img_size: input image size
    
    Returns:
        BiSeNetV2MultiTask model
    """
    model = BiSeNetV2MultiTask(
        n_classes_seg=n_classes_seg,
        n_classes_cls=n_classes_cls,
        img_size=img_size
    )
    return model


if __name__ == "__main__":
    model = get_bisenetv2_model(img_size=640)
    x = torch.randn(2, 3, 640, 640)
    cls_out, seg_out = model(x)
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

