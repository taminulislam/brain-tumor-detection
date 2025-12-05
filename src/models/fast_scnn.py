"""
Fast-SCNN: Fast Semantic Segmentation Network
Lightweight model for real-time segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block from MobileNetV2"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""
    def __init__(self, in_channels=3, out_channels=64):
        super(LearningToDownsample, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dsconv1 = DepthwiseSeparableConv(32, 48, stride=2)
        self.dsconv2 = DepthwiseSeparableConv(48, out_channels, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor"""
    def __init__(self, in_channels=64, out_channels=128):
        super(GlobalFeatureExtractor, self).__init__()
        self.block1 = nn.Sequential(
            InvertedResidual(in_channels, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6)
        )
        self.block2 = nn.Sequential(
            InvertedResidual(64, 96, stride=2, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6)
        )
        self.block3 = nn.Sequential(
            InvertedResidual(96, out_channels, stride=1, expand_ratio=6),
            InvertedResidual(out_channels, out_channels, stride=1, expand_ratio=6),
            InvertedResidual(out_channels, out_channels, stride=1, expand_ratio=6)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class FeatureFusion(nn.Module):
    """Feature fusion module"""
    def __init__(self, high_channels, low_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.dwconv = DepthwiseSeparableConv(low_channels, out_channels, stride=1)
        self.conv_low_res = nn.Conv2d(high_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, high_res, low_res):
        low_res = F.interpolate(low_res, size=high_res.shape[2:], mode='bilinear', align_corners=True)
        low_res = self.conv_low_res(low_res)
        high_res = self.dwconv(high_res)
        x = high_res + low_res
        x = self.bn(x)
        x = self.relu(x)
        return x


class FastSCNNMultiTask(nn.Module):
    """
    Fast-SCNN with multi-task heads for classification and segmentation
    """
    def __init__(self, n_classes_seg=2, n_classes_cls=2, img_size=640):
        super(FastSCNNMultiTask, self).__init__()
        self.img_size = img_size
        
        self.learning_to_downsample = LearningToDownsample(3, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, 128)
        self.feature_fusion = FeatureFusion(128, 64, 128)
        
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
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 64, stride=1),
            nn.Dropout(0.1)
        )
        
        self.seg_head = nn.Conv2d(64, n_classes_seg, kernel_size=1)
    
    def forward(self, x):
        low_level = self.learning_to_downsample(x)
        global_features = self.global_feature_extractor(low_level)
        
        fused = self.feature_fusion(low_level, global_features)
        
        # Classification
        cls_features = self.cls_pool(global_features)
        cls_output = self.cls_head(cls_features)
        
        # Segmentation
        seg_features = self.seg_decoder(fused)
        seg_output = self.seg_head(seg_features)
        seg_output = F.interpolate(seg_output, size=(self.img_size, self.img_size), 
                                   mode='bilinear', align_corners=True)
        
        return cls_output, seg_output


def get_fast_scnn_model(n_classes_seg=2, n_classes_cls=2, img_size=640):
    """
    Get Fast-SCNN model for multi-task learning
    
    Args:
        n_classes_seg: number of segmentation classes
        n_classes_cls: number of classification classes
        img_size: input image size
    
    Returns:
        FastSCNNMultiTask model
    """
    model = FastSCNNMultiTask(
        n_classes_seg=n_classes_seg,
        n_classes_cls=n_classes_cls,
        img_size=img_size
    )
    return model


if __name__ == "__main__":
    model = get_fast_scnn_model(img_size=640)
    x = torch.randn(2, 3, 640, 640)
    cls_out, seg_out = model(x)
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

