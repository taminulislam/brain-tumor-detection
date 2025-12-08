
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConvSE(nn.Module):
    """Depthwise separable convolution with SE block"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(DepthwiseSeparableConvSE, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x


class InvertedResidualSE(nn.Module):
    """Inverted residual block with SE attention"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio, use_se=True):
        super(InvertedResidualSE, self).__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        if self.use_res_connect:
            return x + out
        return out


class EnhancedLearningToDownsample(nn.Module):
    """Enhanced learning to downsample with SE blocks"""
    def __init__(self, in_channels=3, out_channels=64):
        super(EnhancedLearningToDownsample, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dsconv1 = DepthwiseSeparableConvSE(32, 48, stride=2, use_se=True)
        self.dsconv2 = DepthwiseSeparableConvSE(48, out_channels, stride=2, use_se=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class LightHamModule(nn.Module):
    """
    Light Hamburger Module - Efficient global context modeling
    Uses simplified attention for lightweight global context
    """
    def __init__(self, channels, ham_channels=64, num_bases=4):
        super(LightHamModule, self).__init__()
        self.channels = channels
        self.ham_channels = ham_channels
        
        self.query = nn.Conv2d(channels, ham_channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, ham_channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, ham_channels, kernel_size=1, bias=False)
        
        self.out_proj = nn.Sequential(
            nn.Conv2d(ham_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        q = self.query(x).view(B, self.ham_channels, -1)
        k = self.key(x).view(B, self.ham_channels, -1)
        v = self.value(x).view(B, self.ham_channels, -1)
        
        # Efficient attention
        attn = torch.bmm(q.transpose(1, 2), k)
        attn = F.softmax(attn / (self.ham_channels ** 0.5), dim=-1)
        
        context = torch.bmm(v, attn.transpose(1, 2))
        context = context.view(B, self.ham_channels, H, W)
        
        out = self.out_proj(context)
        
        return x + self.gamma * out


class MultiScaleFeatureNeck(nn.Module):
    """Lightweight multi-scale feature aggregation neck"""
    def __init__(self, in_channels_list, out_channels):
        super(MultiScaleFeatureNeck, self).__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for in_ch in in_channels_list
        ])
        
        self.fpn_conv = nn.Sequential(
            DepthwiseSeparableConvSE(out_channels * len(in_channels_list), out_channels, use_se=True)
        )
        
        self.attention = LightHamModule(out_channels, ham_channels=32, num_bases=4)
    
    def forward(self, features, target_size):
        laterals = []
        for i, feat in enumerate(features):
            lateral = self.lateral_convs[i](feat)
            lateral = F.interpolate(lateral, size=target_size, mode='bilinear', align_corners=True)
            laterals.append(lateral)
        
        fused = torch.cat(laterals, dim=1)
        fused = self.fpn_conv(fused)
        fused = self.attention(fused)
        
        return fused


class GlobalFeatureExtractorPlus(nn.Module):
    """Enhanced global feature extractor with multi-scale outputs"""
    def __init__(self, in_channels=64, out_channels=128):
        super(GlobalFeatureExtractorPlus, self).__init__()
        
        self.stage1 = nn.Sequential(
            InvertedResidualSE(in_channels, 64, stride=2, expand_ratio=6, use_se=True),
            InvertedResidualSE(64, 64, stride=1, expand_ratio=6, use_se=False),
            InvertedResidualSE(64, 64, stride=1, expand_ratio=6, use_se=True)
        )
        
        self.stage2 = nn.Sequential(
            InvertedResidualSE(64, 96, stride=2, expand_ratio=6, use_se=True),
            InvertedResidualSE(96, 96, stride=1, expand_ratio=6, use_se=False),
            InvertedResidualSE(96, 96, stride=1, expand_ratio=6, use_se=True)
        )
        
        self.stage3 = nn.Sequential(
            InvertedResidualSE(96, out_channels, stride=1, expand_ratio=6, use_se=True),
            InvertedResidualSE(out_channels, out_channels, stride=1, expand_ratio=6, use_se=False),
            InvertedResidualSE(out_channels, out_channels, stride=1, expand_ratio=6, use_se=True)
        )
    
    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3


class AuxiliaryHead(nn.Module):
    """Auxiliary head for deep supervision"""
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super(AuxiliaryHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        )
    
    def forward(self, x, target_size):
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return x


class FeatureFusionPlus(nn.Module):
    """Enhanced feature fusion with attention"""
    def __init__(self, high_channels, low_channels, out_channels):
        super(FeatureFusionPlus, self).__init__()
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = DepthwiseSeparableConvSE(out_channels, out_channels, stride=1, use_se=True)
    
    def forward(self, high_res, low_res):
        low_res = F.interpolate(low_res, size=high_res.shape[2:], mode='bilinear', align_corners=True)
        low_res = self.conv_low(low_res)
        high_res = self.conv_high(high_res)
        
        combined = torch.cat([high_res, low_res], dim=1)
        attn = self.attention(combined)
        
        x = high_res * attn + low_res * (1 - attn)
        x = self.out_conv(x)
        return x


class EfficientMTNet(nn.Module):
    def __init__(self, n_classes_seg=2, n_classes_cls=2, img_size=640, use_aux=True):
        super(EfficientMTNet, self).__init__()
        self.img_size = img_size
        self.use_aux = use_aux
        
        # Encoder
        self.learning_to_downsample = EnhancedLearningToDownsample(3, 64)
        self.global_feature_extractor = GlobalFeatureExtractorPlus(64, 128)
        
        # Neck - Multi-scale feature aggregation
        self.neck = MultiScaleFeatureNeck([64, 96, 128], out_channels=128)
        
        # Feature fusion (low_level=64ch, neck_out=128ch)
        self.feature_fusion = FeatureFusionPlus(64, 128, 128)
        
        # Classification head with global context
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_attention = LightHamModule(128, ham_channels=32, num_bases=4)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes_cls)
        )
        
        # Segmentation decoder
        self.seg_decoder = nn.Sequential(
            DepthwiseSeparableConvSE(128, 128, stride=1, use_se=True),
            DepthwiseSeparableConvSE(128, 64, stride=1, use_se=True),
            nn.Dropout(0.1)
        )
        
        self.seg_head = nn.Conv2d(64, n_classes_seg, kernel_size=1)
        
        # Auxiliary heads for deep supervision
        if use_aux:
            self.aux_head1 = AuxiliaryHead(64, n_classes_seg)
            self.aux_head2 = AuxiliaryHead(96, n_classes_seg)
    
    def forward(self, x):
        target_size = (self.img_size, self.img_size)
        
        # Encoder
        low_level = self.learning_to_downsample(x)
        f1, f2, f3 = self.global_feature_extractor(low_level)
        
        # Multi-scale neck
        neck_out = self.neck([f1, f2, f3], low_level.shape[2:])
        
        # Feature fusion
        fused = self.feature_fusion(low_level, neck_out)
        
        # Classification branch
        cls_features = self.cls_attention(f3)
        cls_features = self.cls_pool(cls_features)
        cls_output = self.cls_head(cls_features)
        
        # Segmentation branch
        seg_features = self.seg_decoder(fused)
        seg_output = self.seg_head(seg_features)
        seg_output = F.interpolate(seg_output, size=target_size, mode='bilinear', align_corners=True)
        
        if self.training and self.use_aux:
            aux1 = self.aux_head1(f1, target_size)
            aux2 = self.aux_head2(f2, target_size)
            return cls_output, seg_output, aux1, aux2
        
        return cls_output, seg_output


def get_efficient_mtnet_model(n_classes_seg=2, n_classes_cls=2, img_size=640, use_aux=True):
    """
    Get EfficientMTNet model for multi-task learning
    
    Args:
        n_classes_seg: number of segmentation classes
        n_classes_cls: number of classification classes
        img_size: input image size
        use_aux: use auxiliary heads for deep supervision
    
    Returns:
        EfficientMTNet model
    """
    model = EfficientMTNet(
        n_classes_seg=n_classes_seg,
        n_classes_cls=n_classes_cls,
        img_size=img_size,
        use_aux=use_aux
    )
    return model


if __name__ == "__main__":
    model = get_efficient_mtnet_model(img_size=640, use_aux=True)
    model.eval()
    x = torch.randn(2, 3, 640, 640)
    cls_out, seg_out = model(x)
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Segmentation output shape: {seg_out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Training mode test
    model.train()
    cls_out, seg_out, aux1, aux2 = model(x)
    print(f"\nTraining mode:")
    print(f"  Main seg output: {seg_out.shape}")
    print(f"  Aux1 output: {aux1.shape}")
    print(f"  Aux2 output: {aux2.shape}")

