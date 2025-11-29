"""
Model factory for creating different models
"""

from .unet import get_unet_model
from .resnet_unet import get_resnet_unet_model
from .vit_model import get_vit_model
from .swin_transformer import get_swin_model
from .lightweight_transformer import get_lightweight_transformer_model


def get_model(model_name, n_classes_seg=2, n_classes_cls=2, img_size=640):
    """
    Factory function to get model by name

    Args:
        model_name: name of the model
        n_classes_seg: number of segmentation classes
        n_classes_cls: number of classification classes
        img_size: input image size

    Returns:
        model instance
    """
    if model_name == 'unet':
        return get_unet_model(n_classes_seg=n_classes_seg, n_classes_cls=n_classes_cls)
    elif model_name == 'resnet_unet':
        return get_resnet_unet_model(n_classes_seg=n_classes_seg, n_classes_cls=n_classes_cls)
    elif model_name == 'vit':
        return get_vit_model(n_classes_seg=n_classes_seg, n_classes_cls=n_classes_cls, img_size=img_size)
    elif model_name == 'swin':
        return get_swin_model(n_classes_seg=n_classes_seg, n_classes_cls=n_classes_cls, img_size=img_size)
    elif model_name == 'lightweight_transformer':
        return get_lightweight_transformer_model(n_classes_seg=n_classes_seg, n_classes_cls=n_classes_cls, img_size=img_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
