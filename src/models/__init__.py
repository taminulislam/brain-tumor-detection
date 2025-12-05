"""
Model factory for creating different models
"""

from .unet import get_unet_model
from .resnet_unet import get_resnet_unet_model
from .vit_model import get_vit_model
from .swin_transformer import get_swin_model
from .segformer import get_segformer_model
from .fast_scnn import get_fast_scnn_model
from .bisenetv2 import get_bisenetv2_model
from .segnext import get_segnext_model
from .efficient_mtnet import get_efficient_mtnet_model


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
    elif model_name == 'segformer':
        return get_segformer_model(n_classes_seg=n_classes_seg, n_classes_cls=n_classes_cls, img_size=img_size)
    elif model_name == 'fast_scnn':
        return get_fast_scnn_model(n_classes_seg=n_classes_seg, n_classes_cls=n_classes_cls, img_size=img_size)
    elif model_name == 'bisenetv2':
        return get_bisenetv2_model(n_classes_seg=n_classes_seg, n_classes_cls=n_classes_cls, img_size=img_size)
    elif model_name == 'segnext':
        return get_segnext_model(n_classes_seg=n_classes_seg, n_classes_cls=n_classes_cls, img_size=img_size)
    elif model_name == 'efficient_mtnet':
        return get_efficient_mtnet_model(n_classes_seg=n_classes_seg, n_classes_cls=n_classes_cls, img_size=img_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
