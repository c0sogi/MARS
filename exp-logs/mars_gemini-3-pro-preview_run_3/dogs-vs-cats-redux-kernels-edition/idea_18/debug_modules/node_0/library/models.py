import timm
import torch.nn as nn


def get_model(
    model_name: str, pretrained: bool = True, num_classes: int = 1
) -> nn.Module:
    """
    Creates a model architecture using the timm library.

    This function initializes a model with the specified architecture name.
    It supports loading pre-trained weights (e.g., ImageNet) and modifying
    the final classification head to match the required number of classes
    (default is 1 for binary classification with BCEWithLogitsLoss).

    Args:
        model_name (str): The specific timm model name (e.g., 'resnet50.a1_in1k',
                          'convnext_small.fb_in1k', 'maxvit_tiny_tf_224.in1k').
        pretrained (bool): If True, loads the pretrained weights specified by the model_name tags.
                           Defaults to True.
        num_classes (int): The number of output units in the final layer.
                           Defaults to 1 for binary classification.

    Returns:
        nn.Module: The instantiated PyTorch model ready for training or inference.
    """
    try:
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to create model '{model_name}': {e}")
