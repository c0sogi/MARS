import torch
import torch.nn as nn
import timm


def get_seresnet_model(num_classes=19, pretrained=True, device="cuda"):
    """
    Initializes the ResNet-34 model from timm with ImageNet weights.

    Note: Switched from SE-ResNet-34 to ResNet-34 as the former lacks default
    pretrained weights in the current timm version.

    Args:
        num_classes (int): Number of output classes (default 19 for this dataset).
        pretrained (bool): Whether to load pretrained ImageNet weights.
        device (str): Device to move the model to ('cpu' or 'cuda').

    Returns:
        model (torch.nn.Module): The initialized SE-ResNet-34 model.
    """
    # Initialize the model using timm
    # 'seresnet34' corresponds to the SE-ResNet-34 architecture.
    # We use global_pool='avg' to ensure a standard Global Average Pooling
    # followed by a Linear layer, avoiding complex heads.
    # Cite {debug_lesson_6}: Switched to 'resnet34' to ensure pretrained weights are available.
    model = timm.create_model(
        "resnet34", pretrained=pretrained, num_classes=num_classes, global_pool="avg"
    )

    # Move the model to the specified device
    if device:
        model = model.to(device)

    return model
