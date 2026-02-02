import torch
import torch.nn as nn
import torch.optim as optim
import timm


def get_model(num_classes: int = 1010, pretrained: bool = True):
    """
    Initializes the ConvNeXt-Tiny model using timm.

    Args:
        num_classes (int): Number of target classes. Default is 1010 for iNaturalist 2019.
        pretrained (bool): Whether to load ImageNet-1k pretrained weights.

    Returns:
        torch.nn.Module: The configured model.
    """
    # Load ConvNeXt-Tiny.
    # 'convnext_tiny' corresponds to the ConvNeXt Tiny variant.
    # pretrained=True downloads/loads ImageNet-1k weights.
    # num_classes argument automatically replaces the head.
    model = timm.create_model(
        "convnext_tiny", pretrained=pretrained, num_classes=num_classes
    )

    return model


def get_criterion(label_smoothing: float = 0.1):
    """
    Returns the Cross Entropy Loss with Label Smoothing.

    Args:
        label_smoothing (float): The smoothing factor (default 0.1).

    Returns:
        nn.Module: The loss function.
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def get_optimizer(
    model: nn.Module, learning_rate: float = 1e-4, weight_decay: float = 0.05
):
    """
    Returns the AdamW optimizer.

    Args:
        model (nn.Module): The model to optimize.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay factor (default 0.05).

    Returns:
        optim.Optimizer: The configured optimizer.
    """
    return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def get_scheduler(optimizer: optim.Optimizer, T_max: int, eta_min: float = 1e-6):
    """
    Returns the Cosine Annealing Learning Rate Scheduler.

    Args:
        optimizer (optim.Optimizer): The optimizer.
        T_max (int): Maximum number of iterations (epochs).
        eta_min (float): Minimum learning rate.

    Returns:
        optim.lr_scheduler.LRScheduler: The scheduler.
    """
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
