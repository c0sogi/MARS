import torch
import torch.nn as nn
import timm
from library.utils import set_seed


def create_model(num_classes=19, pretrained=True, drop_path_rate=0.0, head_dropout=0.0):
    """
    Constructs a ResNet-34 model with optional Stochastic Depth and Head Dropout
    designed for Noisy Student distillation.

    Args:
        num_classes (int): Number of target classes (19 for this dataset).
        pretrained (bool): If True, initializes with ImageNet weights.
        drop_path_rate (float): Probability of dropping residual paths (Stochastic Depth).
                                Recommended: 0.1 for Student, 0.0 for Teacher.
        head_dropout (float): Dropout probability before the final classification layer.
                              Recommended: 0.5 for Student, 0.0 for Teacher.

    Returns:
        nn.Module: The constructed PyTorch model.
    """
    # Initialize the ResNet-34 backbone using timm
    # drop_path_rate handles the Stochastic Depth regularization
    # in_chans=3 expects RGB inputs (spectrograms should be replicated to 3 channels)
    model = timm.create_model(
        "resnet34",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        in_chans=3,
        global_pool="avg",
    )

    # Apply Head Dropout if specified
    # We explicitly replace the fully connected layer to ensure the structure:
    # Pooling -> Flatten -> Dropout -> Linear
    # This serves as the 'noise' injection for the Student model's head.
    if head_dropout > 0.0:
        # In timm's ResNet implementation, the final layer is named 'fc'
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=head_dropout), nn.Linear(in_features, num_classes)
        )

    return model
