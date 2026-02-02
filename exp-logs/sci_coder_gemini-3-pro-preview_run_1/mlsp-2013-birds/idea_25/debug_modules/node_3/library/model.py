import torch
import torch.nn as nn
import timm
from library.utils import set_seed


def create_bird_model(num_classes=19, pretrained=True):
    """
    Constructs the Anti-Aliased ResNet-34d model for bird species classification.

    This function utilizes the `timm` library to create a ResNet-34d model.
    - 'resnet34d': Uses a "Deep Stem" (3 consecutive 3x3 convolutions) instead of the
      standard 7x7 convolution + max pooling. This preserves fine-grained spectrotemporal
      details critical for audio classification.
    - antialiased=True: Replaces standard downsampling layers with BlurPool operations,
      providing shift invariance and robustness to small temporal jitters.

    Args:
        num_classes (int): The number of output classes (bird species). Default is 19.
        pretrained (bool): If True, loads weights pretrained on ImageNet. Default is True.

    Returns:
        model (torch.nn.Module): The initialized PyTorch model.
    """

    # If training from scratch, ensure deterministic initialization
    if not pretrained:
        set_seed(42)

    # Create the model
    # in_chans=3 matches the channel replication performed in the Dataset class.
    model = timm.create_model(
        "resnet34d",
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3,
    )

    return model
