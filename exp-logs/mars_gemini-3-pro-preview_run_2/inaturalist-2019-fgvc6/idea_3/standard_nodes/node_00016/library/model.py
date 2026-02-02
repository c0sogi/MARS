import timm
import torch.nn as nn
from library.config import CFG


def build_model():
    """
    Constructs the EfficientNetV2-Small model using timm.

    The model is initialized with pretrained weights on ImageNet.
    The classification head is adjusted to match the number of classes in the dataset (1010).
    Stochastic depth (DropPath) is applied for regularization.

    Returns:
        model (nn.Module): The PyTorch model.
    """
    model = timm.create_model(
        CFG.model_name,
        pretrained=CFG.pretrained,
        num_classes=CFG.num_classes,
        drop_path_rate=CFG.drop_path_rate,
    )

    return model
