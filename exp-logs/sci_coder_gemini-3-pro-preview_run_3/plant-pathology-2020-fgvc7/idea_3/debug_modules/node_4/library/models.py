import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleDiseaseModel(nn.Module):
    """
    A PyTorch model wrapper for Apple Leaf Disease Detection.
    Uses the timm library to create backbones (EfficientNet, ConvNeXt)
    and adapts the classification head for the specific number of classes.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Args:
            model_name (str): The name of the architecture to use (e.g., 'efficientnet_b3').
            num_classes (int): The number of output classes.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(AppleDiseaseModel, self).__init__()

        # Instantiate the model using timm.
        # passing num_classes tells timm to replace the original head
        # with a new one having the correct number of outputs.

        # Cite debug_lesson_2: Query Library Registries to Resolve Invalid Identifier Errors
        if model_name not in timm.list_models():
            matches = timm.list_models(f"*{model_name}*")
            if matches:
                # Prefer 'in1k' models if available to match standard ImageNet pretraining
                preferred = [m for m in matches if "in1k" in m]
                new_name = preferred[0] if preferred else matches[0]
                print(
                    f"Warning: Model '{model_name}' not found. Using '{new_name}' instead."
                )
                model_name = new_name

        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        return self.model(x)


def create_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=True):
    """
    Factory function to create an instance of the AppleDiseaseModel.

    Args:
        model_name (str): The name of the model architecture.
        num_classes (int, optional): Number of target classes. Defaults to Config.NUM_CLASSES.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.

    Returns:
        torch.nn.Module: The initialized model.
    """
    model = AppleDiseaseModel(
        model_name=model_name, num_classes=num_classes, pretrained=pretrained
    )
    return model
