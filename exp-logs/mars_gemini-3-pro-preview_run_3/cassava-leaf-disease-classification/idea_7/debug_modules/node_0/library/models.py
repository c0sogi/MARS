import torch
import torch.nn as nn
import timm


class CassavaClassifier(nn.Module):
    """
    A unified wrapper for timm models to support the Heterogeneous Ensemble strategy.
    Can instantiate both Transformer (ViT) and CNN (EfficientNet) backbones.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Initialize the model.

        Args:
            model_name (str): The name of the model architecture to load from timm
                              (e.g., 'vit_base_patch16_384' or 'tf_efficientnet_b4').
            num_classes (int): The number of target classes (5 for this task).
            pretrained (bool): Whether to load weights pretrained on ImageNet.
        """
        super(CassavaClassifier, self).__init__()

        # Instantiate the model using timm.
        # passing num_classes tells timm to replace the original head (usually 1000 classes)
        # with a new head suitable for our specific task.
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        return self.model(x)
