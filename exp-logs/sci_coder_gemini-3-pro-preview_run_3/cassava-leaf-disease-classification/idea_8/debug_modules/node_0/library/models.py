import torch
import torch.nn as nn
import timm


class CassavaClassifier(nn.Module):
    """
    A wrapper class for timm models to facilitate Cassava Leaf Disease Classification.
    It initializes a backbone model and adapts the classification head for the specific number of classes.
    """

    def __init__(self, model_name, n_class, pretrained=True):
        super(CassavaClassifier, self).__init__()

        # Initialize the model using timm
        # setting num_classes=n_class automatically:
        # 1. Resets the classification head to match n_class
        # 2. Initializes the new head weights (typically using xavier or truncated normal)
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=n_class
        )

    def forward(self, x):
        return self.model(x)


def get_model(cfg, model_name):
    """
    Factory function to instantiate the CassavaClassifier.

    Args:
        cfg: Configuration object containing model parameters (specifically NUM_CLASSES).
        model_name: The name of the timm model to instantiate (e.g., 'vit_base_patch16_384').

    Returns:
        model: An instance of CassavaClassifier ready for training or inference.
    """
    model = CassavaClassifier(
        model_name=model_name, n_class=cfg.NUM_CLASSES, pretrained=True
    )
    return model
