import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="models")


class DogClassifier(nn.Module):
    """
    Dog Breed Classifier using a pretrained backbone from timm.
    Replaces the default classifier with a custom Dropout + Linear head.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        """
        Args:
            model_name (str): Name of the timm model architecture.
            num_classes (int): Number of target classes.
            pretrained (bool): Whether to load ImageNet-1k pretrained weights.
        """
        super(DogClassifier, self).__init__()

        # Create backbone
        # num_classes=0 removes the default classifier and returns the pooled feature vector
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Determine input features for the head
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback for some models, though num_features is standard in timm
            # Run a dummy forward pass to infer shape if necessary, but num_features is reliable
            in_features = self.backbone.embed_dim

        # Custom Head: Dropout -> Linear
        # Using 0.5 dropout for regularization as per standard fine-tuning practices
        self.head = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        """
        # Get features from backbone (Batch, Num_Features)
        features = self.backbone(x)

        # Pass through custom head
        logits = self.head(features)

        return logits


def get_model(
    model_name: str,
    num_classes: int = Config.NUM_CLASSES,
    device: torch.device = Config.DEVICE,
    pretrained: bool = True,
):
    """
    Factory function to create and load the model.

    Args:
        model_name (str): Name of the architecture (e.g., 'convnext_base.fb_in1k').
        num_classes (int): Number of classes.
        device (torch.device): Device to move the model to.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        nn.Module: The instantiated model on the specified device.
    """
    logger.info(f"Initializing model: {model_name} (Pretrained={pretrained})")

    try:
        model = DogClassifier(model_name, num_classes, pretrained)
        model = model.to(device)
        return model
    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {e}")
        raise e
