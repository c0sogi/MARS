import torch
import torch.nn as nn
import timm


class AppleDiseaseModel(nn.Module):
    """
    Apple Disease Detection Model.

    This class implements a neural network based on the EfficientNetV2-L backbone.
    It is designed for a Multi-Label Decomposition strategy, outputting 2 binary logits
    corresponding to the presence of Rust and Scab diseases.

    Attributes:
        backbone (nn.Module): The pretrained feature extractor (EfficientNetV2-L).
    """

    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize the model.

        Args:
            model_name (str): The name of the backbone model (e.g., 'tf_efficientnetv2_l').
            num_classes (int): The number of output logits. Defaults to 2 (Rust, Scab).
            pretrained (bool): Whether to load ImageNet-pretrained weights.
        """
        super(AppleDiseaseModel, self).__init__()

        # Initialize the backbone using timm
        # We load the pretrained model first with its default head to ensure correct weight loading
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # Replace the classifier head with a custom Linear layer
        # EfficientNetV2 models in timm typically use the attribute 'classifier'
        if hasattr(self.backbone, "classifier"):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, "fc"):
            # Fallback for ResNet-style architectures
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, "head"):
            # Fallback for ViT-style architectures
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
        else:
            raise AttributeError(
                f"Could not identify classifier head for model: {model_name}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Output logits of shape (Batch_Size, num_classes).
        """
        return self.backbone(x)
