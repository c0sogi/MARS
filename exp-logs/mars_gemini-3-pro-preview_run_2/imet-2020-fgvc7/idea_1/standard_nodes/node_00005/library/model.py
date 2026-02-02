import torch
import torch.nn as nn
import timm


class ArtworkClassifier(nn.Module):
    """
    ArtworkClassifier model based on EfficientNetV2-S.

    This class loads a pre-trained backbone from the timm library,
    removes the original classification head, and replaces it with a
    fully connected layer matching the number of artwork attributes.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Args:
            model_name (str): Name of the model architecture (e.g., 'tf_efficientnetv2_s').
            num_classes (int): Number of output classes (attributes).
            pretrained (bool): Whether to load pre-trained ImageNet weights.
        """
        super(ArtworkClassifier, self).__init__()

        # Load the pre-trained model with num_classes=0 to get feature extractor
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Get the number of features from the backbone
        self.num_features = self.backbone.num_features

        # Classification head
        self.cls_head = nn.Linear(self.num_features, num_classes)

        # Regression head for predicting label count
        self.count_head = nn.Linear(self.num_features, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            tuple: (logits, count_pred)
        """
        features = self.backbone(x)
        logits = self.cls_head(features)
        count_pred = self.count_head(features)
        return logits, count_pred.squeeze(-1)
