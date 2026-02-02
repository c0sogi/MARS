import torch
import torch.nn as nn
import timm


class AppleDiseaseModel(nn.Module):
    """
    Apple Disease Detection Model based on EfficientNetV2-M.

    Architecture:
    - Backbone: EfficientNetV2-M (pre-trained on ImageNet)
    - Head: Global Average Pooling -> Dropout -> Linear Layer (4 classes)
    """

    def __init__(
        self,
        model_name="tf_efficientnet_b4_ns",
        num_classes=4,
        pretrained=True,
        drop_rate=0.3,
    ):
        """
        Args:
            model_name (str): Name of the timm model to use. Defaults to 'efficientnetv2_m'.
            num_classes (int): Number of target classes. Defaults to 4.
            pretrained (bool): Whether to load pre-trained weights. Defaults to True.
            drop_rate (float): Dropout probability for the classification head. Defaults to 0.3.
        """
        super(AppleDiseaseModel, self).__init__()

        # Create the backbone model
        # num_classes=0 removes the default classifier and returns the pooled features (Global Average Pooling)
        # by default in timm when global_pool is not set to empty string.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Get the number of input features for the classifier
        self.in_features = self.backbone.num_features

        # Define the custom classification head
        # As per requirements: Dropout -> Linear
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate), nn.Linear(self.in_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        # Pass through backbone (includes Global Average Pooling)
        features = self.backbone(x)

        # Pass through classification head
        logits = self.head(features)

        return logits
