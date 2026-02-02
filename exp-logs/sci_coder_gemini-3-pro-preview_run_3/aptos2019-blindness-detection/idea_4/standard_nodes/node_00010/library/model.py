import torch
import torch.nn as nn
import timm


class RetinopathyModel(nn.Module):
    """
    A regression model for Diabetic Retinopathy severity prediction.
    Uses an EfficientNet-B5 backbone with a custom regression head.
    """

    def __init__(self, model_name="efficientnet_b5", pretrained=True, drop_rate=0.5):
        """
        Args:
            model_name (str): Name of the timm model to use as backbone.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            drop_rate (float): Dropout probability for the regression head.
        """
        super(RetinopathyModel, self).__init__()

        # Load the backbone from timm
        # num_classes=0 removes the default classifier layer
        # global_pool='avg' ensures the backbone output is a pooled feature vector
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the number of input features for the head
        # For EfficientNet-B5, this is typically 2048
        in_features = self.backbone.num_features

        # Define the custom regression head: Dropout -> Linear
        self.head = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(in_features, 1))

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            torch.Tensor: Predicted severity scores of shape (B, 1)
        """
        # Pass through backbone to get pooled features
        features = self.backbone(x)

        # Pass through regression head
        output = self.head(features)

        return output
