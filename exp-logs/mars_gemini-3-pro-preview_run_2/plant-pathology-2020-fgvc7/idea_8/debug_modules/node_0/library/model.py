import torch
import torch.nn as nn
import timm
from library.config import Config


class DiseaseClassifier(nn.Module):
    """
    A PyTorch module for Apple Disease Detection.

    Wraps a timm backbone and replaces the classifier head with a custom linear layer
    for multi-label classification (Rust and Scab).
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = Config.NUM_CLASSES,
        pretrained: bool = True,
    ):
        """
        Args:
            model_name (str): Name of the timm model to load (e.g., 'tf_efficientnetv2_l.in21k_ft_in1k').
            num_classes (int): Number of output classes. Defaults to Config.NUM_CLASSES (2).
            pretrained (bool): Whether to load pretrained weights. Defaults to True.
        """
        super(DiseaseClassifier, self).__init__()
        self.model_name = model_name

        # Load the backbone from timm.
        # num_classes=0 removes the default classifier and pooling, returning the feature vector.
        # However, behavior depends on the specific model in timm.
        # For EfficientNet and ConvNeXt, num_classes=0 usually enables global pooling and returns flat features.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Determine the number of input features for the linear head
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            # Fallback: Inference to determine shape
            # We use a small dummy input. Resolution doesn't strictly matter for channel count
            # but we use a standard size to be safe.
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                self.in_features = features.shape[1]

        # Define the custom classification head
        # We output raw logits. The loss function (BCEWithLogitsLoss) will handle the sigmoid.
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features using the backbone
        features = self.backbone(x)

        # Pass through the classification head
        logits = self.fc(features)

        return logits

    def load_weights(self, path: str, device: str = Config.DEVICE):
        """
        Helper method to load weights from a file.

        Args:
            path (str): Path to the .pth file.
            device (str): Device to map the weights to.
        """
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
