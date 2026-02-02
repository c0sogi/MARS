import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class BirdModel(nn.Module):
    """
    Bird Species Classification Model.

    Wrapper for ResNet18 and DenseNet121 architectures.
    Adapts the final fully connected layer to output logits for the specific number of bird species.
    The native Global Average Pooling (GAP) in these models automatically handles the
    rectangular input size (224x448) defined in the configuration.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): Name of the architecture ('resnet18' or 'densenet121').
            pretrained (bool): If True, loads ImageNet pre-trained weights.
        """
        super(BirdModel, self).__init__()
        self.model_name = model_name
        self.num_classes = Config.NUM_CLASSES

        if model_name == "resnet18":
            # Load ResNet18
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)

            # ResNet structure: ... -> avgpool -> flatten -> fc
            # Replace the final fully connected layer
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, self.num_classes)

        elif model_name == "densenet121":
            # Load DenseNet121
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)

            # DenseNet structure: features -> relu -> avgpool -> flatten -> classifier
            # Replace the final classifier layer
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, self.num_classes)

        else:
            raise ValueError(
                f"Architecture '{model_name}' is not supported. "
                "Please choose 'resnet18' or 'densenet121'."
            )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # The backbone's forward method handles feature extraction,
        # global average pooling, flattening, and the final linear layer.
        return self.backbone(x)


def get_model(model_name, pretrained=True):
    """
    Factory function to instantiate the BirdModel.

    Args:
        model_name (str): Name of the architecture ('resnet18' or 'densenet121').
        pretrained (bool): Whether to use pre-trained weights.

    Returns:
        BirdModel: An instance of the configured model.
    """
    return BirdModel(model_name, pretrained=pretrained)
