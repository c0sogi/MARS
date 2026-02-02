import torch
import torch.nn as nn
import timm


class PlantClassifier(nn.Module):
    """
    PlantClassifier wrapper class that initializes an EfficientNetV2-Small model
    from the timm library. It modifies the classification head to project
    features to the specific number of plant species classes.
    """

    def __init__(
        self, num_classes=15501, model_name="tf_efficientnetv2_s", pretrained=True
    ):
        """
        Args:
            num_classes (int): The number of output classes (plant species).
                               Defaults to 15501 based on dataset analysis.
            model_name (str): The name of the architecture in timm.
                              Defaults to 'tf_efficientnetv2_s'.
            pretrained (bool): Whether to load pretrained ImageNet weights.
                               Defaults to True.
        """
        super(PlantClassifier, self).__init__()

        # Initialize the model using timm.
        # Setting num_classes tells timm to replace the default classifier head
        # with a new Linear layer with the specified number of outputs.
        # For tf_efficientnetv2_s, the feature dim is 1280.
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Output logits of shape (Batch_Size, Num_Classes).
        """
        return self.model(x)
