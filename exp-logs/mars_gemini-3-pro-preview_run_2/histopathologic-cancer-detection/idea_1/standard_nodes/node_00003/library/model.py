import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class TumorClassifier(nn.Module):
    """
    TumorClassifier based on MobileNetV3-Small architecture.

    This model uses a pre-trained MobileNetV3-Small backbone and replaces the
    final classification layer to output a single logit for binary classification
    (Tumor vs No Tumor).
    """

    def __init__(
        self,
        pretrained: bool = Config.PRETRAINED,
        dropout_rate: float = Config.DROPOUT_RATE,
    ):
        """
        Initializes the TumorClassifier.

        Args:
            pretrained (bool): If True, loads ImageNet pre-trained weights.
            dropout_rate (float): Dropout probability for the classification head.
        """
        super(TumorClassifier, self).__init__()

        weights = "DEFAULT" if pretrained else None

        if Config.MODEL_NAME == "resnet18":
            # Cite {solution_lesson_node_00002}
            self.model = models.resnet18(weights=weights)
            # ResNet18 has a 'fc' layer at the end
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, Config.NUM_CLASSES)

        elif Config.MODEL_NAME == "mobilenet_v3_small":
            # Load the MobileNetV3-Small model
            self.model = models.mobilenet_v3_small(weights=weights)

            # MobileNetV3-Small classifier structure typically looks like:
            # Sequential(
            #   (0): Linear(in_features=576, out_features=1024, bias=True)
            #   (1): Hardswish()
            #   (2): Dropout(p=0.2, inplace=True)
            #   (3): Linear(in_features=1024, out_features=1000, bias=True)
            # )

            # Update Dropout rate if the layer exists and is a Dropout layer
            if len(self.model.classifier) > 2 and isinstance(
                self.model.classifier[2], nn.Dropout
            ):
                self.model.classifier[2].p = dropout_rate

            # Replace the final Linear layer
            last_layer_idx = len(self.model.classifier) - 1
            if isinstance(self.model.classifier[last_layer_idx], nn.Linear):
                in_features = self.model.classifier[last_layer_idx].in_features
                self.model.classifier[last_layer_idx] = nn.Linear(
                    in_features, Config.NUM_CLASSES
                )
            else:
                raise ValueError("Unexpected MobileNetV3 classifier structure.")
        else:
            raise ValueError(f"Unsupported model name: {Config.MODEL_NAME}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, 1).
        """
        return self.model(x)
