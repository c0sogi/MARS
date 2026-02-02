import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AppleResNet34(nn.Module):
    """
    ResNet34 model for Apple Disease Detection.
    Implements the architecture required for the 'Calibrated Full-Data Seed Ensemble' strategy.
    Includes specific logic for Discriminative Fine-Tuning parameter grouping.
    """

    def __init__(self):
        super(AppleResNet34, self).__init__()

        # Load ResNet34 backbone
        # We use the weights parameter if PRETRAINED is True
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        self.backbone = models.resnet34(weights=weights)

        # The original ResNet34 fc layer has 512 input features
        in_features = self.backbone.fc.in_features

        # Replace the fully connected layer (head)
        # Simple Global Average Pooling (part of backbone structure) -> Linear
        # We output logits; Softmax/Sigmoid will be applied during loss calculation/inference
        self.backbone.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, NUM_CLASSES)
        """
        return self.backbone(x)

    def get_optimizer_params(self):
        """
        Returns model parameters for uniform optimization.
        Reverts discriminative fine-tuning to standard uniform LR (Cite solution_lesson_node_00044).

        Returns:
            list: A list of dictionaries containing params and specific learning rates.
        """
        return [
            {
                "params": self.parameters(),
                "lr": Config.LEARNING_RATE,
                "weight_decay": Config.WEIGHT_DECAY,
            }
        ]
