import torch
import torch.nn as nn
from torchvision import models
from library.config import MODEL_ARCH


from library.config import MODEL_ARCH, NUM_CLASSES


class PlantClassifier(nn.Module):
    """
    A fine-tunable EfficientNet-B0 model for plant classification.
    """

    def __init__(self, architecture=MODEL_ARCH, num_classes=NUM_CLASSES):
        super(PlantClassifier, self).__init__()

        if architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.model = models.efficientnet_b0(weights=weights)
            in_features = 1280
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Replace the classifier head
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)
