import torch
import torch.nn as nn
from torchvision import models
from library.config import MODEL_ARCH


from library.config import MODEL_ARCH, NUM_CLASSES


class PlantClassifier(nn.Module):
    """
    A fine-tunable EfficientNet-B0 model for plant classification.
    According to lesson solution_lesson_node_00001, frozen backbones fail on high-cardinality
    fine-grained tasks, so we unfreeze weights and add a learnable head.
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
        # EfficientNet classifier is usually a Sequential(Dropout, Linear)
        # We replace it with a new Linear layer for our number of classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)
