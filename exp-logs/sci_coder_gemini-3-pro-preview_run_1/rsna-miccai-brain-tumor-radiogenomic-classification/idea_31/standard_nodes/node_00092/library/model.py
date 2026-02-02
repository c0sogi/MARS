import torch
import torch.nn as nn
import timm
from library.config import (
    MODEL_NAME,
    NUM_CLASSES,
    DROPOUT_RATE,
    NUM_CHANNELS,
)


class RNWIVEfficientNet(nn.Module):
    """
    Standard EfficientNet-B0 with 3-channel input (Cite solution_lesson_node_00009).
    Removed complex weight inflation and structured dropout (Cite solution_lesson_node_00025).
    """

    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        # Cite solution_lesson_node_00012: Use default architecture settings where possible
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=NUM_CHANNELS,  # 3 channels
            drop_rate=DROPOUT_RATE,  # 0.2
        )

        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
