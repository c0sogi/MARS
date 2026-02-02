import torch
import torch.nn as nn
import timm
from library.config import Config


class CassavaResNet(nn.Module):
    """
    Cassava model based on EfficientNet-B3 architecture.
    """

    def __init__(
        self,
        pretrained: bool = Config.PRETRAINED,
        num_classes: int = Config.NUM_CLASSES,
    ):
        super(CassavaResNet, self).__init__()

        # Cite solution_lesson_node_00007: Scaling performance via better architecture.
        # Using tf_efficientnet_b3_ns for better feature extraction.
        self.model = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
