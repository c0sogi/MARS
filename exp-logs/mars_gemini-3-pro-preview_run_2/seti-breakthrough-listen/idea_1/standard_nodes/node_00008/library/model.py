import torch
import torch.nn as nn
import timm
from library.config import Config


class ShallowCNN(nn.Module):
    """
    Wrapper for EfficientNet-B0 with Global Max Pooling.
    (Keeping class name ShallowCNN to avoid refactoring other files, though it is now a deep model)
    """

    def __init__(self):
        super(ShallowCNN, self).__init__()

        # Use EfficientNet-B0 with Global Max Pooling (Cite solution_lesson_node_00007)
        # in_chans=1 because we stack panels vertically
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=Config.NUM_CHANNELS,
            num_classes=1,
            global_pool="max",
        )

    def forward(self, x):
        return self.model(x)
