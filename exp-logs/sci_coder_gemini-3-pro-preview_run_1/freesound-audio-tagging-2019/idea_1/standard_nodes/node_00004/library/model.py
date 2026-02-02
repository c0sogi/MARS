import torch
import torch.nn as nn
import timm
from library.config import Config


class AudioModel(nn.Module):
    """
    ResNet34-based Audio Classifier.
    Uses a pretrained ResNet34 from timm, adapted for 1-channel input.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, in_channels=1):
        super(AudioModel, self).__init__()

        # Input Batch Normalization to adapt spectrogram statistics
        self.bn0 = nn.BatchNorm2d(in_channels)

        # Pretrained ResNet34
        # global_pool='max' acts as a MIL aggregator (Cite solution_lesson_node_00001)
        self.base_model = timm.create_model(
            "resnet34",
            pretrained=True,
            in_chans=in_channels,
            num_classes=num_classes,
            global_pool="max",
        )

    def forward(self, x):
        x = self.bn0(x)
        x = self.base_model(x)
        return x
