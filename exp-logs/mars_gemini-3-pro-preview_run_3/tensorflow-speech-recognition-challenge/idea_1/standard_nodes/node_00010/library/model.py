import torch
import torch.nn as nn
import torchvision.models as models
from library import config


class ResNetAudioClassifier(nn.Module):
    """
    ResNet34-based Audio Classifier.
    Adapts a pretrained ImageNet model for single-channel spectrogram input.
    Cite solution_lesson_node_00003.
    Cite solution_lesson_node_00006.
    """

    def __init__(self, num_classes=config.NUM_CLASSES):
        super(ResNetAudioClassifier, self).__init__()

        # Load pretrained ResNet34
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Modify first layer to accept 1 channel instead of 3
        # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Cite solution_lesson_node_00005: Preserve Pretrained Weights
        original_weights = self.resnet.conv1.weight.data
        new_weights = original_weights.sum(dim=1, keepdim=True)

        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.conv1.weight.data = new_weights

        # Modify the fully connected layer for our number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
