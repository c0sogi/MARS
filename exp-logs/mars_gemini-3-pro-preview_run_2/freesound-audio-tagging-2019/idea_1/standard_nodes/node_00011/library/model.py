import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from library.config import Config


class AudioMobileNet(nn.Module):
    """
    Lightweight 2D CNN based on MobileNetV2 architecture.
    Uses Transfer Learning from ImageNet (Cite solution_lesson_node_00006).
    """

    def __init__(self):
        super(AudioMobileNet, self).__init__()

        # Load pre-trained MobileNetV2
        # Cite solution_lesson_node_00006
        weights = MobileNet_V2_Weights.DEFAULT
        base_model = mobilenet_v2(weights=weights)

        # Modify first layer for 1 channel input (Log-Mel Spectrogram)
        # Cite solution_lesson_node_00006: Sum weights along channel dimension
        original_first_layer = base_model.features[0][0]
        new_first_layer = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 1
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=False,
        )

        with torch.no_grad():
            new_first_layer.weight.data = original_first_layer.weight.data.sum(
                dim=1, keepdim=True
            )

        base_model.features[0][0] = new_first_layer

        self.features = base_model.features

        # Classifier Head
        # MobileNetV2 last channel is 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, Config.NUM_CLASSES),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrogram of shape (Batch, 1, Freq, Time)
        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        # Feature Extraction
        x = self.features(x)

        # Global Max Pooling
        # Collapses spatial dimensions (Freq, Time) to (1, 1)
        # Allows handling of variable length inputs
        # Using Max Pooling to avoid signal dilution from zero-padding (Cite solution_lesson_node_00002)
        x = nn.functional.adaptive_max_pool2d(x, (1, 1))

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)

        return x
