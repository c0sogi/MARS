import torch
import torch.nn as nn
import timm
from library.config import Config


class VAMSHDNet(nn.Module):
    """
    VAMS-HD (View-Adaptive Modality-Structured High-Density) Network.

    Architecture:
    1. Input: (B, 128, 224, 224) - 32 slices * 4 modalities.
    2. Stem: Compresses 128 channels to 64, downsamples to 112x112.
       - Conv2d(128, 64, 3, stride=2, padding=1)
       - BatchNorm2d
       - ReLU
    3. Backbone: EfficientNet-B0 (timm)
       - in_chans=64
       - drop_path_rate=0.2
    4. Head: Global Average Pooling + Linear (Logits)
    """

    def __init__(self):
        super(VAMSHDNet, self).__init__()

        # 1. Stabilized Global-Mixing Stem
        # Compresses high-density input (128ch) to backbone-friendly width (64ch)
        # and downsamples spatial resolution (224 -> 112)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=Config.IN_CHANNELS,
                out_channels=Config.STEM_OUT_CHANNELS,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Config.STEM_OUT_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # Initialize stem weights explicitly for stability
        self._init_weights(self.stem)

        # 2. Backbone & Head
        # EfficientNet-B0 configured to accept the 64-channel output from the stem.
        # We use num_classes=1 to include the final linear layer.
        # The output will be logits (no sigmoid), compatible with BCEWithLogitsLoss.
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            in_chans=Config.STEM_OUT_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

    def _init_weights(self, module):
        """
        Applies Kaiming/He Normal initialization to Convolutional layers
        to prevent gradient explosion/vanishing with high-channel inputs.
        """
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the VAMS-HD Network.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 128, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # Pass through Stem
        # Shape: (B, 128, 224, 224) -> (B, 64, 112, 112)
        x = self.stem(x)

        # Pass through Backbone and Head
        # Shape: (B, 64, 112, 112) -> (B, 1)
        x = self.backbone(x)

        return x
