import torch
import torch.nn as nn
import timm


class BraTSEfficientNet(nn.Module):
    """
    Modality-Grouped Stabilized 2.5D Network.

    Architecture:
    1. Input: (B, 128, 256, 256) - 4 modalities * 32 slices.
    2. Stem:
       - Grouped Conv (groups=4): Processes each modality independently.
       - Pointwise Conv: Fuses features across modalities.
       - Stabilized with BatchNorm, ReLU, and He Initialization.
    3. Backbone: EfficientNet-B0 (timm implementation).
    4. Head: Global Average Pooling -> Linear -> Logits.
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # Modality-Grouped Stabilized Stem
        # ==========================================
        # Layer 1: Modality-Wise Learning
        # Input: 128 channels (4 blocks of 32 slices)
        # Output: 64 channels (4 blocks of 16 features)
        # groups=4 ensures each modality is convolved separately.
        self.stem_conv1 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            padding=1,
            groups=4,
            bias=False,
        )
        self.stem_bn1 = nn.BatchNorm2d(64)
        self.stem_act1 = nn.ReLU(inplace=True)

        # Layer 2: Cross-Modal Fusion
        # 1x1 Convolution to mix features from different modalities
        self.stem_conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, bias=False
        )
        self.stem_bn2 = nn.BatchNorm2d(64)
        self.stem_act2 = nn.ReLU(inplace=True)

        # Initialization
        # Explicit Kaiming/He Normal initialization for stability
        nn.init.kaiming_normal_(
            self.stem_conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.stem_conv2.weight, mode="fan_out", nonlinearity="relu"
        )

        # ==========================================
        # Backbone (EfficientNet-B0)
        # ==========================================
        # in_chans=64 matches the output of the stem.
        # num_classes=0 returns the pooled feature vector.
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, in_chans=64, num_classes=0
        )

        # ==========================================
        # Classification Head
        # ==========================================
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 128, 256, 256).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Stem
        x = self.stem_conv1(x)
        x = self.stem_bn1(x)
        x = self.stem_act1(x)

        x = self.stem_conv2(x)
        x = self.stem_bn2(x)
        x = self.stem_act2(x)

        # Backbone
        # Returns pooled features (B, num_features)
        x = self.backbone(x)

        # Head
        x = self.head(x)

        return x
