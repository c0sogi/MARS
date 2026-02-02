import torch
import torch.nn as nn
import timm


class VFPNet(nn.Module):
    """
    Volumetric Feature Projection (VFP) Network.

    Architecture:
    1. 3D Convolutional Stem: Fuses 4 modalities and extracts local 3D texture.
       Downsamples H/W by 2, preserves Depth.
    2. Depth-wise Max Projection: Collapses the Depth dimension (32 -> 1) to achieve
       Z-axis translation invariance.
    3. 2D Backbone (EfficientNet-B0): Processes the projected feature map.
       The stem of EfficientNet is bypassed; features are fed directly to MBConv blocks.
    4. Head: Global Average Pooling + Linear Layer.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super(VFPNet, self).__init__()

        # ------------------------------------------------------------------
        # 1. 3D Convolutional Stem
        # Input: (B, 4, 32, 256, 256)
        # Output: (B, 32, 32, 128, 128)
        # Stride (1, 2, 2) preserves Depth (32) but halves H/W (256->128)
        # ------------------------------------------------------------------
        self.stem = nn.Conv3d(
            in_channels=4,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
            bias=False,
        )
        self.stem_bn = nn.BatchNorm3d(32)
        self.stem_act = nn.SiLU(inplace=True)  # Swish activation matches EfficientNet

        # ------------------------------------------------------------------
        # 2. Depth-wise Max Projection
        # Input: (B, 32, 32, 128, 128)
        # Output: (B, 32, 1, 128, 128)
        # Collapses the entire depth dimension (32 slices) to 1.
        # ------------------------------------------------------------------
        self.projector = nn.MaxPool3d(kernel_size=(32, 1, 1))

        # ------------------------------------------------------------------
        # 3. 2D Backbone (EfficientNet-B0)
        # We load the model and graft our 3D stem onto its blocks.
        # ------------------------------------------------------------------
        # Load pretrained EfficientNet-B0
        base_model = timm.create_model("efficientnet_b0", pretrained=pretrained)

        # We extract the MBConv blocks and subsequent layers.
        # We bypass base_model.conv_stem, base_model.bn1, base_model.act1
        # The first block of EffNet-B0 expects 32 channels, which matches our stem output.
        self.blocks = base_model.blocks
        self.conv_head = base_model.conv_head
        self.bn2 = base_model.bn2
        self.act2 = getattr(base_model, "act2", nn.Identity())
        self.global_pool = base_model.global_pool

        # ------------------------------------------------------------------
        # 4. Classification Head
        # ------------------------------------------------------------------
        # Input features to classifier
        in_features = base_model.classifier.in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 4, 32, 256, 256)
                              (Batch, Modalities, Depth, Height, Width)
        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        # 1. 3D Stem
        x = self.stem(x)  # -> (B, 32, 32, 128, 128)
        x = self.stem_bn(x)
        x = self.stem_act(x)

        # 2. Projection
        x = self.projector(x)  # -> (B, 32, 1, 128, 128)

        # Squeeze depth dimension to transition to 2D
        x = x.squeeze(2)  # -> (B, 32, 128, 128)

        # 3. 2D Backbone
        x = self.blocks(x)  # MBConv blocks
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)

        # 4. Head
        x = self.global_pool(x)  # -> (B, C) or (B, C, 1, 1) depending on timm version

        # Flatten if necessary
        if x.ndim == 4:
            x = x.flatten(1)

        x = self.classifier(x)  # -> (B, 1)

        return x
