import torch
import torch.nn as nn
import torchvision.models as models


class MLPDepthInjection(nn.Module):
    """
    Projects a scalar depth value into a feature embedding.
    Structure: Linear -> ReLU -> Linear.
    """

    def __init__(self, input_dim=1, hidden_dim=16, output_dim=32):
        super(MLPDepthInjection, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        # Ensure z has shape (B, 1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        return self.net(z)


class LinkNetDecoderBlock(nn.Module):
    """
    LinkNet Decoder Block with 'Wide' internal dimension.
    Standard LinkNet uses out_channels // 4. We use in_channels // 4 to preserve information.

    Flow:
    1. 1x1 Conv (Compression)
    2. 3x3 Transpose Conv (Upsampling, Stride 2)
    3. 1x1 Conv (Expansion)
    """

    def __init__(self, in_channels, out_channels):
        super(LinkNetDecoderBlock, self).__init__()

        # Wide internal dimension strategy: derive from input to avoid bottlenecks
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transpose Conv (Upsample)
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 1x1 Conv
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34WideLinkNet(nn.Module):
    """
    ResNet34 Encoder + Wide LinkNet Decoder with Depth Injection.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super(ResNet34WideLinkNet, self).__init__()

        # 1. Backbone: ResNet34
        backbone = models.resnet34(pretrained=pretrained)

        # 2. Input Adaptation (3 channels -> 1 channel)
        # Sum the weights of the first convolution to preserve pretrained filters
        original_conv1 = backbone.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Encoder Layers
        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels
        self.layer4 = backbone.layer4  # 512 channels

        # 3. Depth Injection Module
        self.depth_mlp = MLPDepthInjection(input_dim=1, hidden_dim=16, output_dim=32)

        # 4. Decoder Blocks
        # Bottleneck: Layer4 (512) + Depth (32) = 544 channels
        self.decoder4 = LinkNetDecoderBlock(512 + 32, 256)
        self.decoder3 = LinkNetDecoderBlock(256, 128)
        self.decoder2 = LinkNetDecoderBlock(128, 64)
        self.decoder1 = LinkNetDecoderBlock(64, 64)

        # 5. Final Head
        # Dec1 output is 64 channels at stride 2 (relative to input).
        # We need to upsample by 2x to get to original resolution.
        self.final_head = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x, z):
        """
        Args:
            x: Image tensor (B, 1, H, W)
            z: Normalized depth scalar (B,) or (B, 1)
        """
        # --- Encoder ---
        x0 = self.conv1(x)  # Stride 2 (64 ch)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x_stem = x0  # Save for skip connection (Stride 2)

        x0 = self.maxpool(x0)  # Stride 4 (64 ch)

        x1 = self.layer1(x0)  # Stride 4 (64 ch)
        x2 = self.layer2(x1)  # Stride 8 (128 ch)
        x3 = self.layer3(x2)  # Stride 16 (256 ch)
        x4 = self.layer4(x3)  # Stride 32 (512 ch)

        # --- Depth Injection ---
        z_emb = self.depth_mlp(z)  # (B, 32)
        # Expand spatially to match x4 dimensions
        z_emb = z_emb.unsqueeze(2).unsqueeze(3)  # (B, 32, 1, 1)
        z_emb = z_emb.expand(-1, -1, x4.size(2), x4.size(3))  # (B, 32, H/32, W/32)

        # Concatenate features and depth
        bottleneck = torch.cat([x4, z_emb], dim=1)  # (B, 544, H/32, W/32)

        # --- Decoder with Additive Skip Connections ---
        # Block 4
        d4 = self.decoder4(bottleneck)  # -> 256 ch, Stride 16
        d4 = d4 + x3

        # Block 3
        d3 = self.decoder3(d4)  # -> 128 ch, Stride 8
        d3 = d3 + x2

        # Block 2
        d2 = self.decoder2(d3)  # -> 64 ch, Stride 4
        d2 = d2 + x1

        # Block 1
        d1 = self.decoder1(d2)  # -> 64 ch, Stride 2
        d1 = d1 + x_stem

        # --- Final Head ---
        out = self.final_head(d1)  # -> num_classes ch, Stride 1

        return out
