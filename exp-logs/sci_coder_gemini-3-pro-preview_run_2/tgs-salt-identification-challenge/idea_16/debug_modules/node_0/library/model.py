import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.

    Structure:
    1. 1x1 Conv: in_channels -> internal_channels
    2. 3x3 Transposed Conv (stride 2): internal_channels -> internal_channels (Upsampling)
    3. 1x1 Conv: internal_channels -> out_channels

    Internal dimension is calculated as in_channels // 4.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Width correction: internal dimension is in_channels // 4
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv (Reduce)
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transposed Conv (Upsample)
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
            # 1x1 Conv (Expand)
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34WideLinkNet(nn.Module):
    """
    ResNet34-WideLinkNet with MLP-Concatenation for Depth Injection.

    Architecture:
    - Backbone: ResNet34 (1-channel input adaptation)
    - Bottleneck: No compression, Depth Injection (Concat)
    - Decoder: Wide-LinkNet blocks with Additive Skip Connections
    """

    def __init__(self):
        super(ResNet34WideLinkNet, self).__init__()

        # Load Pretrained ResNet34
        # Using weights='IMAGENET1K_V1' if available, else pretrained=True logic handled by torchvision
        try:
            from torchvision.models import ResNet34_Weights

            self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        except ImportError:
            self.resnet = resnet34(pretrained=True)

        # Input Adaptation: Modify conv1 for 1-channel input
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Sum weights across the channel dimension (dim 1)
        with torch.no_grad():
            self.resnet.conv1.weight.data = original_conv1.weight.data.sum(
                dim=1, keepdim=True
            )

        # Encoder Layers
        self.encoder_initial = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu
        )
        self.encoder_maxpool = self.resnet.maxpool
        self.encoder_layer1 = self.resnet.layer1  # 64 ch, H/4
        self.encoder_layer2 = self.resnet.layer2  # 128 ch, H/8
        self.encoder_layer3 = self.resnet.layer3  # 256 ch, H/16
        self.encoder_layer4 = self.resnet.layer4  # 512 ch, H/32

        # Depth Injection MLP
        # Projects scalar depth to 32-channel embedding
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, 32)
        )

        # Decoder Blocks
        # Decoder 4: Input (512 enc + 32 depth) -> Output 256 (matches layer3)
        self.dec4 = DecoderBlock(512 + 32, 256)

        # Decoder 3: Input 256 -> Output 128 (matches layer2)
        self.dec3 = DecoderBlock(256, 128)

        # Decoder 2: Input 128 -> Output 64 (matches layer1)
        self.dec2 = DecoderBlock(128, 64)

        # Decoder 1: Input 64 -> Output 64 (matches initial conv output)
        self.dec1 = DecoderBlock(64, 64)

        # Final Block: Upsample from H/2 to H and map to logits
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x, z):
        """
        Args:
            x (torch.Tensor): Image input (N, 1, H, W)
            z (torch.Tensor): Depth input (N, 1) or (N,)

        Returns:
            torch.Tensor: Logits (N, 1, H, W)
        """
        # Ensure z has correct shape (N, 1)
        if z.dim() == 1:
            z = z.view(-1, 1)

        # --- Encoder ---
        x1 = self.encoder_initial(x)  # (N, 64, H/2, W/2)
        x_mp = self.encoder_maxpool(x1)  # (N, 64, H/4, W/4)
        x2 = self.encoder_layer1(x_mp)  # (N, 64, H/4, W/4)
        x3 = self.encoder_layer2(x2)  # (N, 128, H/8, W/8)
        x4 = self.encoder_layer3(x3)  # (N, 256, H/16, W/16)
        x5 = self.encoder_layer4(x4)  # (N, 512, H/32, W/32)

        # --- Bottleneck & Depth Injection ---
        # Process depth
        d_emb = self.depth_mlp(z)  # (N, 32)
        # Expand depth to spatial dimensions of bottleneck features
        d_emb = d_emb.unsqueeze(2).unsqueeze(3)  # (N, 32, 1, 1)
        d_emb = d_emb.expand(-1, -1, x5.size(2), x5.size(3))  # (N, 32, H/32, W/32)

        # Concatenate
        bottleneck = torch.cat([x5, d_emb], dim=1)  # (N, 544, H/32, W/32)

        # --- Decoder (Additive Skip Connections) ---
        d4 = self.dec4(bottleneck)  # Upsamples to H/16
        d4 = d4 + x4  # Additive Skip

        d3 = self.dec3(d4)  # Upsamples to H/8
        d3 = d3 + x3  # Additive Skip

        d2 = self.dec2(d3)  # Upsamples to H/4
        d2 = d2 + x2  # Additive Skip

        d1 = self.dec1(d2)  # Upsamples to H/2
        d1 = d1 + x1  # Additive Skip

        # --- Final ---
        out = self.final(d1)  # Upsamples to H

        return out
