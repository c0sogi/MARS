import torch
import torch.nn as nn
import torchvision


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.

    Structure:
    1x1 Conv (Reduce) -> BN -> ReLU
    3x3 Transpose Conv (Upsample) -> BN -> ReLU
    1x1 Conv (Expand) -> BN -> ReLU

    The internal dimension is calculated as in_channels // 4 to preserve information flow.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Wide-LinkNet strategy: internal dim based on input, not output
        internal_dim = in_channels // 4

        self.block = nn.Sequential(
            # Reduce
            nn.Conv2d(in_channels, internal_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_dim),
            nn.ReLU(inplace=True),
            # Upsample
            nn.ConvTranspose2d(
                internal_dim,
                internal_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(internal_dim),
            nn.ReLU(inplace=True),
            # Expand
            nn.Conv2d(internal_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34WideLinkNet(nn.Module):
    """
    ResNet34-WideLinkNet with Marginalized Depth-Scan support.

    Features:
    - ResNet34 Backbone (pretrained)
    - 1-Channel Input Adaptation
    - Depth Injection at Bottleneck (Concatenation)
    - Wide-LinkNet Decoder with Additive Skip Connections
    """

    def __init__(self, pretrained=True):
        super(ResNet34WideLinkNet, self).__init__()

        # Load Pretrained Backbone
        # Note: Using weights='IMAGENET1K_V1' is the modern equivalent of pretrained=True
        # but for compatibility we rely on the argument or default torchvision behavior.
        resnet = torchvision.models.resnet34(pretrained=pretrained)

        # --- Input Adaptation ---
        # Modify first convolution to accept 1 channel instead of 3
        # We sum the weights of the original RGB channels to preserve feature detection
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(resnet.conv1.weight.sum(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder Layers
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # --- Depth Injection ---
        # Non-Linear MLP to project scalar depth to embedding
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, 32)
        )

        # --- Decoder ---
        # Bottleneck Input: 512 (Layer4) + 32 (Depth) = 544

        # Decoder 4: 544 -> 256 (Matches Layer3)
        self.dec4 = DecoderBlock(544, 256)

        # Decoder 3: 256 -> 128 (Matches Layer2)
        self.dec3 = DecoderBlock(256, 128)

        # Decoder 2: 128 -> 64 (Matches Layer1)
        self.dec2 = DecoderBlock(128, 64)

        # Decoder 1: 64 -> 64 (Matches Conv1 output)
        self.dec1 = DecoderBlock(64, 64)

        # Final Upsampling Block
        # Upsamples from 1/2 resolution (after Dec1) to Full Resolution
        self.final_conv = nn.Sequential(
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
            x (torch.Tensor): Input images [B, 1, H, W]. Should be padded to multiple of 32 (e.g., 128x128).
            z (torch.Tensor): Normalized depth values [B] or [B, 1].
        """
        # --- Encoder ---
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # [B, 64, H/2, W/2]

        x1 = self.maxpool(x0)  # [B, 64, H/4, W/4]
        x1 = self.layer1(x1)  # [B, 64, H/4, W/4]

        x2 = self.layer2(x1)  # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)  # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)  # [B, 512, H/32, W/32]

        # --- Bottleneck with Depth Injection ---
        if z.dim() == 1:
            z = z.view(-1, 1)

        d_emb = self.depth_mlp(z)  # [B, 32]

        # Expand depth embedding to spatial dimensions of x4
        d_emb = d_emb.unsqueeze(2).unsqueeze(3)  # [B, 32, 1, 1]
        d_emb = d_emb.expand(-1, -1, x4.size(2), x4.size(3))

        # Concatenate
        bottleneck = torch.cat([x4, d_emb], dim=1)  # [B, 544, H/32, W/32]

        # --- Decoder with Additive Skips ---
        d4 = self.dec4(bottleneck)  # [B, 256, H/16, W/16]
        d4 = d4 + x3

        d3 = self.dec3(d4)  # [B, 128, H/8, W/8]
        d3 = d3 + x2

        d2 = self.dec2(d3)  # [B, 64, H/4, W/4]
        d2 = d2 + x1

        d1 = self.dec1(d2)  # [B, 64, H/2, W/2]
        d1 = d1 + x0

        # --- Final Output ---
        out = self.final_conv(d1)  # [B, 1, H, W]

        return out
