import torch
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.

    Structure:
    - 1x1 Conv (in -> internal)
    - 3x3 Transposed Conv (internal -> internal)
    - 1x1 Conv (internal -> out)

    The internal dimension is calculated as in_channels // 4 to maintain width/capacity.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Wide-LinkNet strategy: Internal width based on input channels
        mid_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1. Reduce / Project
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 2. Upsample
            nn.ConvTranspose2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 3. Expand / Output
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SaltNet(nn.Module):
    """
    ResNet34-WideLinkNet with Bottleneck Depth Concatenation.

    Features:
    - Backbone: ResNet34 (pretrained, first layer summed for 1-channel input).
    - Bottleneck: No compression. Concatenates encoder features with depth embedding.
    - Depth Embedding: Scalar depth -> MLP -> 32-channel spatial map.
    - Decoder: Wide-LinkNet blocks with additive skip connections.
    """

    def __init__(self):
        super(SaltNet, self).__init__()

        # Load Pretrained ResNet34
        # Note: Using pretrained=True for broader compatibility across torchvision versions
        resnet = models.resnet34(pretrained=True)

        # ---------------------------------------------------------
        # 1. Input Adaptation
        # ---------------------------------------------------------
        # Modify the first convolutional layer to accept 1-channel input (Grayscale)
        # We sum the weights of the original 3 channels to preserve feature detection capabilities
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder Layers
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # ---------------------------------------------------------
        # 2. Depth Injection
        # ---------------------------------------------------------
        # Project scalar depth to a 32-channel embedding
        # MLP: Linear -> ReLU -> Linear
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, 32)
        )

        # ---------------------------------------------------------
        # 3. Decoder
        # ---------------------------------------------------------
        # Bottleneck Input: Layer4 (512) + Depth (32) = 544 channels
        # Target Output: Match Layer3 (256) for skip connection
        self.dec4 = DecoderBlock(544, 256)

        # Input: 256, Target: 128 (Match Layer2)
        self.dec3 = DecoderBlock(256, 128)

        # Input: 128, Target: 64 (Match Layer1)
        self.dec2 = DecoderBlock(128, 64)

        # Input: 64, Target: 64 (Match Layer0 / Pre-Maxpool)
        self.dec1 = DecoderBlock(64, 64)

        # Final Head: Upsample 64 -> 32 -> 1
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary Logits
        )

    def forward(self, x, z):
        """
        Args:
            x: Image tensor (B, 1, H, W)
            z: Depth tensor (B, 1) or (B,)
        """
        # Ensure z is (B, 1)
        if z.dim() == 1:
            z = z.view(-1, 1)

        # ---------------------
        # Encoder Pass
        # ---------------------
        x0 = self.conv1(x)  # (B, 64, H/2, W/2)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x_pool = self.maxpool(x0)  # (B, 64, H/4, W/4)

        e1 = self.layer1(x_pool)  # (B, 64, H/4, W/4)
        e2 = self.layer2(e1)  # (B, 128, H/8, W/8)
        e3 = self.layer3(e2)  # (B, 256, H/16, W/16)
        e4 = self.layer4(e3)  # (B, 512, H/32, W/32)

        # ---------------------
        # Depth Processing
        # ---------------------
        d_emb = self.depth_mlp(z)  # (B, 32)
        # Expand to spatial dimensions of the bottleneck
        d_emb = d_emb.unsqueeze(2).unsqueeze(3)  # (B, 32, 1, 1)
        d_emb = d_emb.expand(-1, -1, e4.size(2), e4.size(3))  # (B, 32, H/32, W/32)

        # ---------------------
        # Bottleneck & Decoder
        # ---------------------
        # Concatenate encoder features and depth embedding
        center = torch.cat([e4, d_emb], dim=1)  # (B, 544, H/32, W/32)

        # Decoder 4
        d4 = self.dec4(center)  # (B, 256, H/16, W/16)
        d4 = d4 + e3  # Additive Skip Connection

        # Decoder 3
        d3 = self.dec3(d4)  # (B, 128, H/8, W/8)
        d3 = d3 + e2  # Additive Skip Connection

        # Decoder 2
        d2 = self.dec2(d3)  # (B, 64, H/4, W/4)
        d2 = d2 + e1  # Additive Skip Connection

        # Decoder 1
        d1 = self.dec1(d2)  # (B, 64, H/2, W/2)
        d1 = d1 + x0  # Additive Skip Connection (to pre-pool features)

        # Final Prediction
        out = self.final(d1)  # (B, 1, H, W)

        return out
