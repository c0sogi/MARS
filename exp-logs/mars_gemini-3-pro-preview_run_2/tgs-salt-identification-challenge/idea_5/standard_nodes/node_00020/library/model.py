import torch
import torch.nn as nn
import torchvision
from library.config import Config


class DecoderBlock(nn.Module):
    """
    LinkNet Decoder Block.
    Structure: 1x1 Conv (Reduce) -> 3x3 Transpose Conv (Upsample) -> 1x1 Conv (Expand).
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Internal width heuristic: in_channels // 4
        # We clamp the minimum channels to ensure capacity at lower depths
        mid_channels = max(in_channels // 4, 16)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # Upsampling: Doubles the spatial dimension
        self.deconv2 = nn.ConvTranspose2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


class DepthAwareLinkNet(nn.Module):
    """
    Depth-Conditioned ResNet34-LinkNet.

    Encoder: ResNet34 (pretrained)
    Decoder: LinkNet (additive skips)
    Bottleneck: Direct pass + Depth Injection (No compression)
    """

    def __init__(self):
        super(DepthAwareLinkNet, self).__init__()

        # ==========================
        # Encoder (ResNet34)
        # ==========================
        # Use updated weights API
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        self.encoder = torchvision.models.resnet34(weights=weights)

        # Adapt first layer for 1-channel input if necessary
        if Config.CHANNELS != 3:
            old_conv = self.encoder.conv1
            new_conv = nn.Conv2d(
                Config.CHANNELS,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias,
            )
            # Initialize with average of ImageNet weights
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            self.encoder.conv1 = new_conv

        # Expose encoder layers for easier access in forward pass
        self.conv1 = self.encoder.conv1  # 128 -> 64
        self.bn1 = self.encoder.bn1
        self.relu = self.encoder.relu
        self.maxpool = self.encoder.maxpool  # 64 -> 32

        self.layer1 = self.encoder.layer1  # 32 -> 32 (64 ch)
        self.layer2 = self.encoder.layer2  # 32 -> 16 (128 ch)
        self.layer3 = self.encoder.layer3  # 16 -> 8  (256 ch)
        self.layer4 = self.encoder.layer4  # 8 -> 4   (512 ch)

        # ==========================
        # Depth Injection
        # ==========================
        self.use_depth = Config.USE_DEPTH
        if self.use_depth:
            self.depth_emb_dim = Config.DEPTH_EMBEDDING_DIM
            self.depth_projector = nn.Sequential(
                nn.Linear(1, self.depth_emb_dim), nn.ReLU(inplace=True)
            )
            # Bottleneck channels = Encoder Output + Depth Embedding
            bottleneck_channels = 512 + self.depth_emb_dim
        else:
            bottleneck_channels = 512

        # ==========================
        # Decoder (LinkNet)
        # ==========================
        # LinkNet adds encoder features to decoder outputs.
        # Decoder output channels must match encoder skip channels.

        # Decoder 4: Input Bottleneck -> Output matches Layer 3 (256)
        self.decoder4 = DecoderBlock(bottleneck_channels, 256)

        # Decoder 3: Input 256 -> Output matches Layer 2 (128)
        self.decoder3 = DecoderBlock(256, 128)

        # Decoder 2: Input 128 -> Output matches Layer 1 (64)
        self.decoder2 = DecoderBlock(128, 64)

        # Decoder 1: Input 64 -> Output matches Conv1/Stem (64)
        self.decoder1 = DecoderBlock(64, 64)

        # ==========================
        # Final Head
        # ==========================
        # Upsample from 64x64 to 128x128
        self.final_deconv = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.final_bn = nn.BatchNorm2d(32)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x, depth=None):
        """
        Args:
            x: Image tensor (B, C, H, W)
            depth: Depth tensor (B, 1) or (B,)
        """
        # ---------------------
        # Encoder
        # ---------------------
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)  # (B, 64, H/2, W/2) -> Skip for Decoder 1

        x_mp = self.maxpool(x0)

        # Blocks
        e1 = self.layer1(x_mp)  # (B, 64, H/4, W/4)  -> Skip for Decoder 2
        e2 = self.layer2(e1)  # (B, 128, H/8, W/8) -> Skip for Decoder 3
        e3 = self.layer3(e2)  # (B, 256, H/16, W/16) -> Skip for Decoder 4
        e4 = self.layer4(e3)  # (B, 512, H/32, W/32) -> Bottleneck

        # ---------------------
        # Depth Injection
        # ---------------------
        if self.use_depth and depth is not None:
            if depth.dim() == 1:
                depth = depth.unsqueeze(1)  # Ensure (B, 1)

            # Project depth
            d_emb = self.depth_projector(depth)  # (B, 16)

            # Expand to spatial dimensions of bottleneck
            d_emb = d_emb.unsqueeze(2).unsqueeze(3)  # (B, 16, 1, 1)
            d_emb = d_emb.expand(-1, -1, e4.size(2), e4.size(3))  # (B, 16, H/32, W/32)

            # Concatenate
            bottleneck = torch.cat([e4, d_emb], dim=1)
        else:
            bottleneck = e4

        # ---------------------
        # Decoder
        # ---------------------
        # D4: Upsample bottleneck, Add e3
        d4 = self.decoder4(bottleneck)
        d4 = d4 + e3

        # D3: Upsample d4, Add e2
        d3 = self.decoder3(d4)
        d3 = d3 + e2

        # D2: Upsample d3, Add e1
        d2 = self.decoder2(d3)
        d2 = d2 + e1

        # D1: Upsample d2, Add x0 (Stem output)
        d1 = self.decoder1(d2)
        d1 = d1 + x0

        # ---------------------
        # Head
        # ---------------------
        out = self.final_deconv(d1)
        out = self.final_bn(out)
        out = self.final_relu(out)
        out = self.final_conv(out)

        return out
