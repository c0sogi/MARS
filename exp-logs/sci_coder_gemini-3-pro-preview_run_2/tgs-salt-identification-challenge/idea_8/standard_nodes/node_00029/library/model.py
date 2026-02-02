import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import DEPTH_DROPOUT_RATE


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Uses an internal width of in_channels // 4 to preserve information flow,
    contrasted with standard LinkNet which often uses narrower bottlenecks.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # "Wide" configuration: internal dimension based on input channels
        # Ensure a minimum width (e.g., 16) to avoid collapse on small layers
        internal_channels = max(in_channels // 4, 16)

        # 1x1 Conv to reduce/project channels
        self.conv1 = nn.Conv2d(
            in_channels, internal_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 3x3 Transpose Conv for upsampling (Stride 2)
        self.trans_conv = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # 1x1 Conv to expand to output channels
        self.conv2 = nn.Conv2d(
            internal_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.trans_conv(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        return out


class DepthRegularizedWideLinkNet(nn.Module):
    """
    U-Shape architecture with ResNet34 encoder, Depth Injection at bottleneck,
    and Wide-LinkNet decoder blocks with additive skip connections.
    """

    def __init__(self, n_classes=1):
        super(DepthRegularizedWideLinkNet, self).__init__()

        # ---------------------------------------------------------------------
        # Encoder: ResNet34 (Pretrained)
        # ---------------------------------------------------------------------
        # Load pretrained weights
        base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Cite solution_lesson_node_00028: Modify first layer for 1-channel input
        # instead of repeating data. Sum weights to preserve initialization magnitude.
        original_conv1 = base_model.conv1
        base_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            base_model.conv1.weight.data = original_conv1.weight.data.sum(
                dim=1, keepdim=True
            )

        # Extract layers for feature access
        # Input Stem: Stride 2
        self.in_conv = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)
        self.maxpool = base_model.maxpool  # Stride 2 -> Total Stride 4

        # ResNet Layers
        self.layer1 = base_model.layer1  # 64 channels, Stride 1 (Total 4)
        self.layer2 = base_model.layer2  # 128 channels, Stride 2 (Total 8)
        self.layer3 = base_model.layer3  # 256 channels, Stride 2 (Total 16)
        self.layer4 = base_model.layer4  # 512 channels, Stride 2 (Total 32)

        # ---------------------------------------------------------------------
        # Depth Injection Mechanism
        # ---------------------------------------------------------------------
        self.depth_embedding_dim = 16
        self.depth_projector = nn.Linear(1, self.depth_embedding_dim)
        self.depth_dropout = nn.Dropout(p=DEPTH_DROPOUT_RATE)

        # ---------------------------------------------------------------------
        # Decoder: Wide-LinkNet
        # ---------------------------------------------------------------------
        # Bottleneck Input: 512 (Encoder) + 16 (Depth) = 528 channels

        # Block 4: 528 -> 256 (Matches layer3)
        self.dec4 = DecoderBlock(512 + self.depth_embedding_dim, 256)

        # Block 3: 256 -> 128 (Matches layer2)
        self.dec3 = DecoderBlock(256, 128)

        # Block 2: 128 -> 64 (Matches layer1)
        self.dec2 = DecoderBlock(128, 64)

        # Block 1: 64 -> 64 (Matches in_conv output)
        self.dec1 = DecoderBlock(64, 64)

        # ---------------------------------------------------------------------
        # Final Head
        # ---------------------------------------------------------------------
        # Upsample from Stride 2 (64ch) to Stride 1 (Original Resolution)
        self.final_trans_conv = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.final_bn = nn.BatchNorm2d(32)
        self.final_relu = nn.ReLU(inplace=True)

        # Final refinement
        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.final_bn2 = nn.BatchNorm2d(32)
        self.final_relu2 = nn.ReLU(inplace=True)

        # Prediction head
        self.head = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x, depth):
        """
        Args:
            x: Image tensor (B, 3, H, W)
            depth: Depth tensor (B,) or (B, 1)
        """
        # ---------------------------------------------------------------------
        # Encoder Pass
        # ---------------------------------------------------------------------
        x1 = self.in_conv(x)  # (B, 64, H/2, W/2)
        x_pool = self.maxpool(x1)  # (B, 64, H/4, W/4)
        x2 = self.layer1(x_pool)  # (B, 64, H/4, W/4)
        x3 = self.layer2(x2)  # (B, 128, H/8, W/8)
        x4 = self.layer3(x3)  # (B, 256, H/16, W/16)
        x5 = self.layer4(x4)  # (B, 512, H/32, W/32)

        # ---------------------------------------------------------------------
        # Depth Injection
        # ---------------------------------------------------------------------
        # Ensure depth is (B, 1)
        if depth.dim() == 1:
            depth = depth.unsqueeze(1)

        # Project and Dropout
        d_emb = self.depth_projector(depth.float())  # (B, 16)
        d_emb = self.depth_dropout(d_emb)

        # Reshape for spatial concatenation: (B, 16, 1, 1)
        d_emb = d_emb.unsqueeze(2).unsqueeze(3)

        # Expand to match feature map size (B, 16, H/32, W/32)
        d_emb = d_emb.expand(-1, -1, x5.size(2), x5.size(3))

        # Concatenate
        center = torch.cat([x5, d_emb], dim=1)  # (B, 528, H/32, W/32)

        # ---------------------------------------------------------------------
        # Decoder Pass (Additive Skip Connections)
        # ---------------------------------------------------------------------
        # Block 4
        d4 = self.dec4(center)  # (B, 256, H/16, W/16)
        d4 = d4 + x4  # Additive LinkNet Skip

        # Block 3
        d3 = self.dec3(d4)  # (B, 128, H/8, W/8)
        d3 = d3 + x3

        # Block 2
        d2 = self.dec2(d3)  # (B, 64, H/4, W/4)
        d2 = d2 + x2

        # Block 1
        d1 = self.dec1(d2)  # (B, 64, H/2, W/2)
        d1 = d1 + x1

        # ---------------------------------------------------------------------
        # Final Upsampling
        # ---------------------------------------------------------------------
        out = self.final_trans_conv(d1)  # (B, 32, H, W)
        out = self.final_bn(out)
        out = self.final_relu(out)

        out = self.final_conv(out)
        out = self.final_bn2(out)
        out = self.final_relu2(out)

        logits = self.head(out)  # (B, n_classes, H, W)

        return logits
