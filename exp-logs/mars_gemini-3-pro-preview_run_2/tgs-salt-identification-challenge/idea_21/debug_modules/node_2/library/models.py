import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Performs dimensionality reduction, upsampling, and expansion.
    Designed to be used with additive skip connections.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Internal width is in_channels // 4 as per LinkNet design
        # For "Wide" variants, this ratio might differ, but standard LinkNet uses /4
        mid_channels = in_channels // 4

        # 1x1 Conv to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 Transpose Conv for upsampling
        self.deconv = nn.ConvTranspose2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 Conv to expand dimensions to match skip connection
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


class BaseResNet34(nn.Module):
    """
    Base class containing the ResNet34 encoder adapted for 1-channel input.
    """

    def __init__(self):
        super(BaseResNet34, self).__init__()

        # Load pretrained ResNet34
        # Handling different torchvision versions for weights
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            self.resnet = models.resnet34(weights=weights)
        except:
            self.resnet = models.resnet34(pretrained=True)

        # Modify first layer for 1-channel input
        # We sum the weights of the original 3 channels to preserve intensity patterns
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            Config.IN_CHANNELS,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        with torch.no_grad():
            self.resnet.conv1.weight.data = original_conv1.weight.data.sum(
                dim=1, keepdim=True
            )

        # Expose Encoder layers
        self.encoder0 = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu
        )
        self.maxpool = self.resnet.maxpool
        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4

    def forward_encoder(self, x):
        """
        Returns features from all levels of the encoder.
        e0: (64, H/2, W/2)
        e1: (64, H/4, W/4)
        e2: (128, H/8, W/8)
        e3: (256, H/16, W/16)
        e4: (512, H/32, W/32)
        """
        e0 = self.encoder0(x)
        mp = self.maxpool(e0)
        e1 = self.encoder1(mp)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        return e0, e1, e2, e3, e4


class PrivilegedTeacher(BaseResNet34):
    """
    Teacher model that uses Ground Truth depth information.
    Depth is projected and concatenated at the bottleneck.
    """

    def __init__(self):
        super(PrivilegedTeacher, self).__init__()

        # Depth Projection MLP
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Decoder
        # Bottleneck input: Encoder (512) + Depth (32) = 544
        # Target output: Match e3 channels (256)
        self.decoder4 = DecoderBlock(512 + 32, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        # Final upsampling block (from H/2 to H)
        self.final_deconv = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.final_bn = nn.BatchNorm2d(32)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, z):
        # Encoder
        e0, e1, e2, e3, e4 = self.forward_encoder(x)

        # Process Depth
        if z.dim() == 1:
            z = z.unsqueeze(1)
        z_feat = self.depth_mlp(z.float())  # (N, 32)

        # Broadcast depth features to spatial dimensions of bottleneck
        z_feat = z_feat.unsqueeze(2).unsqueeze(3)  # (N, 32, 1, 1)
        z_feat = z_feat.expand(-1, -1, e4.size(2), e4.size(3))  # (N, 32, H/32, W/32)

        # Concatenate
        bottleneck = torch.cat([e4, z_feat], dim=1)  # (N, 544, H/32, W/32)

        # Decoder with Additive Skip Connections
        d4 = self.decoder4(bottleneck)
        d4 = d4 + e3

        d3 = self.decoder3(d4)
        d3 = d3 + e2

        d2 = self.decoder2(d3)
        d2 = d2 + e1

        d1 = self.decoder1(d2)
        d1 = d1 + e0

        # Final Prediction
        out = self.final_deconv(d1)
        out = self.final_bn(out)
        out = self.final_relu(out)
        logits = self.final_conv(out)

        return logits


class MultiTaskStudent(BaseResNet34):
    """
    Student model that takes only images.
    Includes an auxiliary depth regression head to learn depth-correlated features.
    """

    def __init__(self):
        super(MultiTaskStudent, self).__init__()

        # Auxiliary Depth Regression Head
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Decoder
        # Bottleneck input: Encoder (512)
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        # Final upsampling block
        self.final_deconv = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.final_bn = nn.BatchNorm2d(32)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0, e1, e2, e3, e4 = self.forward_encoder(x)

        # Auxiliary Depth Prediction
        depth_pred = self.aux_head(e4)

        # Decoder with Additive Skip Connections
        d4 = self.decoder4(e4)
        d4 = d4 + e3

        d3 = self.decoder3(d4)
        d3 = d3 + e2

        d2 = self.decoder2(d3)
        d2 = d2 + e1

        d1 = self.decoder1(d2)
        d1 = d1 + e0

        # Final Prediction
        out = self.final_deconv(d1)
        out = self.final_bn(out)
        out = self.final_relu(out)
        logits = self.final_conv(out)

        return logits, depth_pred
