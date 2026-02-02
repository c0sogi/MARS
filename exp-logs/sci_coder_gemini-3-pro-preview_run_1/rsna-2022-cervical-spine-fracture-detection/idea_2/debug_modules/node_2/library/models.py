import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from library.config import Config


class SpineLocalizer(nn.Module):
    """
    U-Net architecture with a ResNet18 encoder for spine localization.
    Outputs a segmentation mask (logits) indicating the presence of the spine.
    """

    def __init__(self, pretrained=True):
        super(SpineLocalizer, self).__init__()

        # Load ResNet18 backbone
        backbone = models.resnet18(pretrained=pretrained)

        # Encoder layers
        self.encoder_conv1 = backbone.conv1
        self.encoder_bn1 = backbone.bn1
        self.encoder_relu = backbone.relu
        self.encoder_maxpool = backbone.maxpool
        self.encoder_layer1 = backbone.layer1
        self.encoder_layer2 = backbone.layer2
        self.encoder_layer3 = backbone.layer3
        self.encoder_layer4 = backbone.layer4

        # Decoder layers
        # Layer 4 output: 512 channels, 1/32 scale
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(256 + 256, 256)  # Cat with layer3 (256)

        # Layer 3 output: 256 channels, 1/16 scale
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(128 + 128, 128)  # Cat with layer2 (128)

        # Layer 2 output: 128 channels, 1/8 scale
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(64 + 64, 64)  # Cat with layer1 (64)

        # Layer 1 output: 64 channels, 1/4 scale
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64 + 64, 64)  # Cat with conv1/relu output (64)

        # Final upsampling to original size
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = self._conv_block(32, 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Adapt 1-channel input to 3-channel backbone expectation
        if x.shape[1] == 1:
            x_in = x.repeat(1, 3, 1, 1)
        else:
            x_in = x

        # Encoder
        x0 = self.encoder_conv1(x_in)
        x0 = self.encoder_bn1(x0)
        x0 = self.encoder_relu(x0)  # 1/2 scale, 64 ch

        x1 = self.encoder_maxpool(x0)  # 1/4 scale, 64 ch
        x1 = self.encoder_layer1(x1)  # 1/4 scale, 64 ch
        x2 = self.encoder_layer2(x1)  # 1/8 scale, 128 ch
        x3 = self.encoder_layer3(x2)  # 1/16 scale, 256 ch
        x4 = self.encoder_layer4(x3)  # 1/32 scale, 512 ch

        # Decoder with Skip Connections
        u4 = self.up4(x4)
        if u4.size()[2:] != x3.size()[2:]:
            u4 = F.interpolate(
                u4, size=x3.shape[2:], mode="bilinear", align_corners=False
            )
        d4 = torch.cat([u4, x3], dim=1)
        d4 = self.dec4(d4)

        u3 = self.up3(d4)
        if u3.size()[2:] != x2.size()[2:]:
            u3 = F.interpolate(
                u3, size=x2.shape[2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([u3, x2], dim=1)
        d3 = self.dec3(d3)

        u2 = self.up2(d3)
        if u2.size()[2:] != x1.size()[2:]:
            u2 = F.interpolate(
                u2, size=x1.shape[2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([u2, x1], dim=1)
        d2 = self.dec2(d2)

        u1 = self.up1(d2)
        if u1.size()[2:] != x0.size()[2:]:
            u1 = F.interpolate(
                u1, size=x0.shape[2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([u1, x0], dim=1)
        d1 = self.dec1(d1)

        u0 = self.up0(d1)
        # Interpolate to original input size if necessary
        if u0.size()[2:] != x.size()[2:]:
            u0 = F.interpolate(
                u0, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        d0 = self.dec0(u0)

        out = self.final_conv(d0)
        return out


class FractureClassifier(nn.Module):
    """
    2.5D CNN for fracture classification.
    Uses EfficientNetV2-S backbone to predict 8 targets from a stack of slices.
    """

    def __init__(self, model_name=Config.CLS_MODEL_ARCH, pretrained=True):
        super(FractureClassifier, self).__init__()

        # Create model using timm
        # in_chans=3 corresponds to the 2.5D stacking (current, +1, -1 slices)
        # num_classes=8 corresponds to C1-C7 + patient_overall
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=Config.NUM_SLICES_25D,
            num_classes=Config.NUM_CLASSES,
        )

    def forward(self, x):
        # x shape: (Batch, 3, Crop_Size, Crop_Size)
        # Returns logits: (Batch, 8)
        return self.model(x)
