import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import library.config as config


class BEVDetector(nn.Module):
    """
    Rasterized BEV-CNN Object Detector.

    Architecture:
    1. Backbone: ResNet-18 (extracts features, downsamples to 1/32)
    2. Decoder: Upsampling layers (recovers resolution to 1/4)
    3. Heads:
       - Heatmap (Class probabilities)
       - Regression (3D Box parameters)
    """

    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
    ):
        super(BEVDetector, self).__init__()

        # ----------------------------------------------------------------------
        # 1. Backbone
        # ----------------------------------------------------------------------
        # Load pretrained ResNet
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone_channels = 512  # Output channels of layer4
        elif backbone_name == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            self.backbone_channels = 512
        else:
            # Fallback to ResNet18 if other not implemented
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone_channels = 512

        # We do not need the classification head of the backbone
        del self.backbone.avgpool
        del self.backbone.fc

        # ----------------------------------------------------------------------
        # 2. Decoder (Neck)
        # ----------------------------------------------------------------------
        # We need to upsample from 1/32 (Layer4) to 1/4 (Output)
        # This requires 3 stages of upsampling (x2 each)

        self.decoder = nn.Sequential(
            # Stage 1: 1/32 -> 1/16
            nn.Conv2d(
                self.backbone_channels, 256, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Stage 2: 1/16 -> 1/8
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Stage 3: 1/8 -> 1/4
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # ----------------------------------------------------------------------
        # 3. Detection Heads
        # ----------------------------------------------------------------------
        head_conv = config.HEAD_CONV  # Generally 64

        # Heatmap Head: Predicts object center probability
        # Output: (B, num_classes, H/4, W/4)
        self.hm_head = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1),
        )

        # Regression Head: Predicts 8 values per pixel
        # [offset_x, offset_y, z, log(w), log(l), log(h), sin(yaw), cos(yaw)]
        # Output: (B, 8, H/4, W/4)
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 8, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights.
        Crucial for the heatmap head to prevent loss explosion at start.
        """
        # Initialize Decoder and Intermediate Head layers
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.hm_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.reg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Special initialization for the final heatmap classification layer
        # We set the bias such that the prior probability of an object is low (e.g., 0.1)
        # bias = -log((1 - p) / p) -> for p=0.1, bias ~ -2.19
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input BEV tensor of shape (B, C, H, W)

        Returns:
            hm (torch.Tensor): Heatmap logits (B, NumClasses, H_out, W_out)
            reg (torch.Tensor): Regression map (B, 8, H_out, W_out)
        """
        input_size = x.shape[2:]

        # Calculate target output size based on down ratio
        # We enforce this size at the end to handle potential rounding in the backbone
        target_h = input_size[0] // config.DOWN_RATIO
        target_w = input_size[1] // config.DOWN_RATIO

        # 1. Backbone Forward
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # Output is 1/32 stride

        # 2. Decoder Forward
        x = self.decoder(x)

        # 3. Heads Forward (Cite solution_lesson_node_00013)
        # Apply heads on native resolution to preserve spatial integrity
        hm = self.hm_head(x)
        reg = self.reg_head(x)

        # 4. Resize Outputs if necessary
        # Interpolate predictions to match target grid instead of intermediate features
        if hm.shape[2] != target_h or hm.shape[3] != target_w:
            hm = F.interpolate(
                hm, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            reg = F.interpolate(
                reg, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        return hm, reg
