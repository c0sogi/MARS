import torch
import torch.nn as nn
import torchvision.models as models
import math
from library.config import Config


class MonoCenterNet(nn.Module):
    def __init__(self):
        super(MonoCenterNet, self).__init__()

        # 1. Backbone: ResNet34
        # We load the pretrained model and cut it off at layer4
        # ResNet34 layer4 output channels: 512, stride: 32
        resnet = models.resnet34(weights="DEFAULT")
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # 2. Neck: Upsampling layers (Deconv)
        # We need to go from stride 32 to stride 4 (3 upsampling steps)
        # Channels: 512 -> 256 -> 128 -> 64
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 3. Heads
        head_conv = Config.HEAD_CONV  # Typically 64

        # Heatmap Head (Class probabilities)
        self.hm_head = self._make_head(64, head_conv, Config.NUM_CLASSES)

        # Depth Head (1 channel)
        self.depth_head = self._make_head(64, head_conv, 1)

        # Dimensions Head (3 channels: w, l, h)
        self.dim_head = self._make_head(64, head_conv, 3)

        # Rotation Head (2 channels: sin, cos)
        self.rot_head = self._make_head(64, head_conv, 2)

        # Offset Head (2 channels: x, y correction)
        self.off_head = self._make_head(64, head_conv, 2)

        # 4. Initialization
        self._init_weights()

    def _make_head(self, in_channels, inter_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inter_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def _init_weights(self):
        # Initialize Neck
        for m in self.neck.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize Heads
        # Standard initialization for regression heads
        for head in [self.depth_head, self.dim_head, self.rot_head, self.off_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Special initialization for Heatmap Head
        # Initialize bias to -2.19 (corresponds to prior prob 0.1) to prevent instability
        # b = -log((1 - pi) / pi) where pi = 0.1 -> b approx -2.19
        for m in self.hm_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Set the bias of the final 1x1 conv layer of the heatmap head
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # Backbone
        x = self.backbone(x)

        # Neck
        features = self.neck(x)

        # Heads
        out = {
            "hm": self.hm_head(features),
            "depth": self.depth_head(features),
            "dim": self.dim_head(features),
            "rot": self.rot_head(features),
            "offset": self.off_head(features),
        }

        return out
