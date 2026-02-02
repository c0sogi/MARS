import torch
import torch.nn as nn
import torchvision
from library import config


class ResNetBackbone(nn.Module):
    """
    ResNet-34 Backbone for feature extraction.
    Uses pretrained weights from ImageNet.
    """

    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # Load pretrained ResNet34
        try:
            weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            base = torchvision.models.resnet34(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            base = torchvision.models.resnet34(pretrained=True)

        # Extract layers up to layer4
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Output stride 32, 512 channels
        return x


class Neck(nn.Module):
    """
    Upsampling Neck using Deconvolution layers.
    Upsamples features from stride 32 to stride 4 (1/4 resolution).
    """

    def __init__(self, in_channels, out_channels):
        super(Neck, self).__init__()
        # 512 -> 256 (Stride 16) -> 256 (Stride 8) -> out_channels (Stride 4)
        # Maintained width to prevent information bottleneck (Cite solution_lesson_node_00001)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                256, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                256, out_channels, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.deconv_layers(x)


class HeatmapHead(nn.Module):
    """
    Predicts the 'textness' score (heatmap).
    Output: 1 channel.
    """

    def __init__(self, in_channels):
        super(HeatmapHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Initialize bias for focal loss stability (sigmoid(-2.19) approx 0.1)
        self.conv[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        return self.conv(x)


class OffsetHead(nn.Module):
    """
    Predicts local sub-pixel offsets.
    Output: 2 channels (dx, dy).
    """

    def __init__(self, in_channels):
        super(OffsetHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv(x)


class EmbeddingHead(nn.Module):
    """
    Predicts dense embedding vectors for classification.
    Output: 64 channels.
    """

    def __init__(self, in_channels, emb_dim):
        super(EmbeddingHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, emb_dim, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ClassifierMLP(nn.Module):
    """
    Shared MLP to classify sampled embedding vectors.
    """

    def __init__(self, in_dim, num_classes):
        super(ClassifierMLP, self).__init__()
        # Increased capacity to handle 3848 classes (Cite solution_lesson_node_00001)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class SparseCenterNet(nn.Module):
    """
    Main Sparse-Query CenterNet Architecture.
    """

    def __init__(self):
        super(SparseCenterNet, self).__init__()

        # 1. Backbone
        self.backbone = ResNetBackbone()

        # 2. Neck (512 input channels from ResNet18 layer4 -> 64 output channels)
        self.neck = Neck(in_channels=512, out_channels=64)

        # 3. Heads
        self.hm_head = HeatmapHead(in_channels=64)
        self.reg_head = OffsetHead(in_channels=64)
        self.emb_head = EmbeddingHead(in_channels=64, emb_dim=config.HEAD_CHANNELS)

        # 4. Classifier (Stored as submodule for optimizer to track)
        self.classifier = ClassifierMLP(
            in_dim=config.HEAD_CHANNELS, num_classes=config.NUM_CLASSES
        )

    def forward(self, x):
        """
        Forward pass generating dense maps.

        Args:
            x (torch.Tensor): Input images (B, 3, H, W)

        Returns:
            hm (torch.Tensor): Heatmap logits (B, 1, H/4, W/4)
            reg (torch.Tensor): Offsets (B, 2, H/4, W/4)
            emb (torch.Tensor): Embeddings (B, 64, H/4, W/4)
        """
        # Feature extraction
        feat = self.backbone(x)

        # Upsampling
        feat = self.neck(feat)

        # Head predictions
        hm = self.hm_head(feat)
        reg = self.reg_head(feat)
        emb = self.emb_head(feat)

        return hm, reg, emb
