import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    """
    Modified BasicBlock with Squeeze-and-Excitation inserted on the residual branch.
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Standard ResNet BasicBlock components
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes)

        # SE Block
        self.se = SEBlock(planes, reduction=16)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE Block before residual connection
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEResNet18(nn.Module):
    """
    ResNet18 backbone modified with SE blocks and adapted for 10-channel input.
    """

    def __init__(self):
        super(SEResNet18, self).__init__()

        # Construct the structure using the custom SEBasicBlock
        # ResNet18 structure is [2, 2, 2, 2] blocks
        self.backbone = ResNet(block=SEBasicBlock, layers=[2, 2, 2, 2])

        # Modify the first convolution layer for 10 input channels
        # Standard ResNet: conv1 is (3, 64, 7, 7)
        self.backbone.conv1 = nn.Conv2d(
            Config.NUM_SENSORS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the final FC layer as we only want features
        self.backbone.fc = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        # 1. Load standard ResNet18 weights
        # We use weights='IMAGENET1K_V1' equivalent to pretrained=True
        try:
            standard_resnet = models.resnet18(weights="IMAGENET1K_V1")
        except:
            # Fallback for older torchvision versions
            standard_resnet = models.resnet18(pretrained=True)

        state_dict = standard_resnet.state_dict()
        my_state_dict = self.backbone.state_dict()

        # 2. Transfer weights
        for k, v in state_dict.items():
            if k in my_state_dict:
                if k == "conv1.weight":
                    # Adapt 3-channel weights to 10-channel weights
                    # Shape: (64, 3, 7, 7) -> (64, 10, 7, 7)
                    # Average across RGB channels
                    avg_weight = torch.mean(v, dim=1, keepdim=True)  # (64, 1, 7, 7)
                    # Replicate 10 times
                    new_weight = avg_weight.repeat(1, Config.NUM_SENSORS, 1, 1)
                    my_state_dict[k].copy_(new_weight)
                elif k == "fc.weight" or k == "fc.bias":
                    # Skip the final FC layer as we replaced it with Identity
                    continue
                elif v.shape == my_state_dict[k].shape:
                    # Direct copy for matching shapes (convs, bns in blocks)
                    my_state_dict[k].copy_(v)
                else:
                    # Shape mismatch (should not happen for basic layers if architecture matches)
                    pass

        # SE blocks are initialized randomly by default (PyTorch init), which is appropriate.

    def forward(self, x):
        # x shape: (B, 10, Freq, Time)
        # ResNet expects (B, C, H, W). Spectrograms are treated as images.
        return self.backbone(x)


class TabularMLP(nn.Module):
    """
    Wide Multi-Layer Perceptron for processing statistical features.
    """

    def __init__(self, input_dim, hidden_dim=Config.MLP_HIDDEN_DIM):
        super(TabularMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.net(x)


class HybridModel(nn.Module):
    """
    Spectrally-Enhanced SE-ResNet Hybrid Model.
    Fuses visual spectrogram features (via SE-ResNet18) and statistical features (via MLP).
    """

    def __init__(self, tabular_input_dim):
        super(HybridModel, self).__init__()

        # Branch 1: Spectrogram Encoder
        self.cnn_encoder = SEResNet18()
        # ResNet18 output dim is 512 before the final FC
        cnn_out_dim = 512

        # Branch 2: Tabular Encoder
        self.mlp_encoder = TabularMLP(input_dim=tabular_input_dim)
        mlp_out_dim = 128

        # Fusion Head
        fusion_dim = cnn_out_dim + mlp_out_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),  # Regression output
        )

    def forward(self, spec, tabular):
        """
        Args:
            spec: Tensor of shape (B, 10, Freq, Time)
            tabular: Tensor of shape (B, tabular_input_dim)
        """
        # Extract features
        cnn_feats = self.cnn_encoder(spec)  # (B, 512)
        mlp_feats = self.mlp_encoder(tabular)  # (B, 128)

        # Concatenate
        combined = torch.cat([cnn_feats, mlp_feats], dim=1)

        # Predict
        out = self.head(combined)

        # Flatten to (B,)
        return out.squeeze(1)
