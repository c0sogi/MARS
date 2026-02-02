import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from library.config import Config

# -----------------------------------------------------------------------------
# 1. Modified Wide SE-ResNet
# -----------------------------------------------------------------------------


class SEBlock(nn.Module):
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
        return x * y


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ModifiedWideSEResNet(nn.Module):
    def __init__(self, num_classes=1, widen_factor=4):
        super(ModifiedWideSEResNet, self).__init__()
        self.in_planes = 16

        # Modified Stem: 3x3 Conv, stride 1 to preserve 32x32 resolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Layers with increased width
        self.layer1 = self._make_layer(16 * widen_factor, 2, stride=1)
        self.layer2 = self._make_layer(32 * widen_factor, 2, stride=2)
        self.layer3 = self._make_layer(64 * widen_factor, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * widen_factor, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# -----------------------------------------------------------------------------
# 2. Modified DenseNet-BC
# -----------------------------------------------------------------------------


class ModifiedDenseNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ModifiedDenseNet, self).__init__()
        # Load standard DenseNet121
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # Modify Stem for 32x32 input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New: Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Remove MaxPool0 to preserve spatial dimensions early on
        self.model.features.pool0 = nn.Identity()

        # Modify Classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# -----------------------------------------------------------------------------
# 3. Modified EfficientNet
# -----------------------------------------------------------------------------


class ModifiedEfficientNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ModifiedEfficientNet, self).__init__()
        # Use timm to create EfficientNet B0
        self.model = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=num_classes
        )

        # Modify Stem for 32x32 input
        # Original stem usually has stride 2. We want stride 1.
        original_stem = self.model.conv_stem
        out_channels = original_stem.out_channels

        # Create new stem with stride 1
        self.model.conv_stem = nn.Conv2d(
            3, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Attempt to copy weights if shapes match (kernel size is usually 3x3 for both)
        if original_stem.weight.shape == self.model.conv_stem.weight.shape:
            with torch.no_grad():
                self.model.conv_stem.weight.copy_(original_stem.weight)

    def forward(self, x):
        return self.model(x)


# -----------------------------------------------------------------------------
# 4. Compact Convolutional Transformer (CCT)
# -----------------------------------------------------------------------------


class Tokenizer(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        pooling_kernel,
        pooling_stride,
        pooling_padding,
        n_conv_layers,
        n_input_channels,
        n_output_channels,
    ):
        super(Tokenizer, self).__init__()
        n_filter_list = [n_input_channels] + [
            n_output_channels for _ in range(n_conv_layers)
        ]

        self.conv_layers = nn.Sequential()
        for n in range(n_conv_layers):
            self.conv_layers.add_module(
                f"conv_{n}",
                nn.Conv2d(
                    n_filter_list[n],
                    n_filter_list[n + 1],
                    kernel_size=(kernel_size, kernel_size),
                    stride=(stride, stride),
                    padding=(padding, padding),
                    bias=False,
                ),
            )
            self.conv_layers.add_module(f"relu_{n}", nn.ReLU(inplace=True))
            self.conv_layers.add_module(
                f"pool_{n}",
                nn.MaxPool2d(
                    kernel_size=pooling_kernel,
                    stride=pooling_stride,
                    padding=pooling_padding,
                ),
            )

    def forward(self, x):
        return self.conv_layers(x)


class SequencePooling(nn.Module):
    def __init__(self, embedding_dim):
        super(SequencePooling, self).__init__()
        self.attention_pool = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x: (B, N, D)
        w = self.attention_pool(x)  # (B, N, 1)
        w = F.softmax(w, dim=1)  # (B, N, 1)
        out = torch.matmul(x.transpose(1, 2), w)  # (B, D, N) * (B, N, 1) -> (B, D, 1)
        out = out.squeeze(-1)  # (B, D)
        return out


class CCT(nn.Module):
    def __init__(self, num_classes=1):
        super(CCT, self).__init__()
        params = Config.CCT_PARAMS

        self.tokenizer = Tokenizer(
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            padding=params["padding"],
            pooling_kernel=params["pooling_kernel"],
            pooling_stride=params["pooling_stride"],
            pooling_padding=params["pooling_padding"],
            n_conv_layers=params["n_conv_layers"],
            n_input_channels=3,
            n_output_channels=params["embedding_dim"],
        )

        self.embedding_dim = params["embedding_dim"]
        self.num_layers = params["num_layers"]
        self.num_heads = params["num_heads"]
        self.mlp_ratio = params["mlp_ratio"]
        self.dropout = params["dropout"]
        self.attention_dropout = params["attention_dropout"]
        self.stochastic_depth = params["stochastic_depth"]

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.embedding_dim * self.mlp_ratio),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Sequence Pooling
        self.seq_pool = SequencePooling(self.embedding_dim)

        # Classifier
        self.fc = nn.Linear(self.embedding_dim, num_classes)

        # Positional Embedding (Learnable)
        # Calculate sequence length based on 32x32 input and tokenizer params
        # Conv: 32x32 (s=1, p=1) -> 32x32
        # Pool: 32x32 (k=3, s=2, p=1) -> 16x16
        # Seq Len = 16 * 16 = 256
        self.seq_len = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embedding_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # Tokenizer
        x = self.tokenizer(x)  # (B, D, H, W)

        # Flatten
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # (B, N, D)

        # Add Positional Embedding
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        else:
            # Handle potential size mismatches if input size changes
            x = x + self.pos_embed[:, : x.shape[1], :]

        # Transformer
        x = self.transformer(x)

        # Sequence Pooling
        x = self.seq_pool(x)

        # Classifier
        x = self.fc(x)
        return x


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------


def get_model(model_name, num_classes=1):
    """
    Factory function to instantiate models by name.

    Args:
        model_name (str): Name of the model architecture.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: The requested model.
    """
    if model_name == "wide_se_resnet":
        return ModifiedWideSEResNet(num_classes=num_classes)
    elif model_name == "densenet_bc":
        return ModifiedDenseNet(num_classes=num_classes)
    elif model_name == "efficientnet":
        return ModifiedEfficientNet(num_classes=num_classes)
    elif model_name == "cct":
        return CCT(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
