import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Computes a weighted average of the input sequence using a learned attention mechanism.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        # x shape: (Batch, Time, Features)

        # Compute attention scores
        # scores shape: (Batch, Time, 1)
        scores = self.attention(x)

        # Normalize scores across the time dimension
        weights = torch.softmax(scores, dim=1)

        # Weighted sum: (Batch, Features)
        # Broadcasting weights: (B, T, 1) * (B, T, F) -> (B, T, F) -> sum(dim=1)
        output = torch.sum(x * weights, dim=1)

        return output


class MultiResResNetCRNN(nn.Module):
    """
    Multi-Resolution ResNet-CRNN.

    Architecture:
    1. Input: 3-Channel Multi-Resolution Log-Mel Spectrogram.
    2. Backbone: ResNet34 (Pretrained).
       - Strides modified in layers 3 and 4 to preserve temporal resolution.
    3. Neck: Frequency Averaging + BiGRU.
    4. Head: Attention Pooling + Linear Classifier.
    """

    def __init__(self):
        super(MultiResResNetCRNN, self).__init__()

        # 1. Backbone: ResNet34
        weights = ResNet34_Weights.DEFAULT
        self.backbone = resnet34(weights=weights)

        # Modify strides to preserve temporal resolution
        # ResNet has 4 layers. Layer 1 (stride 1), Layer 2 (stride 2), Layer 3 (stride 2), Layer 4 (stride 2).
        # We want to prevent downsampling in time in Layer 3 and 4.

        # Layer 3
        self.backbone.layer3[0].conv1.stride = (1, 1)
        self.backbone.layer3[0].downsample[0].stride = (1, 1)

        # Layer 4
        self.backbone.layer4[0].conv1.stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)

        # Determine backbone output channels
        # ResNet34 layer 4 output channels = 512
        self.backbone_out_channels = 512

        # 2. Recurrent Layer (BiGRU)
        self.gru = nn.GRU(
            input_size=self.backbone_out_channels,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT,
        )

        # 3. Attention Pooling
        # BiGRU output dimension is hidden_size * 2
        gru_out_dim = Config.HIDDEN_SIZE * 2
        self.attn_pooling = AttentionPooling(gru_out_dim)

        # 4. Classifier
        self.classifier = nn.Linear(gru_out_dim, Config.NUM_CLASSES)

    def forward(self, x):
        # Input x: (Batch, 3, F, T)

        # Pass through Backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # Output: (Batch, 512, F', T')

        # Average over Frequency dimension
        x = x.mean(dim=2)  # (Batch, 512, T')

        # Permute for RNN: (Batch, T', 512)
        x = x.permute(0, 2, 1)

        # Pass through BiGRU
        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # (Batch, T', 2*Hidden)

        # Attention Pooling
        x = self.attn_pooling(x)  # (Batch, 2*Hidden)

        # Classification
        x = self.classifier(x)  # (Batch, NumClasses)

        return x
