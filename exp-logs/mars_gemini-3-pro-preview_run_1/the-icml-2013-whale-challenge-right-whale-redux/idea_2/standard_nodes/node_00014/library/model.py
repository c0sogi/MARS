import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling mechanism to dynamically weight time steps.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, Config.ATTENTION_DIM),
            nn.Tanh(),
            nn.Linear(Config.ATTENTION_DIM, 1),
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Feature_Dim)

        # Calculate attention scores
        # scores shape: (Batch, Seq_Len, 1)
        scores = self.attention(x)

        # Normalize scores to weights
        weights = torch.softmax(scores, dim=1)

        # Compute weighted sum
        # weighted_sum shape: (Batch, Feature_Dim)
        weighted_sum = torch.sum(x * weights, dim=1)

        return weighted_sum


class CRNN(nn.Module):
    """
    CRNN Architecture: ResNet-18 Backbone + BiGRU + Attention Pooling.
    """

    def __init__(self):
        super(CRNN, self).__init__()

        # ==========================================
        # 1. Deep Spectral Feature Extraction (ResNet-18)
        # ==========================================
        # Load standard ResNet18 structure
        resnet = models.resnet18(weights=None)

        # Modify first conv layer: 3 channels -> 1 channel
        # Original: kernel=7, stride=2, padding=3
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the classification head (AvgPool and FC)
        # We keep layers up to layer4
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # ==========================================
        # 2. Temporal Context Modeling (BiGRU)
        # ==========================================
        # ResNet-18 layer4 output has 512 channels
        self.feature_dim = 512

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=Config.GRU_HIDDEN_DIM,
            num_layers=Config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.GRU_NUM_LAYERS > 1 else 0,
        )

        # ==========================================
        # 3. Attention-Based Aggregation
        # ==========================================
        # BiGRU output dimension is hidden_dim * 2 (directions)
        self.gru_out_dim = Config.GRU_HIDDEN_DIM * 2
        self.attention = AttentionPooling(self.gru_out_dim)

        # ==========================================
        # 4. Classifier
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT), nn.Linear(self.gru_out_dim, Config.NUM_CLASSES)
        )

    def forward(self, x):
        # Input x: (Batch, 1, F, T) where F=64

        # 1. Extract Features
        # Output: (Batch, 512, F', T')
        # ResNet reduces dimensions by factor of 32.
        # F=64 -> F'=2. T depends on input duration.
        x = self.backbone(x)

        # 2. Prepare for RNN
        # Collapse Frequency dimension (Average Pooling) -> (Batch, 512, T')
        x = torch.mean(x, dim=2)

        # Permute to (Batch, T', 512) for GRU
        x = x.permute(0, 2, 1)

        # 3. Temporal Modeling
        # Output: (Batch, T', Hidden*2)
        x, _ = self.gru(x)

        # 4. Attention Aggregation
        # Output: (Batch, Hidden*2)
        x = self.attention(x)

        # 5. Classification
        # Output: (Batch, 1) - Logits
        logits = self.classifier(x)

        return logits
