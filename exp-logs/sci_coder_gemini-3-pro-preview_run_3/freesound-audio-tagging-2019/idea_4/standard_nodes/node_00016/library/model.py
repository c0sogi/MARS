import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling module to weigh time frames based on their relevance.
    Structure: Linear -> Tanh -> Linear -> Softmax
    """

    def __init__(self, input_dim, hidden_dim=None):
        super(AttentionPooling, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Time, Channels)
        Returns:
            out: Weighted sum tensor of shape (Batch, Channels)
        """
        # Calculate attention weights: (Batch, Time, 1)
        weights = self.attention_net(x)

        # Apply weights to input and sum over time: (Batch, Channels)
        # Broadcasting weights across channels
        out = torch.sum(x * weights, dim=1)
        return out


class AudioClassifier(nn.Module):
    """
    EfficientNet-B0 backbone with Attention Pooling.
    Modified for single-channel input and optimized for multi-label audio tagging.
    """

    def __init__(self):
        super(AudioClassifier, self).__init__()

        # 1. Backbone: EfficientNet-B0 (Pretrained)
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        # Modify first convolutional layer for 1-channel input Cite solution_lesson_node_00015
        # Sum weights across the 3 input channels to preserve pretrained features
        first_conv = self.backbone.features[0][0]
        w_sum = first_conv.weight.sum(dim=1, keepdim=True)
        first_conv.weight = nn.Parameter(w_sum)
        first_conv.in_channels = 1

        self.features = self.backbone.features

        # EfficientNet-B0 outputs 1280 channels at the final layer
        self.feature_dim = 1280

        # 2. Pooling Head: Attention Pooling Cite solution_lesson_node_00007
        # First, pool frequency dimension to 1, keeping time
        self.avg_pool_freq = nn.AdaptiveAvgPool2d((1, None))
        self.att_pooling = AttentionPooling(self.feature_dim)

        # 3. Classifier
        self.classifier = nn.Linear(self.feature_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, 1, Freq, Time)
        Returns:
            logits: Output logits of shape (Batch, Num_Classes)
        """
        # Extract features: (Batch, 1280, F', T')
        x = self.features(x)

        # Collapse Frequency: (Batch, 1280, 1, T')
        x_seq = self.avg_pool_freq(x)
        # Squeeze to (Batch, 1280, T')
        x_seq = x_seq.squeeze(2)
        # Permute to (Batch, T', 1280) for Linear layers
        x_seq = x_seq.permute(0, 2, 1)

        # Apply Attention Pooling
        out = self.att_pooling(x_seq)  # (Batch, 1280)

        # Classification
        logits = self.classifier(out)

        return logits
