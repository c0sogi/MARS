import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Learnable Attention Pooling to aggregate temporal features.
    Computes a weighted sum of time steps, allowing the model to focus on
    active speech segments and suppress silence.
    """

    def __init__(self, input_dim):
        super().__init__()
        # Projects input features to a scalar score per time step
        self.score_net = nn.Sequential(nn.Linear(input_dim, 1), nn.Tanh())

    def forward(self, x):
        # Input x: (Batch, Channels, Time)

        # Transpose for Linear layer: (Batch, Time, Channels)
        x_t = x.transpose(1, 2)

        # Calculate attention scores: (Batch, Time, 1)
        scores = self.score_net(x_t)

        # Softmax over time dimension to get probabilities
        weights = torch.softmax(scores, dim=1)

        # Weighted sum: (Batch, Time, Channels) * (Batch, Time, 1) -> Sum over Time
        # Result: (Batch, Channels)
        context = (x_t * weights).sum(dim=1)

        return context


class AudioEfficientNet(nn.Module):
    """
    Standard EfficientNet-B0 adapted for Audio Spectrograms.
    - 1-Channel Input (Summed weights)
    - Standard Strides (Preserves pretrained hierarchy)
    - Attention Pooling Head
    """

    def __init__(self):
        super().__init__()

        # Initialize EfficientNet-B0 with ImageNet weights
        weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        )
        self.backbone = models.efficientnet_b0(weights=weights)

        # 1. Modify Input Layer for 1 Channel
        self._modify_input_layer()

        # EfficientNet-B0 final conv (stage 8) outputs 1280 channels
        backbone_out_channels = 1280

        # 2. Define Head
        self.pool = AttentionPooling(backbone_out_channels)
        self.dropout = nn.Dropout(p=Config.DROPOUT)
        self.fc = nn.Linear(backbone_out_channels, Config.NUM_CLASSES)

    def _modify_input_layer(self):
        """
        Adapts the first convolutional layer to accept 1-channel input (spectrogram)
        instead of 3-channel input (RGB), initializing by summing original weights.
        Cite solution_lesson_node_00019
        """
        # The first layer in torchvision's EfficientNet is features[0][0]
        original_conv = self.backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Sum weights across the channel dimension: (Out, 3, K, K) -> (Out, 1, K, K)
        # This preserves the magnitude of activations from pre-training.
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)

        # Replace the layer
        self.backbone.features[0][0] = new_conv

    def forward(self, x):
        # Input x: (Batch, 1, F, T)

        # Pass through backbone
        # Output: (Batch, 1280, F_pooled, T_pooled)
        x = self.backbone.features(x)

        # Frequency Pooling: Average over Frequency, keep Time
        # x becomes (Batch, 1280, T_pooled)
        x = x.mean(dim=2)

        # Attention Pooling: Aggregate over Time
        # Cite solution_lesson_node_00011
        x = self.pool(x)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x
