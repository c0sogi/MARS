import torch
import torch.nn as nn
import timm
from library.config import ModelConfig


class MultiHeadAttentionPooling(nn.Module):
    """
    Aggregates temporal features using Multi-Head Attention.
    Computes 'num_heads' distinct attention vectors to capture different
    temporal aspects of the signal.
    """

    def __init__(self, in_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        # Project input to 'num_heads' attention scores
        self.attention_linear = nn.Linear(in_dim, num_heads)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) tensor.
        Returns:
            context: (B, num_heads * D) tensor.
        """
        # Compute attention scores: (B, T, num_heads)
        scores = self.attention_linear(x)
        weights = torch.softmax(scores, dim=1)

        # Weighted sum for each head:
        # Transpose weights to (B, num_heads, T)
        weights = weights.transpose(1, 2)

        # (B, num_heads, T) @ (B, T, D) -> (B, num_heads, D)
        context = torch.bmm(weights, x)

        # Flatten heads: (B, num_heads * D)
        context = context.view(context.size(0), -1)
        return context


class ResNeStCRNN(nn.Module):
    """
    GPU-Accelerated Multi-Resolution ResNeSt-CRNN.

    Architecture:
    1. Input: 3-Channel Multi-Resolution Log-Mel Spectrograms.
    2. Backbone: ResNeSt50d (Split-Attention) with modified strides in deep layers.
    3. Neck: Frequency Pooling + Bidirectional GRU.
    4. Head: Multi-Head Attention Pooling + Linear Classifier.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 1. Backbone
        # Load ResNeSt50d, pretrained, with 3 input channels.
        # global_pool='' ensures we get the spatial feature map (B, C, F, T)
        # instead of a pooled vector.
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            in_chans=config.in_channels,
            global_pool="",
            num_classes=0,  # Remove default classification head
        )

        # 2. Modify Strides (Layer 3 and 4)
        # We set strides to 1 in the deeper layers to prevent excessive downsampling
        # in the time dimension, preserving temporal resolution for the RNN.
        if hasattr(self.backbone, "layer3"):
            self._remove_stride(self.backbone.layer3)
        if hasattr(self.backbone, "layer4"):
            self._remove_stride(self.backbone.layer4)

        # 3. Calculate Feature Dimension
        # Run a dummy forward pass to determine the channel dimension of the backbone output
        with torch.no_grad():
            # Input: (2, 3, 128, 100) - Batch size 2 to avoid BatchNorm error in train mode
            dummy_input = torch.zeros(2, config.in_channels, 128, 100)
            dummy_out = self.backbone(dummy_input)
            # dummy_out: (2, C, F', T')
            self.feature_dim = dummy_out.shape[1]

        # 4. Neck: Bidirectional GRU
        # Processes the sequence of features extracted by the CNN
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        rnn_out_dim = config.hidden_size * 2

        # 5. Head: Multi-Head Attention Pooling
        # Aggregates the RNN hidden states into a single vector
        self.attention = MultiHeadAttentionPooling(
            rnn_out_dim, num_heads=config.num_heads
        )

        # 6. Classifier
        self.classifier = nn.Linear(rnn_out_dim * config.num_heads, config.num_classes)

    def _remove_stride(self, module):
        """
        Recursively sets stride to (1, 1) for all Conv2d and AvgPool2d layers
        within the module that have stride > 1. This targets the downsampling
        layers (including 'avd' layers in ResNeSt) to maintain resolution.
        """
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.AvgPool2d)):
                if m.stride == (2, 2) or m.stride == 2:
                    m.stride = (1, 1)

    def forward(self, x):
        """
        Args:
            x: (B, 3, F, T) Multi-Resolution Spectrograms.
        Returns:
            logits: (B, num_classes).
        """
        # Backbone Feature Extraction
        # Output: (B, C, F', T')
        x = self.backbone(x)

        # Global Frequency Pooling
        # Average over the frequency dimension (dim 2) to collapse spectral features
        # while preserving temporal features.
        # Output: (B, C, T')
        x = x.mean(dim=2)

        # Permute for RNN
        # (B, C, T') -> (B, T', C)
        x = x.permute(0, 2, 1)

        # BiGRU
        # Output: (B, T', 2*hidden_size)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        # Multi-Head Attention Pooling
        # Aggregates temporal steps based on learned importance
        # Output: (B, 2*hidden_size*num_heads)
        x = self.attention(x)

        # Classifier
        # Output: (B, num_classes)
        x = self.classifier(x)

        return x
