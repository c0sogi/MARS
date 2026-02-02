import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer that learns to weight time steps differently.
    Useful for focusing on the speech command within a padded audio clip.
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: (Batch, Time, Features)

        # Calculate attention scores: (Batch, Time, 1)
        scores = self.attention(x)

        # Normalize scores across the time dimension
        weights = torch.softmax(scores, dim=1)

        # Weighted sum of features: (Batch, Features)
        output = torch.sum(x * weights, dim=1)

        return output


class SKResNetCRNN(nn.Module):
    """
    Multi-Resolution Selective Kernel CRNN.

    Architecture:
    1. Input: 3-Channel Multi-Resolution Spectrogram (RGB).
    2. Backbone: SK-ResNet34 (timm) with modified strides in Layer 3 & 4.
    3. Neck: Bidirectional GRU.
    4. Head: Attention Pooling + Linear Classifier.
    """

    def __init__(self):
        super(SKResNetCRNN, self).__init__()

        # 1. Backbone: SK-ResNet34
        # We use 'skresnet34' from timm.
        # in_chans=3 allows us to use the RGB Multi-Resolution input.
        try:
            self.backbone = timm.create_model("skresnet34", pretrained=True, in_chans=3)
        except Exception as e:
            print(f"Warning: Could not load skresnet34 ({e}). Fallback to resnet34.")
            self.backbone = timm.create_model("resnet34", pretrained=True, in_chans=3)

        # 2. Modify Strides for Temporal Preservation
        # Standard ResNet downsamples by 32x. We modify layer3 and layer4
        # to have stride 1, resulting in only 8x downsampling total.
        # This keeps the time dimension length ~12-13 frames for 1s audio.
        self._modify_stride(self.backbone.layer3)
        self._modify_stride(self.backbone.layer4)

        # Get feature dimension (usually 512 for ResNet34 variants)
        self.feature_dim = self.backbone.num_features

        # 3. Recurrent Layer (BiGRU)
        self.rnn = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=Config.GRU_HIDDEN_SIZE,
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.GRU_LAYERS > 1 else 0,
        )

        rnn_out_dim = Config.GRU_HIDDEN_SIZE * 2  # Bidirectional

        # 4. Attention Pooling
        self.attention_pooling = AttentionPooling(rnn_out_dim)

        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT), nn.Linear(rnn_out_dim, Config.NUM_CLASSES)
        )

    def _modify_stride(self, layer):
        """
        Recursively finds Conv2d layers with stride (2,2) and sets them to (1,1).
        This effectively removes downsampling from the specified layer.
        """
        for module in layer.modules():
            # Modify Convolutions
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)

            # Modify Pooling layers (if any exist in the block structure)
            if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                if module.stride == 2 or module.stride == (2, 2):
                    module.stride = 1

    def forward(self, x):
        # Input: (Batch, 3, Freq, Time) -> e.g., (B, 3, 64, 101)

        # Pass through Backbone
        # forward_features returns (B, C, F', T')
        x = self.backbone.forward_features(x)

        # Frequency Pooling: Average over frequency dimension
        # (B, 512, F', T') -> (B, 512, T')
        x = torch.mean(x, dim=2)

        # Permute for RNN: (B, T', 512)
        x = x.permute(0, 2, 1)

        # Pass through BiGRU
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # Output: (B, T', 2*Hidden)

        # Attention Pooling
        # Aggregates time steps into a single vector
        x = self.attention_pooling(x)  # (B, 2*Hidden)

        # Classification
        x = self.classifier(x)  # (B, NumClasses)

        return x


def get_model():
    """Factory function to return the model instance."""
    return SKResNetCRNN()
