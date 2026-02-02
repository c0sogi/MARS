import torch
import torch.nn as nn
import timm
from library.config import LABEL_CONFIG


class AttentivePooling(nn.Module):
    """
    Applies attention over the time dimension to weight active speech segments.
    Input: (Batch, Channels, Time)
    Output: (Batch, Channels)
    """

    def __init__(self, in_channels):
        super().__init__()
        # 1x1 Conv to calculate attention score for each time step
        self.att_conv = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)

        # Calculate attention scores
        # (Batch, C, T) -> (Batch, 1, T)
        attn = self.att_conv(x)
        attn = self.softmax(attn)

        # Weighted sum over time
        # (Batch, C, T) * (Batch, 1, T) -> (Batch, C, T) -> sum -> (Batch, C)
        x = torch.sum(x * attn, dim=-1)
        return x


class MultiSampleDropout(nn.Module):
    """
    Applies multiple dropout masks to the same features and averages the predictions.
    Acts as an implicit ensemble within a single model.
    """

    def __init__(self, in_features, out_features, dropout_rate=0.5, num_samples=8):
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (Batch, In_Features)
        logits = []
        for dropout in self.dropouts:
            # Apply dropout
            out = dropout(x)
            # Apply shared linear layer
            out = self.linear(out)
            logits.append(out)

        # Stack logits: (Batch, Num_Samples, Out_Features)
        # Average over samples: (Batch, Out_Features)
        return torch.stack(logits, dim=1).mean(dim=1)


class DilatedEfficientNetB2(nn.Module):
    """
    EfficientNet-B2 with:
    1. 1-Channel Input (Averaged Weights)
    2. Dilated Convolutions in the final stage (Output Stride 16)
    3. Attentive Pooling
    4. Multi-Sample Dropout Head
    """

    def __init__(self, num_classes=LABEL_CONFIG.num_classes, pretrained=True):
        super().__init__()

        # Load backbone
        # output_stride=16 ensures the last stage uses dilation=2 and stride=1
        # features_only=True allows us to access intermediate feature maps
        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            features_only=True,
            output_stride=16,
        )

        # Modify first conv for 1-channel input
        self._modify_first_conv()

        # Determine feature dimension dynamically
        # We run a dummy pass to check the output shape of the backbone
        with torch.no_grad():
            # Dummy input: (Batch, Channels, Freq, Time)
            # Freq=128, Time=101 (approx 1 sec)
            dummy = torch.randn(1, 1, 128, 101)
            features = self.forward_features(dummy)
            # features shape: (Batch, Channels, Freq', Time')
            self.feature_dim = features.shape[1]

        # Pooling and Head
        self.pooling = AttentivePooling(self.feature_dim)
        self.head = MultiSampleDropout(
            in_features=self.feature_dim,
            out_features=num_classes,
            dropout_rate=0.5,
            num_samples=8,
        )

    def _modify_first_conv(self):
        """
        Replaces the first convolution layer (RGB) with a 1-channel layer.
        Weights are initialized by averaging the original RGB weights.
        """
        old_conv = self.backbone.conv_stem

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize weights
        # old_conv.weight: (Out, 3, K, K)
        # new_conv.weight: (Out, 1, K, K)
        with torch.no_grad():
            new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias

        self.backbone.conv_stem = new_conv

    def forward_features(self, x):
        # timm features_only returns a list of feature maps from different stages.
        # We take the last one.
        return self.backbone(x)[-1]

    def forward(self, x):
        """
        Args:
            x: Log-Mel Spectrogram (Batch, 1, Freq, Time)
        Returns:
            Logits (Batch, Num_Classes)
        """
        # 1. Backbone Feature Extraction
        # Output: (Batch, C, F', T')
        x = self.forward_features(x)

        # 2. Frequency Pooling
        # Collapse the frequency dimension by averaging, preserving Time.
        # Output: (Batch, C, T')
        x = torch.mean(x, dim=2)

        # 3. Attentive Pooling
        # Aggregate over Time dimension.
        # Output: (Batch, C)
        x = self.pooling(x)

        # 4. Classification Head
        # Output: (Batch, Num_Classes)
        x = self.head(x)

        return x
