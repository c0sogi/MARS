import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer that computes a weighted sum of the input sequence.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, Config.ATTENTION_DIM),
            nn.Tanh(),
            nn.Linear(Config.ATTENTION_DIM, 1),
        )

    def forward(self, x):
        # x shape: [Batch, Time, Input_Dim]

        # Compute attention scores
        # [Batch, Time, 1]
        attn_scores = self.attention(x)

        # Normalize scores to weights
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Compute weighted sum
        # [Batch, Time, Input_Dim] * [Batch, Time, 1] -> [Batch, Time, Input_Dim]
        weighted_x = x * attn_weights

        # Sum over time dimension
        # [Batch, Input_Dim]
        pooled = torch.sum(weighted_x, dim=1)

        return pooled


class TimePreservingEfficientNet(nn.Module):
    """
    EfficientNet-B0 backbone with modified strides to preserve temporal resolution,
    followed by BiGRU and Attention Pooling.
    """

    def __init__(self):
        super(TimePreservingEfficientNet, self).__init__()

        # 1. Load Backbone
        # Use features_only=False but remove classifier/pooling to access raw features via forward_features
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
        )

        # 2. Modify First Convolution (3 channels -> 1 channel)
        # EfficientNet uses 'conv_stem' as the first layer
        if hasattr(self.backbone, "conv_stem"):
            old_conv = self.backbone.conv_stem
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialize with average of original weights to preserve pretrained features
            with torch.no_grad():
                new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)
            self.backbone.conv_stem = new_conv

        # 3. Modify Strides for Time Preservation
        # Standard EfficientNet-B0 downsamples at blocks indices: 1, 2, 3, 5.
        # We want to maintain a temporal length of ~50 (4x downsampling total).
        # Stem provides 2x. Block 1 provides 2x. Total 4x.
        # We modify Block 2, 3, and 5 to have stride (2, 1) instead of (2, 2).
        target_indices = [2, 3, 5]

        for idx in target_indices:
            if idx < len(self.backbone.blocks):
                block = self.backbone.blocks[idx]
                # Recursively find Conv2d layers with stride (2, 2) and change to (2, 1)
                for m in block.modules():
                    if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                        m.stride = (2, 1)

        # 4. Determine Feature Dimension
        # Run a dummy forward pass to get the channel count of the final feature map
        dummy_input = torch.zeros(1, 1, Config.N_MELS, Config.WIN_LENGTH)
        with torch.no_grad():
            features = self.backbone.forward_features(dummy_input)
            # features shape: [Batch, Channels, Freq, Time]
            self.feature_dim = features.shape[1]

        # 5. Temporal Modeling (Bi-directional GRU)
        self.rnn = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
        )

        rnn_out_dim = (
            Config.RNN_HIDDEN_SIZE * 2
            if Config.BIDIRECTIONAL
            else Config.RNN_HIDDEN_SIZE
        )

        # 6. Aggregation (Attention Pooling)
        self.attn_pool = AttentionPooling(rnn_out_dim)

        # 7. Classifier
        self.fc = nn.Linear(rnn_out_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrogram [Batch, 1, Freq, Time]
        Returns:
            torch.Tensor: Logits [Batch, 1]
        """
        # Backbone Feature Extraction
        # Returns [Batch, Channels, Freq_down, Time_down]
        x = self.backbone.forward_features(x)

        # Frequency Pooling
        # Average over the frequency dimension to get a sequence of feature vectors
        # [Batch, Channels, Time_down]
        x = torch.mean(x, dim=2)

        # Permute for RNN
        # [Batch, Time_down, Channels]
        x = x.permute(0, 2, 1)

        # RNN Processing
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # [Batch, Time_down, RNN_Dim]

        # Attention Pooling
        # [Batch, RNN_Dim]
        x = self.attn_pool(x)

        # Classification
        # [Batch, 1]
        logits = self.fc(x)

        return logits
