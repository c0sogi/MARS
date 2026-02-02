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
    EfficientNet backbone with modified strides to preserve temporal resolution,
    followed by BiGRU and Attention Pooling.
    """

    def __init__(self):
        super(TimePreservingEfficientNet, self).__init__()

        # 1. Load Backbone
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
            features_only=False,
        )

        # 2. Modify First Convolution (3 channels -> 1 channel)
        if hasattr(self.backbone, "conv1"):
            old_conv = self.backbone.conv1
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialize with average of original weights
            with torch.no_grad():
                new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)
            self.backbone.conv1 = new_conv

        # 3. Modify MaxPool Stride (2, 2) -> (2, 1) to preserve time
        if hasattr(self.backbone, "maxpool"):
            self.backbone.maxpool.stride = (2, 1)

        # 4. Modify Layer Strides
        # Layers 2, 3, 4 usually have stride 2. Change to (2, 1) to preserve time.
        # This keeps time downsampling to just 2x (from conv1) or 4x (if maxpool was 2x time)
        # With MaxPool stride (2, 1), total time downsampling is 2x.
        for layer_name in ["layer2", "layer3", "layer4"]:
            if hasattr(self.backbone, layer_name):
                layer = getattr(self.backbone, layer_name)
                for m in layer.modules():
                    if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                        m.stride = (2, 1)

        # 5. Determine Feature Dimension
        dummy_input = torch.zeros(1, 1, Config.N_MELS, 100)
        with torch.no_grad():
            features = self.backbone.forward_features(dummy_input)
            self.feature_dim = features.shape[1]

        # 6. Temporal Modeling (Bi-directional GRU)
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

        # 7. Aggregation (Attention Pooling)
        self.attn_pool = AttentionPooling(rnn_out_dim)

        # 8. Classifier
        self.fc = nn.Linear(rnn_out_dim, Config.NUM_CLASSES)

    def forward(self, x):
        # Backbone Feature Extraction
        # Returns [Batch, Channels, Freq_down, Time_down]
        x = self.backbone.forward_features(x)

        # Frequency Pooling
        # [Batch, Channels, Time_down]
        x = torch.mean(x, dim=2)

        # Permute for RNN
        # [Batch, Time_down, Channels]
        x = x.permute(0, 2, 1)

        # RNN Processing
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)

        # Attention Pooling
        x = self.attn_pool(x)

        # Classification
        logits = self.fc(x)

        return logits
