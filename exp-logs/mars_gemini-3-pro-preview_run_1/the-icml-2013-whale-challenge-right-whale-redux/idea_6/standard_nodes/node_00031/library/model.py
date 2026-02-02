import torch
import torch.nn as nn
import timm
import math
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Computes a weighted average of the input sequence based on learned attention scores.
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

        # Calculate attention scores
        # scores shape: (Batch, Time, 1)
        scores = self.attention(x)

        # Normalize scores across the time dimension
        weights = torch.softmax(scores, dim=1)

        # Weighted sum
        # Output shape: (Batch, Features)
        output = torch.sum(x * weights, dim=1)

        return output


class TimePreservingEfficientNetBiGRU(nn.Module):
    """
    CRNN with EfficientNet-B0 backbone and Bi-directional GRU head.
    Modifies backbone strides to preserve temporal resolution.
    """

    def __init__(self):
        super(TimePreservingEfficientNetBiGRU, self).__init__()

        # 1. Load Pretrained Backbone
        # features_only=False gives us the full model structure which is easier to modify in-place
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=Config.PRETRAINED, features_only=False
        )

        # 2. Modify Input Layer (3 Channels -> 1 Channel)
        original_stem = self.backbone.conv_stem
        self.backbone.conv_stem = nn.Conv2d(
            in_channels=1,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=False,
        )
        # Initialize with averaged weights from RGB
        with torch.no_grad():
            self.backbone.conv_stem.weight.data = original_stem.weight.data.mean(
                dim=1, keepdim=True
            )

        # 3. Modify Strides for Time Preservation
        # Goal: Limit temporal downsampling to 4x (Stem /2 + One Block /2).
        # Subsequent downsampling blocks should be (2, 1) [Freq, Time].

        downsample_count = 0

        # Check Stem stride
        if self.backbone.conv_stem.stride[0] == 2:
            downsample_count += 1

        # Iterate through stages and blocks
        # timm EfficientNet stores blocks in model.blocks which is a Sequential of Sequentials (Stages)
        for stage in self.backbone.blocks:
            for block in stage:
                # Check if block is a downsampling block
                # In timm, blocks usually have a 'stride' attribute
                is_downsample = False
                if hasattr(block, "stride"):
                    if block.stride == (2, 2) or block.stride == 2:
                        is_downsample = True

                if is_downsample:
                    downsample_count += 1
                    # If we have already downsampled time by 4x (2 steps of /2),
                    # change this and future strides to (2, 1)
                    if downsample_count > 2:
                        # Set block stride attribute
                        block.stride = (2, 1)

                        # Recursively update Conv2d layers inside the block
                        for m in block.modules():
                            if isinstance(m, nn.Conv2d):
                                if m.stride == (2, 2) or m.stride == 2:
                                    m.stride = (2, 1)

        # 4. Determine Backbone Output Size
        # Run a dummy forward pass to get the channel count
        # Input: (Batch, 1, F, T) -> Use typical spectrogram size
        dummy_input = torch.randn(1, 1, Config.N_MELS, 256)
        with torch.no_grad():
            features = self.backbone.forward_features(dummy_input)

        backbone_out_channels = features.shape[1]

        # 5. RNN Head
        self.gru = nn.GRU(
            input_size=backbone_out_channels,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.RNN_DROPOUT if Config.RNN_LAYERS > 1 else 0,
        )

        # 6. Aggregation and Classification
        self.attn_pooling = AttentionPooling(Config.RNN_HIDDEN_SIZE * 2)
        self.fc = nn.Linear(Config.RNN_HIDDEN_SIZE * 2, 1)

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)

        # Extract features using modified backbone
        # Output: (Batch, Channels, F_out, T_out)
        x = self.backbone.forward_features(x)

        # Frequency Pooling: Average over the frequency dimension
        # We want to keep the Time dimension for the RNN
        # Output: (Batch, Channels, T_out)
        x = torch.mean(x, dim=2)

        # Permute for RNN: (Batch, Time, Channels)
        x = x.permute(0, 2, 1)

        # RNN Processing
        # Output: (Batch, Time, Hidden*2)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        # Attention Pooling
        # Output: (Batch, Hidden*2)
        x = self.attn_pooling(x)

        # Classification
        # Output: (Batch, 1)
        x = self.fc(x)

        return x
