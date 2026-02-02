import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import HIDDEN_DIM, NUM_CLASSES


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, time_steps, input_dim)
        # Calculate attention scores
        scores = self.attention(x)  # (batch_size, time_steps, 1)
        weights = torch.softmax(scores, dim=1)  # (batch_size, time_steps, 1)

        # Weighted sum of time steps
        # (batch_size, time_steps, input_dim) * (batch_size, time_steps, 1) -> sum over time
        context_vector = torch.sum(x * weights, dim=1)  # (batch_size, input_dim)
        return context_vector


class TimePreservingEfficientNet(nn.Module):
    def __init__(self):
        super(TimePreservingEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Modify First Convolution for 1-Channel Input
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        # Initialize with average of original weights to preserve pre-training info
        with torch.no_grad():
            new_conv.weight[:] = torch.mean(original_conv.weight, dim=1, keepdim=True)
        self.backbone.features[0][0] = new_conv

        # 3. Modify Strides for Time Preservation
        # We target the downsampling blocks in the deeper stages to set stride to (2, 1)
        # Indices in features: 3 (Stage 3), 4 (Stage 4), 6 (Stage 6)
        target_indices = [3, 4, 6]

        for idx in target_indices:
            # The first block in the stage performs the downsampling
            block = self.backbone.features[idx][0]
            self._modify_block_stride(block)

        # 4. Define Heads
        # EfficientNet-B0 output channels is 1280
        self.backbone_out_dim = 1280

        self.gru = nn.GRU(
            input_size=self.backbone_out_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.attn_pool = AttentionPooling(HIDDEN_DIM * 2)
        self.classifier = nn.Linear(HIDDEN_DIM * 2, NUM_CLASSES)

    def _modify_block_stride(self, mb_conv_block):
        # Access the sequential block within the MBConv module
        # Structure usually: Expand -> Depthwise -> SE -> Project
        # We look for the Depthwise Conv (groups > 1) with stride (2, 2)

        for i, layer in enumerate(mb_conv_block.block):
            # Each layer is Conv2dNormActivation, so layer[0] is the Conv2d
            if isinstance(layer[0], nn.Conv2d):
                conv = layer[0]
                # Check if it's the depthwise conv with downsampling
                if conv.stride == (2, 2) and conv.groups == conv.in_channels:
                    new_conv = nn.Conv2d(
                        in_channels=conv.in_channels,
                        out_channels=conv.out_channels,
                        kernel_size=conv.kernel_size,
                        stride=(2, 1),  # Modified stride: (Freq, Time)
                        padding=conv.padding,
                        groups=conv.groups,
                        bias=conv.bias is not None,
                    )
                    # Copy weights
                    with torch.no_grad():
                        new_conv.weight[:] = conv.weight
                        if conv.bias is not None:
                            new_conv.bias[:] = conv.bias

                    # Replace the conv in the Conv2dNormActivation module
                    layer[0] = new_conv
                    # We only need to modify the first matching depthwise conv
                    break

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)

        # Backbone Feature Extraction
        x = self.backbone.features(x)
        # Output shape: (Batch, 1280, F', T')

        # Frequency Pooling
        # We average over the remaining frequency bins to get a sequence of feature vectors
        x = torch.mean(x, dim=2)  # (Batch, 1280, T')

        # Permute for GRU (Batch, Time, Channels)
        x = x.permute(0, 2, 1)

        # Bi-Directional GRU
        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # (Batch, T', 2 * Hidden)

        # Attention Pooling
        x = self.attn_pool(x)  # (Batch, 2 * Hidden)

        # Classification
        logits = self.classifier(x)  # (Batch, 1)

        return logits
