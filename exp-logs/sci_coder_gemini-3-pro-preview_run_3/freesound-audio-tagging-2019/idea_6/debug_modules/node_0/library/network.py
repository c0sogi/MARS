import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.configuration import Config


class AttentionPooling(nn.Module):
    """
    Applies a learned attention mechanism over the spatial/temporal dimensions
    of the feature map.
    Structure: Linear -> Tanh -> Linear -> Softmax
    """

    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x shape: (N, C, H, W)
        N, C, H, W = x.shape

        # Flatten spatial dimensions: (N, C, L) where L = H * W
        x = x.view(N, C, -1)

        # Permute for Linear layer: (N, L, C)
        x_perm = x.permute(0, 2, 1)

        # Calculate attention weights: (N, L, 1)
        weights = self.attention(x_perm)

        # Weighted sum: (N, L, C) * (N, L, 1) -> (N, L, C) -> Sum over L -> (N, C)
        weighted = torch.sum(x_perm * weights, dim=1)

        return weighted


class DualStreamPooling(nn.Module):
    """
    Combines Global Max Pooling and Attention Pooling.
    Outputs a concatenated vector of size 2 * in_dim.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.attn_pool = AttentionPooling(in_dim)

    def forward(self, x):
        # x shape: (N, C, H, W)

        # Stream 1: Global Max Pooling -> (N, C)
        x_max = self.max_pool(x).view(x.size(0), -1)

        # Stream 2: Attention Pooling -> (N, C)
        x_attn = self.attn_pool(x)

        # Concatenate -> (N, 2*C)
        return torch.cat([x_max, x_attn], dim=1)


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout to accelerate convergence and improve generalization.
    Applies multiple dropout masks and averages the predictions from a shared linear layer.
    """

    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (N, in_features)
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout and pass through shared linear layer
            logits_list.append(self.fc(dropout(x)))

        # Stack and average logits: (num_samples, N, out_features) -> (N, out_features)
        return torch.stack(logits_list).mean(dim=0)


class ConvNeXtAudio(nn.Module):
    """
    Main Audio Tagging Model.
    Backbone: ConvNeXt-Nano (Modified for 1-channel input)
    Pooling: Dual-Stream (Max + Attention)
    Head: Multi-Sample Dropout
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # 1. Load Pretrained Backbone
        # We load with in_chans=3 initially to get pretrained weights, then patch it.
        # num_classes=0 removes the default classification head.
        self.backbone = timm.create_model(
            config.BACKBONE, pretrained=config.PRETRAINED, num_classes=0, in_chans=3
        )

        # 2. Modify Input Layer for Single Channel
        self._modify_first_layer()

        # Get feature dimension (typically 640 for convnext_nano)
        self.num_features = self.backbone.num_features

        # 3. Setup Pooling
        if config.POOLING_TYPE == "dual_stream":
            self.pooling = DualStreamPooling(self.num_features)
            self.head_in_features = self.num_features * 2
        elif config.POOLING_TYPE == "avg":
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.head_in_features = self.num_features
        elif config.POOLING_TYPE == "max":
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
            self.head_in_features = self.num_features
        else:
            raise ValueError(f"Unknown pooling type: {config.POOLING_TYPE}")

        # 4. Setup Classification Head
        if config.USE_MULTI_SAMPLE_DROPOUT:
            self.head = MultiSampleDropout(
                in_features=self.head_in_features,
                out_features=config.NUM_CLASSES,
                num_samples=config.NUM_DROPOUT_SAMPLES,
                dropout_rate=config.DROPOUT_RATE,
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(self.head_in_features, config.NUM_CLASSES),
            )

    def _modify_first_layer(self):
        """
        Adapts the first convolutional layer to accept 1-channel input
        by summing the weights of the original 3-channel (RGB) kernels.
        """
        # In ConvNeXt, the stem contains the first convolution at index 0
        if hasattr(self.backbone, "stem"):
            old_conv = self.backbone.stem[0]

            # Create new Conv2d with in_channels=1
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            # Sum the weights across the channel dimension (dim 1)
            # Shape: (Out, 3, K, K) -> (Out, 1, K, K)
            with torch.no_grad():
                new_conv.weight[:] = torch.sum(old_conv.weight, dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias

            # Replace the layer
            self.backbone.stem[0] = new_conv

    def forward(self, x):
        # x shape: (N, 1, F, T)

        # Extract features from backbone
        # Output shape: (N, C, H, W)
        x = self.backbone.forward_features(x)

        # Apply Pooling
        # Output shape: (N, head_in_features)
        if self.config.POOLING_TYPE in ["avg", "max"]:
            x = self.pooling(x).flatten(1)
        else:
            x = self.pooling(x)

        # Classification Head
        # Output shape: (N, num_classes)
        x = self.head(x)

        return x
