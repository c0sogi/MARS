import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed(Config.seed)


class DualStreamPooling(nn.Module):
    """
    Dual-Stream Pooling Head.
    Stream A: Non-Linear Attention Pooling (Context)
    Stream B: Global Max Pooling (Salience)
    """

    def __init__(self, in_features, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        # Stream A: Attention Mechanism
        # Linear -> Tanh -> Linear -> Softmax
        self.attn = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x: (Batch, Channels, Freq, Time)
        B, C, Freq, T = x.shape

        # --- Stream A: Attention Pooling ---
        # Flatten spatial dimensions: (B, C, F*T) -> Permute to (B, N, C)
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        # Calculate attention scores
        # weights: (B, N, 1)
        weights = self.attn(x_flat)

        # Weighted sum of features
        # (B, N, C) * (B, N, 1) -> sum over N -> (B, C)
        context = torch.sum(x_flat * weights, dim=1)

        # --- Stream B: Global Max Pooling ---
        # Max activation across F and T
        # (B, C, F, T) -> (B, C)
        salience = F.adaptive_max_pool2d(x, (1, 1)).view(B, -1)

        # --- Fusion ---
        # Concatenate both streams: (B, 2*C)
        return torch.cat([context, salience], dim=1)


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout Head.
    Passes the embedding through multiple dropout masks and averages the logits.
    """

    def __init__(self, in_features, out_features, num_samples=5, drop_rate=0.5):
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(drop_rate) for _ in range(num_samples)]
        )
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x: (B, In_Features)
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout then classifier
            logits_list.append(self.linear(dropout(x)))

        # Stack and average logits: (Num_Samples, B, Out) -> (B, Out)
        return torch.stack(logits_list).mean(dim=0)


class ConvNeXtAudio(nn.Module):
    """
    Audio Tagger using ConvNeXt-Nano backbone with Dual-Stream Pooling
    and Multi-Sample Dropout.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # 1. Load Pretrained Backbone
        # num_classes=0 removes the default head, leaving us with the feature extractor structure
        print(f"Initializing {config.backbone} backbone...")
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,
            in_chans=3,  # Load RGB weights initially
        )

        # 2. Modify First Layer for Single-Channel Input
        # ConvNeXt stem is usually: Sequential(Conv2d, LayerNorm)
        # We access the Conv2d layer at index 0 of the stem
        if config.in_channels != 3:
            old_conv = self.backbone.stem[0]
            new_conv = nn.Conv2d(
                in_channels=config.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            # Sum the pretrained RGB weights to initialize the single channel
            # Shape: (Out, 3, K, K) -> sum(dim=1) -> (Out, 1, K, K)
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.sum(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias

            self.backbone.stem[0] = new_conv
            print("Modified first layer for single-channel input (summed weights).")

        # 3. Determine Feature Dimensions
        self.num_features = self.backbone.num_features

        # 4. Pooling Layer
        if config.pooling_type == "dual_stream":
            self.pooling = DualStreamPooling(self.num_features)
            self.head_in_features = self.num_features * 2
        elif config.pooling_type == "max":
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
            self.head_in_features = self.num_features
        elif config.pooling_type == "avg":
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.head_in_features = self.num_features
        else:
            raise ValueError(f"Unknown pooling type: {config.pooling_type}")

        # 5. Classification Head
        if config.use_multi_sample_dropout:
            self.head = MultiSampleDropout(
                in_features=self.head_in_features,
                out_features=config.num_classes,
                num_samples=config.multi_sample_dropout_count,
                drop_rate=config.drop_rate,
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(config.drop_rate),
                nn.Linear(self.head_in_features, config.num_classes),
            )

    def forward(self, x):
        # Input x: (Batch, 1, Freq, Time)

        # Backbone Feature Extraction
        # Use forward_features to get the spatial map: (B, C, H, W)
        # Note: timm's ConvNeXt forward_features returns (B, C, H, W)
        x = self.backbone.forward_features(x)

        # Pooling
        if isinstance(self.pooling, DualStreamPooling):
            # DualStream expects (B, C, F, T)
            x = self.pooling(x)
        else:
            # Standard pooling expects (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
            x = self.pooling(x).flatten(1)

        # Classification Head
        logits = self.head(x)

        return logits
