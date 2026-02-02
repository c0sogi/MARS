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


class ContextAwareHead(nn.Module):
    """
    Custom Head for Spectro-Temporal Features.
    Performs Frequency Pooling -> 1D Context Conv -> Attention Pooling -> Classification.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # 1D Context Layer (Kernel size 5) to model local temporal dependencies
        self.context_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Attention Pooling to aggregate the sequence
        self.attn_pool = AttentionPooling(in_channels)

        # Regularization
        self.dropout = nn.Dropout(p=Config.DROPOUT)

        # Final Classifier
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # Input x: (Batch, C, F, T) from EfficientNet backbone

        # 1. Frequency-wise Pooling: Collapse F dimension
        # We average over frequency bins, keeping Time dimension intact.
        # x becomes (Batch, C, T)
        x = x.mean(dim=2)

        # 2. 1D Context Layer
        # Models transitions between frames (e.g., phonemes)
        x = self.context_conv(x)

        # 3. Attention Pooling
        # Collapses T dimension -> (Batch, C)
        x = self.attn_pool(x)

        # 4. Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


class TimeResolvedEfficientNet(nn.Module):
    """
    EfficientNet-B0 modified for Audio Spectrograms.
    - 1-Channel Input
    - Asymmetric Strides (2, 1) in deeper layers to preserve Time resolution
    - Context-Aware Attention Head
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

        # 2. Modify Strides for Time Resolution
        self._modify_strides()

        # Get the number of output channels from the backbone
        # EfficientNet-B0 final conv (stage 8) outputs 1280 channels
        backbone_out_channels = 1280

        # 3. Define Custom Head
        self.head = ContextAwareHead(backbone_out_channels, Config.NUM_CLASSES)

    def _modify_input_layer(self):
        """
        Adapts the first convolutional layer to accept 1-channel input (spectrogram)
        instead of 3-channel input (RGB), initializing by summing original weights.
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

    def _modify_strides(self):
        """
        Modifies the strides of the final two downsampling stages to be (2, 1).
        This preserves temporal resolution while downsampling frequency.
        """
        # Identify blocks that perform downsampling (stride=2)
        downsampling_indices = []

        for i, module in enumerate(self.backbone.features):
            # In torchvision, MBConv blocks have a 'stride' attribute
            if hasattr(module, "stride"):
                s = module.stride
                # Check for stride 2 (can be int 2 or tuple (2, 2))
                if s == 2 or s == (2, 2):
                    downsampling_indices.append(i)

        # We want to modify the LAST TWO downsampling blocks to avoid over-compressing time
        if len(downsampling_indices) >= 2:
            target_indices = downsampling_indices[-2:]

            for idx in target_indices:
                module = self.backbone.features[idx]

                # 1. Update the module's stride attribute (for forward logic)
                module.stride = (2, 1)

                # 2. Update the actual Conv2d layers inside the module
                # Recursively find Conv2d layers with stride 2 and change them
                for layer in module.modules():
                    if isinstance(layer, nn.Conv2d):
                        if layer.stride == (2, 2) or layer.stride == 2:
                            layer.stride = (2, 1)

    def forward(self, x):
        # Input x: (Batch, 1, F, T)

        # Pass through modified backbone
        # Output: (Batch, 1280, F_pooled, T_preserved)
        x = self.backbone.features(x)

        # Pass through context-aware head
        x = self.head(x)

        return x
