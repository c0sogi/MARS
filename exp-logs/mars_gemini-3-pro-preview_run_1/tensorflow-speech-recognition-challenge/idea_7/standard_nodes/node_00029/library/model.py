import torch
import torch.nn as nn
import timm
from library.config import model_config


class AttentivePooling(nn.Module):
    """
    Attentive Pooling layer that dynamically weights temporal frames.
    Formula: y = sum(x_t * softmax(w^T * tanh(W * x_t)))
    """

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # Input x shape: (Batch, Channels, Time)

        # Transpose to (Batch, Time, Channels) for Linear layers
        x = x.transpose(1, 2)

        # Calculate attention weights
        # weights shape: (Batch, Time, 1)
        weights = self.attention(x)

        # Weighted sum over time
        # (Batch, Time, Channels) * (Batch, Time, 1) -> Sum over Time -> (Batch, Channels)
        out = torch.sum(x * weights, dim=1)
        return out


class DilatedEfficientNet(nn.Module):
    """
    EfficientNet-B2 with 1-channel input and dilated convolutions in the final stage
    to preserve temporal resolution for audio tasks.
    """

    def __init__(self, num_classes):
        super().__init__()

        # 1. Create Backbone
        # We use num_classes=0 to remove the classification head and pooling
        self.backbone = timm.create_model(
            model_config.backbone,
            pretrained=model_config.pretrained,
            num_classes=0,
            global_pool="",
            drop_rate=model_config.drop_rate,
            drop_path_rate=model_config.drop_path_rate,
        )

        # 2. Adapt First Layer (3 channels -> 1 channel)
        self._adapt_first_layer()

        # 3. Apply Dilation to Final Stage
        self._apply_dilation_to_final_stage()

        # 4. Determine feature dimension dynamically
        # Run a dummy forward pass to get the output shape
        with torch.no_grad():
            # Dummy input: (Batch, Channel, Freq, Time)
            # 128 Mels, ~100 frames (approx 1 sec audio)
            dummy = torch.randn(1, 1, 128, 100)
            features = self.backbone(dummy)
            # Features shape: (B, C, F, T)
            self.feature_dim = features.shape[1]

        # 5. Head
        self.pool = AttentivePooling(self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _adapt_first_layer(self):
        """
        Modifies the first convolution layer to accept 1-channel input
        by averaging the weights of the original 3 channels.
        """
        if not hasattr(self.backbone, "conv_stem"):
            return

        old_conv = self.backbone.conv_stem
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Average weights: (Out, 3, K, K) -> (Out, 1, K, K)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

        self.backbone.conv_stem = new_conv

    def _apply_dilation_to_final_stage(self):
        """
        Modifies the final stage of the backbone to use dilated convolutions
        (dilation=2, stride=1) to preserve spatial resolution.
        """
        if not hasattr(self.backbone, "blocks"):
            return

        # The last stage is the last element in the blocks list
        last_stage = self.backbone.blocks[-1]

        for module in last_stage.modules():
            if isinstance(module, nn.Conv2d):
                # 1. Reset Stride: If stride is 2 (downsampling), set to 1
                if module.stride == (2, 2) or module.stride == 2:
                    module.stride = (1, 1)

                # 2. Apply Dilation: If it's a spatial conv (kernel > 1)
                if module.kernel_size[0] > 1 or module.kernel_size[1] > 1:
                    # Set dilation to 2
                    new_dilation = (2, 2)
                    module.dilation = new_dilation

                    # 3. Adjust Padding to maintain 'same' padding behavior
                    # Padding = (dilation * (kernel_size - 1)) / 2
                    k = module.kernel_size[0]
                    p = (new_dilation[0] * (k - 1)) // 2
                    module.padding = (p, p)

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)
        x = self.backbone(x)  # Output: (Batch, Channels, Freq', Time')

        # Global Average Pooling over Frequency axis
        # We want to keep Time axis for Attentive Pooling
        x = x.mean(dim=2)  # Output: (Batch, Channels, Time')

        # Attentive Pooling over Time axis
        x = self.pool(x)  # Output: (Batch, Channels)

        # Classification
        logits = self.classifier(x)  # Output: (Batch, Num_Classes)

        return logits
