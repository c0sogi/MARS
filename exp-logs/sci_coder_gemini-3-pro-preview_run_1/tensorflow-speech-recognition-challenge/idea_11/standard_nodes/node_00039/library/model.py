import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentivePooling(nn.Module):
    """
    Attentive Pooling Layer.

    Pools the frequency dimension via averaging, then applies a learned attention
    mechanism over the time dimension to weight active speech segments.
    """

    def __init__(self, in_channels):
        super().__init__()
        # Project channels to a single attention score per time step
        self.attn_conv = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Channels, Freq, Time)
        Returns:
            Tensor of shape (Batch, Channels)
        """
        # 1. Global Average Pooling over Frequency dimension
        # Shape: (Batch, Channels, Freq, Time) -> (Batch, Channels, Time)
        x_time = torch.mean(x, dim=2)

        # 2. Calculate Attention Scores over Time
        # Shape: (Batch, 1, Time)
        attn_logits = self.attn_conv(x_time)
        attn_weights = self.softmax(attn_logits)

        # 3. Weighted Sum
        # (B, C, T) * (B, 1, T) -> (B, C, T) -> Sum over T -> (B, C)
        x_weighted = torch.sum(x_time * attn_weights, dim=-1)

        return x_weighted


class DilatedEfficientNet(nn.Module):
    """
    RGB-Temporal Dilated EfficientNet-B2.

    Uses a pretrained EfficientNet-B2 backbone with dilated convolutions in the
    final stages to preserve temporal resolution. Accepts 3-channel inputs
    (Mel, Delta, Delta-Delta) and uses Attentive Pooling for classification.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        super().__init__()

        # Load EfficientNet-B2
        # output_stride=16 replaces the final stride-2 downsampling with dilation=2,
        # preserving higher resolution in the feature maps (spatial/temporal).

        # Initialize with 3 channels to get original RGB weights
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            in_chans=3,
            output_stride=16,
            features_only=True,
        )

        # Adapt first layer for 1-channel input if necessary
        # Cite solution_lesson_node_00016: Average weights instead of random init/sum
        if Config.IN_CHANNELS == 1 and hasattr(self.backbone, "conv_stem"):
            old_layer = self.backbone.conv_stem
            new_layer = nn.Conv2d(
                1,
                old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )
            if pretrained:
                # Average the weights: (Out, 3, K, K) -> (Out, 1, K, K)
                new_layer.weight.data = old_layer.weight.data.mean(dim=1, keepdim=True)
                if old_layer.bias is not None:
                    new_layer.bias.data = old_layer.bias.data

            self.backbone.conv_stem = new_layer

        # Determine output channels dynamically
        # We perform a dummy forward pass to get the shape of the last feature map
        dummy_input = torch.zeros(1, Config.IN_CHANNELS, 128, 128)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            # features is a list of tensors; we want the last one
            last_feat = features[-1]
            out_channels = last_feat.shape[1]

        self.attentive_pooling = AttentivePooling(out_channels)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(out_channels, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 3, F, T)

        # Backbone forward pass
        # Returns list of feature maps
        features_list = self.backbone(x)

        # Take the last feature map: (Batch, C, F', T')
        x = features_list[-1]

        # Apply Attentive Pooling -> (Batch, C)
        x = self.attentive_pooling(x)

        # Classification -> (Batch, Num_Classes)
        logits = self.classifier(x)

        return logits


def get_model(device=Config.DEVICE):
    """
    Factory function to create and move the model to the configured device.
    """
    model = DilatedEfficientNet()
    model.to(device)
    return model
