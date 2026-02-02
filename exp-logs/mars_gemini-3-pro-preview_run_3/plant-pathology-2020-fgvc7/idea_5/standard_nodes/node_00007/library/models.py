import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiLevelEfficientNet(nn.Module):
    """
    EfficientNet-B4 with Multi-Level Feature Aggregation.

    This model extracts feature maps from the last three reduction stages (indices 2, 3, 4
    corresponding to strides 8, 16, 32 in standard EfficientNet), applies Global Average Pooling
    to each, concatenates the resulting vectors, and feeds the fused representation into a
    final fully connected layer.
    """

    def __init__(
        self,
        model_name=Config.MODEL_EFFNET,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        super(MultiLevelEfficientNet, self).__init__()

        # Initialize backbone with features_only=True to access intermediate layers
        # out_indices=(2, 3, 4) selects the last three stages (stride 8, 16, 32)
        self.backbone = timm.create_model(
            model_name, features_only=True, pretrained=pretrained, out_indices=(2, 3, 4)
        )

        # Dynamically calculate the total number of channels for the linear layer
        # We sum the channels from all extracted feature maps
        try:
            # timm's feature_info.channels() returns the channel counts for the selected out_indices
            channels_list = self.backbone.feature_info.channels()
        except AttributeError:
            # Fallback: Perform a dummy forward pass to determine output shapes if feature_info is missing
            dummy_input = torch.randn(
                1, 3, Config.IMG_SIZE_EFFNET, Config.IMG_SIZE_EFFNET
            )
            with torch.no_grad():
                features = self.backbone(dummy_input)
            channels_list = [f.shape[1] for f in features]

        total_channels = sum(channels_list)

        # Pooling and Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(total_channels, num_classes)

    def forward(self, x):
        # Forward pass through backbone returns a list of feature maps
        features = self.backbone(x)

        pooled_features = []
        for f in features:
            # Apply Global Average Pooling: [B, C, H, W] -> [B, C, 1, 1]
            x_pool = self.avgpool(f)
            # Flatten: [B, C, 1, 1] -> [B, C]
            x_flat = x_pool.flatten(1)
            pooled_features.append(x_flat)

        # Concatenate features from all levels: [B, C_total]
        concat_features = torch.cat(pooled_features, dim=1)

        # Final classification
        out = self.fc(concat_features)
        return out


class SwinTransformerModel(nn.Module):
    """
    Swin Transformer (Tiny) Wrapper.

    Utilizes the Shifted Window Attention mechanism. This class wraps the standard
    timm implementation for consistency.
    """

    def __init__(
        self,
        model_name=Config.MODEL_SWIN,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        super(SwinTransformerModel, self).__init__()

        # Initialize standard Swin Transformer for classification
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
