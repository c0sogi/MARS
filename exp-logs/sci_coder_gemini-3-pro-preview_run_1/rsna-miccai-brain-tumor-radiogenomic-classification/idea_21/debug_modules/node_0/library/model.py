import torch
import torch.nn as nn
import timm
from library.config import MODEL_NAME, DROPOUT_RATE, NUM_CHANNELS


class SFWIVModel(nn.Module):
    """
    Spatially-Fixed Weight-Inflated Volumetric (SF-WIV) Network.

    This model uses an EfficientNet-B0 backbone with a modified input layer
    to accept 9-channel volumetric slabs (3 modalities x 3 depths).

    The weights of the first layer are initialized using 'Energy-Preserving Weight Inflation'
    to adapt the pretrained RGB weights to the 9-channel input without destroying
    the learned features or altering initial activation magnitudes.
    """

    def __init__(self, model_name=MODEL_NAME, pretrained=True, num_classes=1):
        super(SFWIVModel, self).__init__()

        # 1. Load Pretrained Backbone
        # efficientnet_b0 is specified in config
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # 2. Energy-Preserving Weight Inflation
        # Identify the first convolutional layer (stem)
        # In timm efficientnet, this is usually 'conv_stem'
        old_conv = self.backbone.conv_stem
        out_channels = old_conv.out_channels
        kernel_size = old_conv.kernel_size
        stride = old_conv.stride
        padding = old_conv.padding
        bias = old_conv.bias is not None

        # Create new conv layer with 9 input channels
        new_conv = nn.Conv2d(
            in_channels=NUM_CHANNELS,  # 9
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Initialize weights
        # Original weights shape: (out, 3, k, k)
        # New weights shape: (out, 9, k, k)
        old_weights = old_conv.weight.data
        new_weights = torch.zeros_like(new_conv.weight.data)

        # Mapping logic based on Description:
        # Input Channels:
        # 0: FLAIR (Depth A) -> maps to RGB Red (idx 0)
        # 1: T1wCE (Depth A) -> maps to RGB Green (idx 1)
        # 2: T2w   (Depth A) -> maps to RGB Blue (idx 2)
        # 3: FLAIR (Depth B) -> maps to RGB Red (idx 0)
        # 4: T1wCE (Depth B) -> maps to RGB Green (idx 1)
        # 5: T2w   (Depth B) -> maps to RGB Blue (idx 2)
        # 6: FLAIR (Depth C) -> maps to RGB Red (idx 0)
        # 7: T1wCE (Depth C) -> maps to RGB Green (idx 1)
        # 8: T2w   (Depth C) -> maps to RGB Blue (idx 2)

        # We divide by 3 because each original channel is now split into 3 depth slices.
        # This preserves the expected magnitude of the convolution output (Energy Preserving).

        for i in range(3):  # For each depth (0, 1, 2)
            # Offset for the current depth group (0, 3, 6)
            offset = i * 3

            # Copy scaled weights
            # FLAIR -> Red
            new_weights[:, offset + 0, :, :] = old_weights[:, 0, :, :] / 3.0
            # T1wCE -> Green
            new_weights[:, offset + 1, :, :] = old_weights[:, 1, :, :] / 3.0
            # T2w -> Blue
            new_weights[:, offset + 2, :, :] = old_weights[:, 2, :, :] / 3.0

        new_conv.weight.data = new_weights

        # If bias exists, copy it (though efficientnet conv_stem usually has no bias due to BN)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

        # 3. Classifier Head
        # Get number of features from the backbone output
        # Since we used num_classes=0 and global_pool='', we need to pool ourselves or use the model's structure
        # Actually, let's use num_classes=0 with global_pool='avg' to get the pooled feature vector directly
        # Re-instantiate to ensure clean state for pooling
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.backbone.conv_stem = new_conv  # Re-apply the stem modification

        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE), nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 9, H, W)
        features = self.backbone(x)  # (Batch, num_features)
        logits = self.classifier(features)  # (Batch, 1)
        return logits
