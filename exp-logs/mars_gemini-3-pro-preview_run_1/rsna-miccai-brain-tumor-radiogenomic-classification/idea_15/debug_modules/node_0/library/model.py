import torch
import torch.nn as nn
import timm
from library.config import BACKBONE_NAME, DROPOUT_RATE, NUM_CLASSES, IN_CHANNELS


class WITSNetwork(nn.Module):
    """
    Weight-Inflated Thick-Slab Independent-Instance (WITS-II) Network.

    This model adapts a standard 3-channel EfficientNet-B0 to accept 9-channel inputs
    (3 modalities x 3 slices) by 'inflating' the pretrained weights of the first
    convolutional layer. This preserves the ImageNet priors while allowing the model
    to process volumetric slabs.
    """

    def __init__(self):
        super().__init__()

        # Load the backbone (EfficientNet-B0)
        # num_classes=0 returns the model with the classification head removed
        # (but keeps the global pooling, outputting a feature vector)
        self.backbone = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0)

        # Adapt the first layer to accept 9 channels instead of 3
        self._inflate_weights()

        # Define the classifier head
        # self.backbone.num_features gives the output dimension of the backbone (e.g., 1280 for B0)
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(self.backbone.num_features, NUM_CLASSES),
        )

    def _inflate_weights(self):
        """
        Replaces the first convolutional layer (conv_stem) with a 9-channel version.
        Initializes the new weights by distributing the energy of the original RGB weights
        across the corresponding modality slabs.
        """
        # Target the stem convolution layer
        old_layer = self.backbone.conv_stem

        # Get layer parameters
        out_channels = old_layer.out_channels
        kernel_size = old_layer.kernel_size
        stride = old_layer.stride
        padding = old_layer.padding
        bias = old_layer.bias is not None

        # Create the new layer with IN_CHANNELS (9)
        new_layer = nn.Conv2d(
            in_channels=IN_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # --- Weight Initialization Strategy ---
        # Original weights shape: (Out, 3, K, K)
        # New weights shape: (Out, 9, K, K)
        old_w = old_layer.weight.data
        new_w = torch.zeros_like(new_layer.weight.data)

        # 1. FLAIR (Channels 0, 1, 2) <- Red Channel (Index 0)
        # We take the Red kernel, divide by 3, and replicate it 3 times.
        w_red = old_w[:, 0:1, :, :] / 3.0
        new_w[:, 0:3, :, :] = w_red.repeat(1, 3, 1, 1)

        # 2. T1wCE (Channels 3, 4, 5) <- Green Channel (Index 1)
        w_green = old_w[:, 1:2, :, :] / 3.0
        new_w[:, 3:6, :, :] = w_green.repeat(1, 3, 1, 1)

        # 3. T2w (Channels 6, 7, 8) <- Blue Channel (Index 2)
        w_blue = old_w[:, 2:3, :, :] / 3.0
        new_w[:, 6:9, :, :] = w_blue.repeat(1, 3, 1, 1)

        # Assign weights to new layer
        new_layer.weight.data = new_w

        # Copy bias if it exists
        if bias:
            new_layer.bias.data = old_layer.bias.data

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_layer

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        # Extract features
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)

        return logits
