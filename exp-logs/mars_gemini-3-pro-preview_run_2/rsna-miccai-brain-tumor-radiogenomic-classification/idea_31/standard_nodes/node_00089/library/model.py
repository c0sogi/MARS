import torch
import torch.nn as nn
import timm
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    EfficientNet-B0 with a surgically modified stem for multi-modal MRI analysis.

    Implements:
    1. Grouped Convolutional Stem (groups=4) to isolate modalities.
    2. Asymmetric Filter Initialization to distribute ImageNet weights.
    3. Regularized Classification Head.
    """

    def __init__(self, model_name=None, pretrained=True, num_classes=1):
        super(AsymmetricEfficientNet, self).__init__()

        # Use Config defaults if arguments are not provided
        self.model_name = model_name if model_name else Config.MODEL_NAME

        # 1. Load Pre-trained Backbone
        # We load the model with num_classes=1, but we will rebuild the head anyway
        # to ensure the dropout structure matches our requirements.
        self.backbone = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=num_classes
        )

        # 2. Surgically Replace Stem
        self._replace_stem()

        # 3. Modify Classification Head
        self._replace_head(num_classes)

    def _replace_stem(self):
        """
        Replaces the first convolutional layer with a Grouped Convolution
        and applies Asymmetric Filter Initialization.
        """
        # Retrieve the original stem (standard RGB convolution)
        # In timm's EfficientNet, this is 'conv_stem'
        old_stem = self.backbone.conv_stem

        # Extract geometry from the original layer
        out_channels = old_stem.out_channels
        kernel_size = old_stem.kernel_size
        stride = old_stem.stride
        padding = old_stem.padding
        bias = old_stem.bias

        # Create the new Grouped Convolutional Stem
        # Input: 12 channels (Config.IN_CHANNELS)
        # Groups: 4 (Config.STEM_GROUPS) -> Splits input into 4 groups of 3 channels
        new_stem = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=Config.STEM_GROUPS,
            bias=bias is not None,
        )

        # Initialize weights
        self._init_asymmetric_weights(old_stem, new_stem)

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_stem

    def _init_asymmetric_weights(self, old_layer, new_layer):
        """
        Distributes the full bank of pre-trained ImageNet filters across the
        modality groups.

        PyTorch Grouped Conv Weight Shape: (Out, In/Groups, K, K)
        Original Conv Weight Shape:        (Out, In, K, K)

        For EfficientNet-B0:
        Original: (32, 3, 3, 3)
        New:      (32, 12//4, 3, 3) -> (32, 3, 3, 3)

        Since shapes are identical, direct copying maps specific filters to
        specific groups (modalities).
        """
        if old_layer.weight.shape == new_layer.weight.shape:
            new_layer.weight.data = old_layer.weight.data.clone()
            if old_layer.bias is not None and new_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.data.clone()
        else:
            # Fallback for safety, though architecture guarantees match
            nn.init.kaiming_normal_(
                new_layer.weight, mode="fan_out", nonlinearity="relu"
            )

    def _replace_head(self, num_classes):
        """
        Replaces the final classifier with a Dropout + Linear block.
        """
        # In timm EfficientNet, 'classifier' is the final Linear layer
        original_classifier = self.backbone.classifier
        in_features = original_classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROP_RATE), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
