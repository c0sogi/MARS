import torch
import torch.nn as nn
import timm
import library.config as config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    This model adapts a standard 2D EfficientNet to process volumetric multi-modal MRI data.
    It uses Grouped Convolutions in the stem to process 4 modalities (FLAIR, T1w, T1wCE, T2w)
    independently in the first layer, while initializing weights from ImageNet to preserve
    feature detection capabilities.

    Args:
        model_name (str): Name of the timm model (default: 'efficientnet_b0').
        pretrained (bool): Whether to load ImageNet weights.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Backbone
        # num_classes=0 removes the default FC layer and returns pooled features
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # 2. Modify Stem for 12-channel input
        # Original Stem: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        old_stem = self.backbone.conv_stem

        # Retrieve original parameters
        out_channels = old_stem.out_channels
        kernel_size = old_stem.kernel_size
        stride = old_stem.stride
        padding = old_stem.padding

        # Create New Stem
        # We use groups=4 to isolate the 4 modalities (FLAIR, T1w, T1wCE, T2w).
        # Input: 12 channels. Output: 32 channels.
        # This implies each group processes 3 input channels (12/4) and produces 8 output channels.
        # Since the input per group is 3, it matches the pre-trained filter depth (3).
        self.new_stem = nn.Conv2d(
            in_channels=config.TOTAL_CHANNELS,  # 12
            out_channels=out_channels,  # 32
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=4,
        )

        # 3. Asymmetric Weight Initialization
        if pretrained:
            self._init_stem_weights(old_stem.weight, self.new_stem.weight)

        # Replace the stem in the backbone
        self.backbone.conv_stem = self.new_stem

        # 4. Custom Classification Head
        # EfficientNet-B0 outputs 1280 features
        self.num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE), nn.Linear(self.num_features, 1)
        )

    def _init_stem_weights(self, old_weights, new_weights):
        """
        Adapts ImageNet weights (3 channels) to Volumetric weights (3 channels per group).

        Since we use 3 slices per modality and groups=4, the input depth per group is 3.
        This matches the geometry of the pre-trained filters (3 channels).
        We can directly copy the weights without scaling or heuristic mapping.
        (Cite solution_lesson_node_00023, Cite solution_lesson_node_00068)
        """
        # old_weights shape: (32, 3, 3, 3) -> (Out, In/Groups, K, K)
        # new_weights shape: (32, 3, 3, 3) -> (Out, In/Groups, K, K)

        with torch.no_grad():
            # Direct copy of diverse pre-trained filters
            new_weights.copy_(old_weights)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 20, H, W).
                              Channels are ordered: [FLAIR_0..4, T1w_0..4, T1wCE_0..4, T2w_0..4]
        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Extract features using backbone
        features = self.backbone(x)

        # Predict logits
        logits = self.head(features)

        return logits
