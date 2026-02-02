import torch
import torch.nn as nn
import timm
from library.config import Config


class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism for Multi-Instance Learning.
    Aggregates a bag of instance feature vectors into a single representation
    using a learnable attention mechanism.

    Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning".
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(GatedAttention, self).__init__()

        # Attention V: Linear -> Tanh
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        # Attention U: Linear -> Sigmoid (Gating)
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())

        # Attention weights mapping
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Num_Instances, Input_Dim)

        Returns:
            torch.Tensor: Weighted sum of features (Batch, Input_Dim)
            torch.Tensor: Attention weights (Batch, Num_Instances, 1)
        """
        # x shape: (B, K, L)

        # Calculate attention scores
        v_out = self.attention_V(x)  # (B, K, M)
        u_out = self.attention_U(x)  # (B, K, M)

        # Element-wise product (Gating mechanism)
        gated = v_out * u_out

        # Calculate scalar score for each instance
        scores = self.attention_w(gated)  # (B, K, 1)

        # Normalize scores to probability distribution via Softmax over instances
        attn_weights = torch.softmax(scores, dim=1)

        # Weighted sum of instance features
        # (B, K, 1) * (B, K, L) -> (B, K, L) -> Sum over K -> (B, L)
        weighted_features = torch.sum(attn_weights * x, dim=1)

        return weighted_features, attn_weights


class MILEfficientNet(nn.Module):
    """
    Multi-Instance EfficientNet-B0 with Grouped Convolutional Stem and Attention Pooling.

    Architecture:
    1. Input: (Batch, K, 12, H, W)
    2. Backbone: EfficientNet-B0 (Shared weights across K instances)
       - Stem modified for 12 channels, groups=4.
    3. Pooling: Gated Attention aggregation of K feature vectors.
    4. Classifier: Linear layer.
    """

    def __init__(self):
        super(MILEfficientNet, self).__init__()

        # 1. Load Pretrained Backbone
        # num_classes=0 returns the global pool features (flat vector)
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0
        )

        # 2. Modify Stem for 12 Channels + Grouped Conv
        # Original stem: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        old_stem = self.backbone.conv_stem

        # Validate assumptions about the backbone
        assert old_stem.in_channels == 3, "Backbone expected to have 3 input channels"

        # Create new stem
        # We use groups=4 to isolate modalities (3 channels per group)
        new_stem = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None,
            groups=Config.GROUPS,
        )

        # 3. Initialize Weights
        # PyTorch Grouped Conv Weight Shape: (Out, In/Groups, K, K)
        # Old Shape: (32, 3, 3, 3)
        # New Shape: (32, 12/4, 3, 3) -> (32, 3, 3, 3)
        # The shapes match perfectly. We copy the pretrained RGB weights directly.
        # This effectively initializes each modality's processing path with
        # standard RGB edge detectors.
        with torch.no_grad():
            new_stem.weight.copy_(old_stem.weight)
            if old_stem.bias is not None:
                new_stem.bias.copy_(old_stem.bias)

        # Replace the stem in the backbone
        self.backbone.conv_stem = new_stem

        # 4. Attention Pooling Head
        # EfficientNet-B0 num_features is typically 1280
        self.feature_dim = self.backbone.num_features
        self.attention = GatedAttention(self.feature_dim)

        # 5. Classifier
        self.classifier = nn.Linear(self.feature_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input batch of shape (Batch, Num_Instances, Channels, H, W)

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        b, k, c, h, w = x.shape

        # Collapse Batch and Instance dimensions to process in parallel
        # (B*K, C, H, W)
        x = x.view(b * k, c, h, w)

        # Extract features using the backbone
        # Output shape: (B*K, Feature_Dim)
        features = self.backbone(x)

        # Reshape back to separate instances
        # (B, K, Feature_Dim)
        features = features.view(b, k, -1)

        # Aggregate features using Attention
        # aggregated: (B, Feature_Dim)
        aggregated_features, _ = self.attention(features)

        # Classification
        # (B, Num_Classes)
        logits = self.classifier(aggregated_features)

        return logits
