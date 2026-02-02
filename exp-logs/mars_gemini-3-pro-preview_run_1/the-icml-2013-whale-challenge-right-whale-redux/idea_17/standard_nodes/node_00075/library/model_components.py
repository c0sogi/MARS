import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextGatingBlock(nn.Module):
    """
    Context-Gating Block for Hierarchical Feature Aggregation.

    This block implements a Top-Down Attention mechanism where deeper,
    semantically rich features (context) are used to gate (filter)
    shallower, high-resolution features.

    Mechanism:
    1. Upsample 'context' features to match the spatial resolution of input 'x'.
    2. Project 'context' features to match the channel dimension of 'x'.
    3. Generate a soft attention mask using a Sigmoid activation.
    4. Apply the mask to 'x' via element-wise multiplication.
    """

    def __init__(self, in_channels, context_channels):
        """
        Args:
            in_channels (int): Number of channels in the shallow feature map (x).
            context_channels (int): Number of channels in the deep context feature map.
        """
        super(ContextGatingBlock, self).__init__()

        # Projection layer to align channel dimensions
        # 1x1 Convolution is used as a learnable projection
        self.project = nn.Sequential(
            nn.Conv2d(context_channels, in_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, context):
        """
        Args:
            x (torch.Tensor): Shallow features of shape (B, C_in, H, W)
            context (torch.Tensor): Deep context features of shape (B, C_ctx, H', W')

        Returns:
            torch.Tensor: Gated features of shape (B, C_in, H, W)
        """
        # 1. Upsample context to match x's spatial dimensions (H, W)
        # We use bilinear interpolation for smooth upsampling of feature maps
        target_size = x.shape[2:]

        # Only interpolate if spatial dimensions differ
        if context.shape[2:] != target_size:
            context_up = F.interpolate(
                context, size=target_size, mode="bilinear", align_corners=False
            )
        else:
            context_up = context

        # 2. Generate Attention Mask
        # (B, C_ctx, H, W) -> (B, C_in, H, W)
        mask = self.project(context_up)

        # 3. Apply Mask (Gating)
        # Element-wise multiplication emphasizes relevant regions in x
        return x * mask


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer for Temporal Sequences.

    Aggregates a sequence of feature vectors into a single vector using
    learnable attention weights. This allows the model to focus on
    informative time steps (e.g., the whale call) while suppressing noise.
    """

    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): Dimensionality of the input feature vectors.
        """
        super(AttentionPooling, self).__init__()
        # Learnable linear projection to compute attention scores
        # We use a simple linear layer: Score = W*x + b
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (Batch, Time, Features)

        Returns:
            torch.Tensor: Pooled representation of shape (Batch, Features)
        """
        # 1. Compute attention scores
        # (B, T, C) -> (B, T, 1)
        scores = self.attention_weights(x)

        # 2. Normalize scores to probabilities (weights)
        # Softmax over the time dimension (dim=1)
        weights = F.softmax(scores, dim=1)

        # 3. Weighted Aggregation
        # (B, T, C) * (B, T, 1) -> (B, T, C)
        weighted_sequence = x * weights

        # Sum over time dimension to get fixed-size vector
        # (B, T, C) -> (B, C)
        pooled_output = torch.sum(weighted_sequence, dim=1)

        return pooled_output
