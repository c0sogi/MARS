import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling module to dynamically weight spatial features.

    Structure:
    Input (B, H, W, C) -> Flatten (B, N, C) -> MLP -> Scores (B, N, 1) -> Softmax -> Weighted Sum (B, C)
    """

    def __init__(self, in_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x shape: (B, H, W, C) from Swin Transformer
        B, H, W, C = x.shape

        # Flatten spatial dimensions: (B, N, C) where N = H * W
        x_flat = x.view(B, H * W, C)

        # Compute attention scores
        # weights shape: (B, N, 1)
        weights = self.attention(x_flat)

        # Weighted sum of features
        # out shape: (B, C)
        out = torch.sum(weights * x_flat, dim=1)

        return out


class SwinAudioClassifier(nn.Module):
    """
    Swin Transformer (Tiny) adapted for 1-channel audio spectrograms with Attention Pooling.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(SwinAudioClassifier, self).__init__()

        # 1. Load Pretrained Swin Transformer (Tiny)
        # We use DEFAULT weights (ImageNet)
        print("Loading Swin Transformer (Tiny) backbone...")
        original_model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

        # 2. Adapt Input Layer for 1-Channel Spectrograms
        # The first layer is part of features[0], which is a Sequential containing the PatchPartition
        # features[0][0] is the Conv2d layer
        original_conv = original_model.features[0][0]

        # Create new Conv2d with in_channels=1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
        )

        # Initialize weights by summing the original RGB weights
        # This preserves the activation statistics expected by the rest of the network
        with torch.no_grad():
            new_conv.weight.copy_(original_conv.weight.sum(dim=1, keepdim=True))
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        # Replace the layer in the model
        original_model.features[0][0] = new_conv

        # 3. Extract Backbone Components
        # We keep the feature extractor and the final LayerNorm
        self.features = original_model.features
        self.norm = original_model.norm

        # Determine embedding dimension (Swin-T usually 768 at the end)
        # We can get it from the original head's input features
        embed_dim = original_model.head.in_features

        # 4. Define Custom Head
        self.pool = AttentionPooling(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Input x: (Batch, 1, 224, 224)

        # Pass through Swin Transformer backbone
        # Output x: (Batch, H, W, C)
        x = self.features(x)
        x = self.norm(x)

        # Apply Attention Pooling
        # Output x: (Batch, C)
        x = self.pool(x)

        # Final Classification
        # Output x: (Batch, Num_Classes)
        x = self.classifier(x)

        return x
