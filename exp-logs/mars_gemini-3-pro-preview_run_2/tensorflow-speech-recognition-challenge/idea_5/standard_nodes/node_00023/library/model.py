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


class EfficientNetAudioClassifier(nn.Module):
    """
    EfficientNet-B0 adapted for 1-channel audio spectrograms with Attention Pooling.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(EfficientNetAudioClassifier, self).__init__()

        # 1. Load Pretrained EfficientNet-B0 (Cite solution_lesson_node_00020)
        print("Loading EfficientNet-B0 backbone...")
        original_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # 2. Adapt Input Layer for 1-Channel Spectrograms
        # features[0][0] is the first Conv2d layer
        original_conv = original_model.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Initialize weights by summing the original RGB weights (Cite solution_lesson_node_00019)
        with torch.no_grad():
            new_conv.weight.copy_(original_conv.weight.sum(dim=1, keepdim=True))

        original_model.features[0][0] = new_conv

        # 3. Extract Backbone
        self.features = original_model.features

        # Determine embedding dimension (EfficientNet-B0 is 1280)
        # The classifier input features tells us the dimension
        embed_dim = original_model.classifier[1].in_features

        # 4. Define Custom Head
        self.pool = AttentionPooling(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Input x: (Batch, 1, F, T)

        # Pass through backbone
        # Output x: (Batch, C, H, W)
        x = self.features(x)

        # Permute to (Batch, H, W, C) for AttentionPooling
        x = x.permute(0, 2, 3, 1)

        # Apply Attention Pooling
        # Output x: (Batch, C)
        x = self.pool(x)

        # Final Classification
        # Output x: (Batch, Num_Classes)
        x = self.classifier(x)

        return x
