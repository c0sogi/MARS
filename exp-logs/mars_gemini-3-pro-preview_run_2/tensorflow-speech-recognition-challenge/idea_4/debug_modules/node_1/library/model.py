import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Learns to assign weights to temporal frames to emphasize important parts of the audio
    and suppress silence or background noise.
    """

    def __init__(self, input_dim, hidden_dim=128):
        """
        Args:
            input_dim (int): Dimensionality of the input features (C).
            hidden_dim (int): Hidden dimension for the attention mechanism.
        """
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (Batch, Channels, Time).

        Returns:
            Tensor: Pooled tensor of shape (Batch, Channels).
        """
        # Permute to (Batch, Time, Channels) for Linear layers
        x = x.permute(0, 2, 1)

        # Calculate attention weights: (Batch, Time, 1)
        weights = self.attention(x)

        # Apply weighted sum over the time dimension
        # (Batch, Time, Channels) * (Batch, Time, 1) -> Sum over dim 1
        out = torch.sum(x * weights, dim=1)

        return out


class ConvNeXtAudio(nn.Module):
    """
    ConvNeXt-Tiny based Audio Classifier.
    Uses a pretrained ConvNeXt-Tiny backbone modified for 1-channel spectrogram input,
    followed by Attention Pooling and a Linear Classifier.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        super(ConvNeXtAudio, self).__init__()

        # Load the model with standard 3 channels first
        # global_pool='' and num_classes=0 removes the default head
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # Modify the first conv layer to accept 1 channel (Spectrogram)
        self._modify_first_layer()

        # Get the feature dimension of the backbone (typically 768 for Tiny)
        self.num_features = self.backbone.num_features

        # Attention Pooling Head
        self.att_pool = AttentionPooling(self.num_features)

        # Final Classification Layer
        self.fc = nn.Linear(self.num_features, num_classes)

    def _modify_first_layer(self):
        """
        Replaces the first convolutional layer (stem) to accept 1-channel input.
        Initializes the new weights by averaging the original RGB weights.
        """
        # In timm's ConvNeXt, the stem is a Sequential block, and index 0 is the Conv2d
        old_conv = self.backbone.stem[0]

        # Create a new Conv2d layer with in_channels=1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize weights by averaging the original RGB weights
        # old_conv.weight shape: (Out, 3, K, K)
        # mean(1) shape: (Out, K, K) -> unsqueeze(1) -> (Out, 1, K, K)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace the layer in the backbone
        self.backbone.stem[0] = new_conv

    def forward(self, x):
        """
        Args:
            x (Tensor): Input spectrogram of shape (Batch, 1, Freq, Time).

        Returns:
            Tensor: Logits of shape (Batch, NumClasses).
        """
        # Pass through backbone
        # Output shape: (Batch, Channels, Freq', Time')
        x = self.backbone(x)

        # Average over the Frequency dimension (dim 2)
        # We preserve the Time dimension for the attention mechanism
        # Shape becomes: (Batch, Channels, Time')
        x = x.mean(dim=2)

        # Apply Attention Pooling over Time
        # Output shape: (Batch, Channels)
        x = self.att_pool(x)

        # Classification
        # Output shape: (Batch, NumClasses)
        x = self.fc(x)

        return x
