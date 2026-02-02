import torch
import torch.nn as nn
import timm
from library.config import Config


class SpecEfficientNet(nn.Module):
    """
    Spectrogram-Conversion CNN using EfficientNet-B0 backbone.

    This model takes 2D spectrogram images (generated from EEG signals) as input
    and outputs the probability distribution over the 6 harmful brain activity classes.
    """

    def __init__(self, config=Config, pretrained=True):
        """
        Args:
            config: Configuration class or object containing model parameters.
            pretrained (bool): Whether to load ImageNet pre-trained weights.
        """
        super().__init__()
        self.config = config

        # 1. Backbone: EfficientNet-B0
        # We use timm to create the model.
        # - in_chans=config.IN_CHANNELS: Adapts the first conv layer to the input depth.
        #   If input is 3 channels (RGB), it loads weights directly.
        #   If input were 1 channel (Grayscale), timm would sum/average the RGB weights.
        # - num_classes=0 & global_pool='': Removes the default classifier and pooling,
        #   returning the raw spatial feature maps (B, C, H, W).
        self.backbone = timm.create_model(
            config.MODEL_NAME,
            pretrained=pretrained,
            in_chans=config.IN_CHANNELS,
            num_classes=0,
            global_pool="",
            drop_path_rate=config.DROP_PATH_RATE,
        )

        # Get the number of output channels from the backbone (typically 1280 for B0)
        self.num_features = self.backbone.num_features

        # 2. Custom Classifier Head
        # As per requirements: Global Average Pooling -> Linear -> Softmax
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(config.DROP_RATE)
        self.fc = nn.Linear(self.num_features, config.NUM_CLASSES)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).
                              Channels should match config.IN_CHANNELS (3).

        Returns:
            torch.Tensor: Predicted probabilities for each class of shape (Batch, 6).
                          Sum of probabilities across the last dimension equals 1.
        """
        # Pass through backbone to get spatial features
        # Shape: (Batch, num_features, H', W')
        features = self.backbone(x)

        # Apply Global Average Pooling
        # Shape: (Batch, num_features, 1, 1)
        pooled = self.global_pool(features)

        # Flatten the features
        # Shape: (Batch, num_features)
        flattened = pooled.view(pooled.size(0), -1)

        # Apply Dropout
        flattened = self.dropout(flattened)

        # Pass through the fully connected layer
        # Shape: (Batch, num_classes)
        logits = self.fc(flattened)

        # Apply Softmax to get probabilities
        # Shape: (Batch, num_classes)
        probs = self.softmax(logits)

        return probs
