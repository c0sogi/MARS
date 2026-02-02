import torch
import torch.nn as nn
import timm
from library.config import Config


class DualStreamEfficientNet(nn.Module):
    """
    Dual-Stream EfficientNet-B1 model for EEG and Spectrogram fusion.

    Architecture:
    1. Stream 1 (EEG Encoder): EfficientNet-B1 processing generated EEG Mel-Spectrograms.
    2. Stream 2 (Kaggle Encoder): EfficientNet-B1 processing pre-computed Kaggle Spectrograms.
    3. Fusion Head: Concatenates pooled features from both streams, applies Dropout,
       and maps to class probabilities via a Fully Connected layer.
    """

    def __init__(self, config=Config):
        """
        Initializes the model components based on the provided configuration.

        Args:
            config (Config): Configuration class containing model hyperparameters.
        """
        super(DualStreamEfficientNet, self).__init__()

        # ---------------------------------------------------------------------
        # Stream 1: EEG Encoder
        # ---------------------------------------------------------------------
        # num_classes=0 removes the classification head and returns the pooled features
        self.eeg_encoder = timm.create_model(
            config.BACKBONE,
            pretrained=config.PRETRAINED,
            num_classes=0,
            in_chans=config.IN_CHANNELS,
            drop_path_rate=config.DROP_PATH_RATE,
        )

        # ---------------------------------------------------------------------
        # Stream 2: Kaggle Spectrogram Encoder
        # ---------------------------------------------------------------------
        self.kaggle_encoder = timm.create_model(
            config.BACKBONE,
            pretrained=config.PRETRAINED,
            num_classes=0,
            in_chans=config.IN_CHANNELS,
            drop_path_rate=config.DROP_PATH_RATE,
        )

        # ---------------------------------------------------------------------
        # Fusion Head
        # ---------------------------------------------------------------------
        # Get the number of output features from the backbone (e.g., 1280 for B1)
        # We can infer this by checking the num_features attribute of the encoder
        if hasattr(self.eeg_encoder, "num_features"):
            n_features = self.eeg_encoder.num_features
        else:
            # Fallback or manual calculation if attribute is missing (unlikely in timm)
            # For EfficientNet-B1, it is typically 1280
            n_features = 1280

        # The fusion concatenates features from both streams, so input dim is 2 * n_features
        self.head = nn.Sequential(
            nn.Dropout(p=config.DROP_RATE),
            nn.Linear(2 * n_features, config.NUM_CLASSES),
        )

    def forward(self, eeg_spec, kaggle_spec):
        """
        Forward pass of the dual-stream network.

        Args:
            eeg_spec (torch.Tensor): EEG Mel-Spectrogram batch. Shape (B, C, H, W).
            kaggle_spec (torch.Tensor): Kaggle Spectrogram batch. Shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits (raw scores) for the 6 classes. Shape (B, 6).
        """
        # Extract features from Stream 1
        # Output shape: (Batch, n_features)
        eeg_feat = self.eeg_encoder(eeg_spec)

        # Extract features from Stream 2
        # Output shape: (Batch, n_features)
        kaggle_feat = self.kaggle_encoder(kaggle_spec)

        # Fuse features via concatenation
        # Output shape: (Batch, 2 * n_features)
        combined_feat = torch.cat([eeg_feat, kaggle_feat], dim=1)

        # Classification head
        logits = self.head(combined_feat)

        return logits
