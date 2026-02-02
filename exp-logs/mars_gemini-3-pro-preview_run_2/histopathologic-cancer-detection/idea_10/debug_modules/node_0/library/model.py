import torch
import torch.nn as nn
import timm
from library.config import Config


class ConvNeXtTinyMSD(nn.Module):
    """
    ConvNeXt-Tiny with Multi-Sample Dropout (MSD) Head.

    Backbone: ConvNeXt-Tiny (Pretrained)
    Head: Global Average Pooling -> LayerNorm -> Multi-Sample Dropout -> Linear

    Strategy:
    - Uses 'num_classes=0' in timm to get pooled, normalized features.
    - Implements multiple parallel dropout masks feeding into a shared linear layer.
    - During training, returns stacked logits for MSD loss calculation.
    - During inference, returns averaged logits for ensemble prediction.
    """

    def __init__(self):
        super().__init__()

        # Load backbone
        # num_classes=0 keeps the Global Pool and final Norm, but removes the Linear layer
        self.backbone = timm.create_model(
            Config.model_name,
            pretrained=Config.pretrained,
            num_classes=0,
            in_chans=Config.in_channels,
        )

        # Determine input features for the classifier
        self.in_features = self.backbone.num_features

        # Multi-Sample Dropout Configuration
        self.use_msd = Config.use_multi_sample_dropout
        self.msd_count = Config.multi_sample_dropout_count
        self.msd_rate = Config.multi_sample_dropout_rate

        # Create multiple dropout layers
        # We use a ModuleList to ensure they are registered properly
        self.dropouts = nn.ModuleList(
            [nn.Dropout(self.msd_rate) for _ in range(self.msd_count)]
        )

        # Shared Linear Classifier
        self.fc = nn.Linear(self.in_features, Config.num_classes)

        # Initialize weights for the new linear layer
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the classifier head."""
        nn.init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # Extract features: (Batch, Num_Features)
        # timm with num_classes=0 returns pooled and normalized features
        features = self.backbone(x)

        if self.training and self.use_msd:
            # Multi-Sample Dropout Training Mode
            # Generate logits for each dropout mask independently
            logits_list = []
            for dropout in self.dropouts:
                # Apply dropout then linear
                logits_list.append(self.fc(dropout(features)))

            # Stack logits: (Batch, MSD_Count, Num_Classes)
            return torch.stack(logits_list, dim=1)

        else:
            # Inference or Standard Training Mode
            # Average the predictions across all dropout masks (Internal Ensemble)
            logits_list = []
            for dropout in self.dropouts:
                logits_list.append(self.fc(dropout(features)))

            # Stack and compute mean: (Batch, Num_Classes)
            stacked_logits = torch.stack(logits_list, dim=1)
            return torch.mean(stacked_logits, dim=1)


class MSDLoss(nn.Module):
    """
    Custom Loss function for Multi-Sample Dropout.
    Wraps BCEWithLogitsLoss to handle stacked logits.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        """
        Args:
            preds: Model output.
                   Shape (B, N, C) if MSD training.
                   Shape (B, C) if standard/inference.
            targets: Ground truth. Shape (B,) or (B, 1).
        """
        # Ensure targets have the correct shape (B, 1) for BCE
        if targets.ndim == 1:
            targets = targets.view(-1, 1)

        # Check for MSD output (3 dimensions: Batch, Sample, Class)
        if preds.ndim == 3:
            loss = 0.0
            num_samples = preds.size(1)

            # Calculate loss for each dropout sample and accumulate
            for i in range(num_samples):
                loss += self.criterion(preds[:, i, :], targets)

            # Return average loss
            return loss / num_samples

        else:
            # Standard loss calculation
            return self.criterion(preds, targets)


def get_model():
    """Factory function to create the model."""
    return ConvNeXtTinyMSD()


def get_loss_fn():
    """Factory function to create the loss function."""
    return MSDLoss()
