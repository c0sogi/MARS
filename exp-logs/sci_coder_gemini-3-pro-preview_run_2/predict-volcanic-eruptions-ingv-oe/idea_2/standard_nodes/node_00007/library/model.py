import torch
import torch.nn as nn
import timm
from library.config import Config


class HybridModel(nn.Module):
    """
    A Hybrid Neural Network architecture for Seismic Eruption Prediction.

    Combines:
    1. A 2D CNN Branch (ResNet18) processing Log-Mel Spectrograms.
    2. An MLP Branch processing engineered statistical features.

    The outputs are fused to predict the scalar time_to_eruption.
    """

    def __init__(self, num_tabular_features):
        """
        Args:
            num_tabular_features (int): The dimension of the statistical feature vector.
        """
        super().__init__()

        # ---------------------------------------------------------
        # 1. CNN Branch (Spectrograms)
        # ---------------------------------------------------------
        # Create backbone with no classifier (returns feature vector)
        self.cnn = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Modify the first convolutional layer to accept N_SENSORS channels
        # ResNet usually uses 'conv1', EfficientNet uses 'conv_stem'
        first_conv_name = None
        if hasattr(self.cnn, "conv1"):
            first_conv_name = "conv1"
        elif hasattr(self.cnn, "conv_stem"):
            first_conv_name = "conv_stem"

        if first_conv_name:
            old_conv = getattr(self.cnn, first_conv_name)

            # Create new layer with correct input channels
            new_conv = nn.Conv2d(
                in_channels=Config.NUM_SENSORS,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            # Initialize weights: Average the original RGB weights and replicate/use
            # Shape is (Out, In, H, W). We average over dim 1 (In=3).
            with torch.no_grad():
                avg_weight = torch.mean(old_conv.weight, dim=1, keepdim=True)
                # Replicate across the new number of channels to preserve filter structure
                new_conv.weight.copy_(avg_weight.repeat(1, Config.NUM_SENSORS, 1, 1))
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

            setattr(self.cnn, first_conv_name, new_conv)

        # Get the output dimension of the CNN backbone
        self.cnn_out_dim = self.cnn.num_features

        # ---------------------------------------------------------
        # 2. MLP Branch (Tabular Features)
        # ---------------------------------------------------------
        mlp_layers = []
        in_dim = num_tabular_features

        for dim in Config.MLP_HIDDEN_DIMS:
            mlp_layers.append(nn.Linear(in_dim, dim))
            mlp_layers.append(nn.BatchNorm1d(dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(Config.DROPOUT))
            in_dim = dim

        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp_out_dim = in_dim

        # ---------------------------------------------------------
        # 3. Fusion Head
        # ---------------------------------------------------------
        fusion_in_dim = self.cnn_out_dim + self.mlp_out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_in_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(fusion_in_dim // 2, 1),
        )

    def forward(self, spectrogram, tabular_features):
        """
        Forward pass of the hybrid model.

        Args:
            spectrogram (torch.Tensor): Shape (Batch, 10, F, T)
            tabular_features (torch.Tensor): Shape (Batch, Num_Features)

        Returns:
            torch.Tensor: Prediction of shape (Batch,)
        """
        # CNN Branch
        # Input: (B, 10, F, T) -> Output: (B, cnn_out_dim)
        cnn_emb = self.cnn(spectrogram)

        # MLP Branch
        # Input: (B, Num_Feats) -> Output: (B, mlp_out_dim)
        mlp_emb = self.mlp(tabular_features)

        # Fusion
        combined = torch.cat([cnn_emb, mlp_emb], dim=1)

        # Final Prediction
        output = self.head(combined)

        # Remove last dimension to return (Batch,)
        return output.squeeze(1)
