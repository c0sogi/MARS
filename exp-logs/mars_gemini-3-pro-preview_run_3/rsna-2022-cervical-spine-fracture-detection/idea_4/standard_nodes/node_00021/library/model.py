import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class FractureMILModel(nn.Module):
    """
    Sequence-Smoothed Max-Pooling MIL Network.
    Uses ResNet18 backbone, 1D Convolution for context, and Global Max Pooling.
    Cite solution_lesson_node_00019 (Max Pooling > Attention)
    Cite solution_lesson_node_00020 (1D Conv > Positional Injection)
    """

    def __init__(self):
        super(FractureMILModel, self).__init__()

        # 1. Backbone: ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        # 2. Sequence Smoother (1D Convolution)
        # Models inter-slice dependencies (z-axis context)
        self.seq_smoother = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=3,
            padding=1,
        )

        # 3. Classifier Head
        self.classifier = nn.Linear(self.feature_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input volume of shape (Batch, Slices, Channels, Height, Width)

        Returns:
            torch.Tensor: Logits for C1-C7 of shape (Batch, 7)
        """
        batch_size, num_slices, channels, height, width = x.shape

        # --- 1. Feature Extraction ---
        # Reshape to process all slices in parallel: (B*S, C, H, W)
        x_flat = x.view(batch_size * num_slices, channels, height, width)

        # Pass through backbone
        features = self.backbone(x_flat)  # (B*S, 512, 1, 1)
        features = features.view(batch_size, num_slices, -1)  # (B, S, 512)

        # --- 2. Sequence Smoothing ---
        # Permute to (B, C, S) for Conv1d
        features = features.permute(0, 2, 1)
        features = self.seq_smoother(features)
        # Permute back to (B, S, C)
        features = features.permute(0, 2, 1)

        # --- 3. Classification per Slice ---
        logits_seq = self.classifier(features)  # (B, S, 7)

        # --- 4. Aggregation (Max Pooling) ---
        # Select the most confident slice for each class
        # This acts as a hard attention mechanism
        logits, _ = torch.max(logits_seq, dim=1)  # (B, 7)

        return logits
