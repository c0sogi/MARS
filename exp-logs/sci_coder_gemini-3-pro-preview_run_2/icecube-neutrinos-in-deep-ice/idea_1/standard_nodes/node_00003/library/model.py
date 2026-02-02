import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    N_FEATURES,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    OUTPUT_DIM,
)


class NeutrinoBiGRU(nn.Module):
    """
    Temporal Recurrent Regressor (TRR) for Neutrino Direction Prediction.

    Architecture:
    1. Stem: 1D Convolution to project input features into a latent embedding.
    2. Backbone: Bidirectional GRU to capture temporal evolution of the pulse sequence.
    3. Head: MLP to regress the 3D direction vector from the aggregated hidden states.
    """

    def __init__(self):
        super(NeutrinoBiGRU, self).__init__()

        # Stem: Project N_FEATURES (6) -> EMBED_DIM (64)
        # Kernel size 1 acts as a learnable linear projection per timestep
        self.stem = nn.Conv1d(
            in_channels=N_FEATURES, out_channels=EMBED_DIM, kernel_size=1
        )

        # Backbone: Bi-Directional GRU
        self.gru = nn.GRU(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )

        # Head: MLP
        # Input is concatenation of final forward and backward hidden states (2 * HIDDEN_DIM)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, N_Features).

        Returns:
            torch.Tensor: Predicted 3D direction vector (Batch, 3).
        """
        # x shape: (Batch, Seq_Len, N_Features)

        # Permute for Conv1d: (Batch, N_Features, Seq_Len)
        x = x.permute(0, 2, 1)

        # Apply Stem Projection
        x = self.stem(x)

        # Permute back for GRU: (Batch, Seq_Len, Embed_Dim)
        x = x.permute(0, 2, 1)

        # Apply GRU Backbone
        # output: (Batch, Seq_Len, 2 * Hidden_Dim)
        # h_n: (Num_Layers * 2, Batch, Hidden_Dim)
        _, h_n = self.gru(x)

        # Extract final hidden states from the last layer
        # h_n is stacked as [Layer1_Fwd, Layer1_Bwd, Layer2_Fwd, Layer2_Bwd, ...]
        # We take the last two entries corresponding to the last layer's forward and backward states
        last_fwd = h_n[-2]
        last_bwd = h_n[-1]

        # Concatenate to form global event descriptor: (Batch, 2 * Hidden_Dim)
        global_desc = torch.cat([last_fwd, last_bwd], dim=1)

        # Predict direction vector: (Batch, 3)
        # Note: The output is not normalized here; normalization happens in Loss or Inference
        direction = self.head(global_desc)

        return direction


class NeutrinoLoss(nn.Module):
    """
    Cosine Similarity Loss for Angular Error Minimization.

    Optimizes the model by maximizing the cosine of the angle between
    the predicted vector and the true vector derived from azimuth/zenith.
    """

    def __init__(self):
        super(NeutrinoLoss, self).__init__()

    def forward(self, pred_vector, target_angles):
        """
        Args:
            pred_vector (torch.Tensor): Predicted vectors (Batch, 3).
            target_angles (torch.Tensor): Ground truth angles (Batch, 2) -> [azimuth, zenith].

        Returns:
            torch.Tensor: Scalar loss value = 1 - mean(cosine_similarity).
        """
        # 1. Normalize predicted vector to unit length
        # Adding epsilon for numerical stability is handled by F.normalize default (eps=1e-12)
        pred_norm = F.normalize(pred_vector, p=2, dim=1)

        # 2. Convert ground truth spherical coordinates to Cartesian unit vectors
        # target_angles: [azimuth, zenith]
        azimuth = target_angles[:, 0]
        zenith = target_angles[:, 1]

        # Spherical to Cartesian conversion
        sin_zenith = torch.sin(zenith)
        true_x = torch.cos(azimuth) * sin_zenith
        true_y = torch.sin(azimuth) * sin_zenith
        true_z = torch.cos(zenith)

        # Stack to form true unit vectors: (Batch, 3)
        true_vector = torch.stack([true_x, true_y, true_z], dim=1)

        # 3. Calculate Cosine Similarity
        # Dot product of two unit vectors equals the cosine of the angle between them
        # shape: (Batch,)
        cosine_sim = torch.sum(pred_norm * true_vector, dim=1)

        # 4. Compute Loss
        # We want to maximize cosine_sim (make it close to 1).
        # Loss = 1 - cosine_sim
        loss = 1.0 - torch.mean(cosine_sim)

        return loss
