import torch
import torch.nn as nn
from library.config import MLP_PARAMS


class DualBranchMLP(nn.Module):
    """
    A dual-branch neural network architecture that fuses semantic text embeddings
    with dense metadata features.

    Branch 1 (Text): Processes SBERT embeddings with high dropout for regularization.
    Branch 2 (Meta): Processes tabular metadata with Batch Normalization and low dropout.
    Fusion: Concatenates branch outputs and predicts a probability via a sigmoid activation.
    """

    def __init__(
        self,
        meta_dim,
        sbert_dim=MLP_PARAMS["sbert_dim"],
        hidden_dim_text=MLP_PARAMS["hidden_dim_text"],
        hidden_dim_meta=MLP_PARAMS["hidden_dim_meta"],
        fusion_dim=MLP_PARAMS["fusion_dim"],
        dropout_text=MLP_PARAMS["dropout_text"],
        dropout_meta=MLP_PARAMS["dropout_meta"],
    ):
        """
        Args:
            meta_dim (int): Dimension of the input metadata features.
            sbert_dim (int): Dimension of the input SBERT embeddings.
            hidden_dim_text (int): Hidden dimension for the text branch.
            hidden_dim_meta (int): Hidden dimension for the metadata branch.
            fusion_dim (int): Dimension of the fused representation before final projection.
            dropout_text (float): Dropout rate for the text branch.
            dropout_meta (float): Dropout rate for the metadata branch.
        """
        super(DualBranchMLP, self).__init__()

        # Branch 1: Semantic Text Processing
        # High dropout is used here to regularize the distributed representation
        self.text_branch = nn.Sequential(
            nn.Linear(sbert_dim, hidden_dim_text), nn.ReLU(), nn.Dropout(dropout_text)
        )

        # Branch 2: Augmented Metadata Processing
        # Batch Normalization is crucial here for stabilizing the raw/ratio feature distributions
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim_meta),
            nn.BatchNorm1d(hidden_dim_meta),
            nn.ReLU(),
            nn.Dropout(dropout_meta),
        )

        # Fusion and Classification
        # Concatenates outputs and projects to a single probability score
        combined_dim = hidden_dim_text + hidden_dim_meta
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim), nn.ReLU(), nn.Linear(fusion_dim, 1)
        )

    def forward(self, text_emb, meta_features):
        """
        Forward pass of the model.

        Args:
            text_emb (torch.Tensor): Tensor of shape (batch_size, sbert_dim).
            meta_features (torch.Tensor): Tensor of shape (batch_size, meta_dim).

        Returns:
            torch.Tensor: Probability scores of shape (batch_size, 1).
        """
        # Process inputs through respective branches
        text_out = self.text_branch(text_emb)
        meta_out = self.meta_branch(meta_features)

        # Fuse representations
        combined = torch.cat([text_out, meta_out], dim=1)

        # Generate logits
        logits = self.fusion(combined)

        # Return probabilities
        return torch.sigmoid(logits)
