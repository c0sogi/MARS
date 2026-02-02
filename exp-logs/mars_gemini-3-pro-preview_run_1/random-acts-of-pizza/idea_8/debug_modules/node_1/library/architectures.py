import torch
import torch.nn as nn


class GatedFusionMLP(nn.Module):
    """
    A Multi-Layer Perceptron architecture that fuses semantic text embeddings,
    community history embeddings, and tabular metadata using a Gated Fusion mechanism.

    The architecture uses the metadata to generate a 'gate' that modulates the
    semantic representations, allowing the model to weigh the narrative evidence
    based on the user's credibility metrics.
    """

    def __init__(self, text_dim, comm_dim, tab_dim, hidden_dim, dropout=0.3):
        """
        Args:
            text_dim (int): Dimension of the request text embeddings (e.g., SBERT 384).
            comm_dim (int): Dimension of the community history embeddings.
            tab_dim (int): Number of tabular metadata features.
            hidden_dim (int): Size of the hidden layers.
            dropout (float): Dropout probability.
        """
        super(GatedFusionMLP, self).__init__()

        # Combined dimension of semantic inputs (Request Text + Community History)
        self.sem_dim = text_dim + comm_dim

        # Branch 3: Tabular Metadata Encoder
        # Processes raw/scaled metadata into a latent representation
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated Fusion Mechanism
        # Generates a gate vector from the metadata context to modulate semantics.
        # Output dimension matches sem_dim for element-wise multiplication.
        self.gate_generator = nn.Sequential(
            nn.Linear(hidden_dim, self.sem_dim), nn.Sigmoid()
        )

        # Final Classifier
        # Input: Gated Semantic Vector + Encoded Metadata
        self.classifier = nn.Sequential(
            nn.Linear(self.sem_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_emb, comm_emb, tab_features):
        """
        Forward pass of the model.

        Args:
            text_emb (torch.Tensor): Tensor of shape (batch, text_dim).
            comm_emb (torch.Tensor): Tensor of shape (batch, comm_dim).
            tab_features (torch.Tensor): Tensor of shape (batch, tab_dim).

        Returns:
            torch.Tensor: Probability scores of shape (batch, 1).
        """
        # 1. Construct Unified Semantic Vector
        # Concatenate request text and community history
        sem_vector = torch.cat([text_emb, comm_emb], dim=1)

        # 2. Encode Metadata
        tab_encoded = self.tab_encoder(tab_features)

        # 3. Apply Gated Fusion
        # Generate gate based on metadata context
        gate = self.gate_generator(tab_encoded)

        # Modulate semantic vector (Element-wise multiplication)
        # If metadata suggests low credibility, the gate can suppress the text signal.
        gated_sem = sem_vector * gate

        # 4. Final Classification
        # Combine the modulated semantics with the metadata context
        combined = torch.cat([gated_sem, tab_encoded], dim=1)
        output = self.classifier(combined)

        return output
