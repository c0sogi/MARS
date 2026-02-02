import torch
import torch.nn as nn
from library.config import Config


class SemanticAnchorClassifier(nn.Module):
    """
    A dynamic classification model that predicts the position of markdown cells relative to code cells.

    It projects markdown embeddings into a query space and compares them against code cell embeddings
    (anchors) plus a learnable 'End-of-Notebook' token using dot-product attention.
    """

    def __init__(self, config: Config):
        """
        Initialize the SemanticAnchorClassifier.

        Args:
            config (Config): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.config = config

        # Projection Head: Maps frozen markdown embeddings (input_dim) to query space (input_dim).
        # Structure: Linear -> ReLU -> Dropout -> Linear
        # This acts as an adapter to align markdown semantics with the code/structure space.
        self.projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

        # Symmetric Projection for Code Embeddings (Anchors)
        self.code_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

        # Learnable 'End-of-Notebook' vector.
        # Represents the position after the last code cell.
        # Shape: (1, 1, input_dim)
        self.end_token = nn.Parameter(torch.randn(1, 1, config.input_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for the MLP and the end token."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize end token with a small normal distribution
        nn.init.normal_(self.end_token, mean=0, std=0.02)

    def forward(
        self, markdown_embeddings: torch.Tensor, code_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the classifier.

        Args:
            markdown_embeddings: Tensor of shape (Batch, Num_MD, Dim).
                                 Frozen embeddings of markdown cells.
            code_embeddings: Tensor of shape (Batch, Num_Code, Dim).
                             Frozen embeddings of code cells (anchors).

        Returns:
            logits: Tensor of shape (Batch, Num_MD, Num_Code + 1).
                    Unnormalized scores representing the likelihood of a markdown cell
                    appearing before each code cell (indices 0 to N-1) or at the end (index N).
        """
        batch_size = markdown_embeddings.size(0)

        # 1. Project Markdown Embeddings to Query Space
        # Shape: (Batch, Num_MD, Dim)
        queries = self.projection(markdown_embeddings)

        # 2. Project Code Embeddings to Key Space
        # Shape: (Batch, Num_Code, Dim)
        code_embeddings = self.code_projection(code_embeddings)

        # 3. Prepare Anchors
        # Expand end_token to match batch size: (Batch, 1, Dim)
        end_tokens_expanded = self.end_token.expand(batch_size, -1, -1)

        # Concatenate code embeddings with the end token
        # Shape: (Batch, Num_Code + 1, Dim)
        anchors = torch.cat([code_embeddings, end_tokens_expanded], dim=1)

        # 4. Compute Attention Scores (Dot Product)
        # Queries: (Batch, Num_MD, Dim)
        # Anchors Transposed: (Batch, Dim, Num_Code + 1)
        # Logits: (Batch, Num_MD, Num_Code + 1)
        logits = torch.bmm(queries, anchors.transpose(1, 2))

        return logits
