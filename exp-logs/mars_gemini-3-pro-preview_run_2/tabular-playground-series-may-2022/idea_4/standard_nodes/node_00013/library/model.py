import torch
import torch.nn as nn
from library.config import Config


class GatedLinearBlock(nn.Module):
    """
    A single block of the Gated Wide MLP.
    Performs the operation: y = BN(Dropout((W1*x + b1) * Sigmoid(W2*x + b2)))
    """

    def __init__(self, in_features, out_features, dropout_rate):
        super(GatedLinearBlock, self).__init__()

        # Content branch (W1 * x + b1)
        self.linear_act = nn.Linear(in_features, out_features)

        # Gating branch (W2 * x + b2)
        self.linear_gate = nn.Linear(in_features, out_features)

        # Activation for the gate
        # SwiGLU variant uses SiLU instead of Sigmoid
        self.act = nn.SiLU()

        # Regularization: Batch Norm followed by Dropout
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Calculate content and gate
        content = self.linear_act(x)
        gate = self.act(self.linear_gate(x))

        # Gated Linear Unit operation: Element-wise multiplication
        y = content * gate

        # Apply Batch Norm and Dropout
        y = self.bn(y)
        y = self.dropout(y)

        return y


class GatedWideMLP(nn.Module):
    """
    Gated Wide MLP (GLU-MLP) for Manufacturing Control Data.

    Architecture:
    1. Embeddings for categorical tokens (f_27).
    2. Concatenation of flattened embeddings and continuous features.
    3. Sequence of GatedLinearBlocks (Wide layers).
    4. Final output with Sigmoid activation.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_continuous=len(Config.NUM_COLS),
        hidden_dims=Config.HIDDEN_DIMS,
        dropout_rate=Config.DROPOUT,
    ):
        super(GatedWideMLP, self).__init__()

        # 1. Embedding Layer
        # We use the vocab_size determined during data processing (passed as arg or default)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

        # 2. Calculate Input Dimension
        # Flattened embedding size = Sequence Length * Embedding Dimension
        flattened_embed_dim = Config.SEQ_LEN * embed_dim
        input_dim = num_continuous + flattened_embed_dim

        # 3. Build Gated Backbone
        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(GatedLinearBlock(current_dim, h_dim, dropout_rate))
            current_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # 4. Output Head
        self.output_linear = nn.Linear(current_dim, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, continuous, tokens):
        """
        Forward pass of the model.

        Args:
            continuous (torch.Tensor): Normalized continuous features. Shape: (Batch, 30)
            tokens (torch.Tensor): Tokenized categorical features. Shape: (Batch, 10)

        Returns:
            torch.Tensor: Probability of class 1. Shape: (Batch, 1)
        """
        # Embed tokens: (B, 10) -> (B, 10, Embed_Dim)
        embeds = self.embedding(tokens)

        # Flatten embeddings: (B, 10, Embed_Dim) -> (B, 10 * Embed_Dim)
        embeds_flat = embeds.view(embeds.size(0), -1)

        # Concatenate continuous features and flattened embeddings
        # Shape: (B, 30 + 320) = (B, 350)
        x = torch.cat([continuous, embeds_flat], dim=1)

        # Pass through Gated Backbone
        x = self.backbone(x)

        # Final prediction
        logits = self.output_linear(x)
        probs = self.output_activation(logits)

        return probs
