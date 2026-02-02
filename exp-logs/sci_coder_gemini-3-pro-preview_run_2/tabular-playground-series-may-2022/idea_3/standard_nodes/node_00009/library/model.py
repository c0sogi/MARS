import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A Residual Block for the ResMLP architecture.
    Implements: x_{l+1} = Activation(x_l + F(x_l))
    F = Linear -> BN -> Activation -> Dropout -> Linear -> BN -> Dropout
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(dropout_rate)

        self.final_act = nn.GELU()

    def forward(self, x):
        residual = x

        # F(x) path
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.drop2(out)

        # Add residual and apply final activation
        return self.final_act(residual + out)


class ResMLP(nn.Module):
    """
    Deep Residual MLP Architecture.
    Combines categorical embeddings and continuous features, projects them to a
    hidden dimension, and passes them through a stack of residual blocks.
    """

    def __init__(self):
        super(ResMLP, self).__init__()

        # ----------------------------------------------------------------------
        # Configuration
        # ----------------------------------------------------------------------
        vocab_size = Config.VOCAB_SIZE
        embed_dim = Config.EMBEDDING_DIM
        seq_len = Config.F_27_SEQ_LENGTH
        num_continuous = Config.NUM_CONTINUOUS_FEATURES
        hidden_dim = Config.HIDDEN_DIM
        num_blocks = Config.NUM_RES_BLOCKS
        dropout_rate = Config.DROPOUT_RATE

        # ----------------------------------------------------------------------
        # Layers
        # ----------------------------------------------------------------------

        # 1. Input Processing
        # Embedding for the 10 character tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Calculate size of flattened embeddings: 10 * 32 = 320
        flat_embed_dim = seq_len * embed_dim

        # Total input dimension: 30 (continuous) + 320 (categorical) = 350
        input_dim = num_continuous + flat_embed_dim

        # Projection to uniform hidden dimension
        self.projection = nn.Linear(input_dim, hidden_dim)

        # 2. Residual Backbone
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        # 3. Output Head
        # Outputs logits. Sigmoid is applied during inference or via loss function.
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, continuous, categorical):
        """
        Args:
            continuous (torch.Tensor): Shape (B, 30), float32
            categorical (torch.Tensor): Shape (B, 10), int64 (indices)

        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # 1. Process Categorical Data
        # (B, 10) -> (B, 10, 32)
        emb = self.embedding(categorical)
        # Flatten -> (B, 320)
        emb_flat = emb.view(emb.size(0), -1)

        # 2. Concatenate with Continuous Data
        # (B, 30) + (B, 320) -> (B, 350)
        x = torch.cat([continuous, emb_flat], dim=1)

        # 3. Project to Hidden Dimension
        # (B, 350) -> (B, 512)
        x = self.projection(x)

        # 4. Pass through Residual Blocks
        for block in self.blocks:
            x = block(x)

        # 5. Output Head
        # (B, 512) -> (B, 1)
        logits = self.head(x)

        return logits
