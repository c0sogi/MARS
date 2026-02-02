import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualGatedBlock(nn.Module):
    """
    Implements the Residual Gated Block:
    x_out = x_in + Dropout(BatchNorm(GLU(Linear(x_in))))
    """

    def __init__(self, input_dim, dropout_rate):
        super(ResidualGatedBlock, self).__init__()
        # GLU halves the dimension, so Linear must project to 2 * input_dim
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.bn = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Linear -> GLU
        # x: (B, D) -> (B, 2D) -> (B, D)
        out = self.linear(x)
        out = F.glu(out, dim=1)

        # BN -> Dropout
        out = self.bn(out)
        out = self.dropout(out)

        # Residual connection
        return x + out


class GatedTransition(nn.Module):
    """
    Transition between stages with dimension change using GLU.
    Maintains a projected residual connection.
    """

    def __init__(self, in_dim, out_dim, dropout_rate):
        super(GatedTransition, self).__init__()
        # Main path: Linear -> GLU -> BN -> Dropout
        self.main_linear = nn.Linear(in_dim, out_dim * 2)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Skip path projection
        self.skip_path = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # Main path
        out = self.main_linear(x)
        out = F.glu(out, dim=1)
        out = self.bn(out)
        out = self.dropout(out)

        # Skip path
        skip = self.skip_path(x)

        return out + skip


class ResFunnelGLU(nn.Module):
    """
    ResFunnel-GLU Architecture.
    Deep hierarchical funnel network with residual GLU blocks and gated transitions.
    """

    def __init__(self):
        super(ResFunnelGLU, self).__init__()

        # --- Configuration ---
        self.num_numeric = Config.NUM_CONTINUOUS_FEATURES
        self.seq_len = Config.F_27_SEQ_LEN
        self.vocab_size = Config.VOCAB_SIZE
        self.emb_dim = Config.EMBEDDING_DIM
        self.stages = Config.FUNNEL_STAGES  # [512, 256, 128]
        self.dropout = Config.DROPOUT_RATE

        # --- Input Processing ---
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)

        # Calculate flattened input dimension
        # 30 numeric + (10 * 32) embedding = 350
        self.input_dim = self.num_numeric + (self.seq_len * self.emb_dim)

        # --- Deep ResFunnel Backbone ---

        # Initial projection to Stage 1 width
        self.input_proj = nn.Linear(self.input_dim, self.stages[0])
        self.input_bn = nn.BatchNorm1d(self.stages[0])
        self.input_act = nn.PReLU()

        # Stage 1
        self.stage1 = nn.Sequential(
            ResidualGatedBlock(self.stages[0], self.dropout),
            ResidualGatedBlock(self.stages[0], self.dropout),
        )

        # Transition 1 -> 2
        self.trans1 = GatedTransition(self.stages[0], self.stages[1], self.dropout)

        # Stage 2
        self.stage2 = nn.Sequential(
            ResidualGatedBlock(self.stages[1], self.dropout),
            ResidualGatedBlock(self.stages[1], self.dropout),
        )

        # Transition 2 -> 3
        self.trans2 = GatedTransition(self.stages[1], self.stages[2], self.dropout)

        # Stage 3
        self.stage3 = nn.Sequential(
            ResidualGatedBlock(self.stages[2], self.dropout),
            ResidualGatedBlock(self.stages[2], self.dropout),
        )

        # Final Head
        self.head = nn.Linear(self.stages[2], 1)

    def forward(self, numeric, categorical):
        """
        Forward pass of the model.
        """
        # 1. Process Embeddings
        emb = self.embedding(categorical)
        emb_flat = emb.view(emb.size(0), -1)

        # 2. Create Input Vector
        x = torch.cat([numeric, emb_flat], dim=1)

        # 3. Deep Stream Execution
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_act(x)

        # Stage 1
        x = self.stage1(x)

        # Transition Stage 1 -> Stage 2
        x = self.trans1(x)

        # Stage 2
        x = self.stage2(x)

        # Transition Stage 2 -> Stage 3
        x = self.trans2(x)

        # Stage 3
        x = self.stage3(x)

        # Final Output
        logits = self.head(x)

        return logits
