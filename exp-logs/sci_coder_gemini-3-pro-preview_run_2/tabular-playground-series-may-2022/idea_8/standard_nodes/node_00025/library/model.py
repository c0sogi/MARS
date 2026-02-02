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


class ProjectedTransition(nn.Module):
    """
    Transition between stages with dimension change.
    Uses a projected residual connection to preserve signal identity
    across dimensionality changes.
    """

    def __init__(self, in_dim, out_dim):
        super(ProjectedTransition, self).__init__()
        # Main path transformation
        self.main_path = nn.Linear(in_dim, out_dim)
        # Skip path projection (1x1 equivalent for dense layers)
        self.skip_path = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = nn.PReLU()

    def forward(self, x):
        main = self.main_path(x)
        skip = self.skip_path(x)

        # Sum main and skip paths
        out = main + skip

        # Stabilize
        out = self.bn(out)
        out = self.activation(out)
        return out


class WideDeepResFunnel(nn.Module):
    """
    Wide & Deeply Supervised ResFunnel-GLU Architecture.
    Combines a shallow wide linear model with a deep hierarchical funnel network.
    """

    def __init__(self):
        super(WideDeepResFunnel, self).__init__()

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

        # --- Stream 2: Wide Path ---
        # Linear projection from raw input to scalar
        self.wide_head = nn.Linear(self.input_dim, 1)

        # --- Stream 1: Deep ResFunnel Backbone ---

        # Initial projection to Stage 1 width
        self.input_proj = nn.Linear(self.input_dim, self.stages[0])
        self.input_bn = nn.BatchNorm1d(self.stages[0])
        self.input_act = nn.PReLU()

        # Stage 1
        # Stack of Residual Gated Blocks
        self.stage1 = nn.Sequential(
            ResidualGatedBlock(self.stages[0], self.dropout),
            ResidualGatedBlock(self.stages[0], self.dropout),
        )
        # Deep Supervision Head 1
        self.aux1_head = nn.Linear(self.stages[0], 1)

        # Transition 1 -> 2
        self.trans1 = ProjectedTransition(self.stages[0], self.stages[1])

        # Stage 2
        self.stage2 = nn.Sequential(
            ResidualGatedBlock(self.stages[1], self.dropout),
            ResidualGatedBlock(self.stages[1], self.dropout),
        )
        # Deep Supervision Head 2
        self.aux2_head = nn.Linear(self.stages[1], 1)

        # Transition 2 -> 3
        self.trans2 = ProjectedTransition(self.stages[1], self.stages[2])

        # Stage 3
        self.stage3 = nn.Sequential(
            ResidualGatedBlock(self.stages[2], self.dropout),
            ResidualGatedBlock(self.stages[2], self.dropout),
        )

        # Final Deep Head
        self.deep_head = nn.Linear(self.stages[2], 1)

    def forward(self, numeric, categorical):
        """
        Forward pass of the model.

        Args:
            numeric: (Batch, 30) FloatTensor containing standardized numerical features.
            categorical: (Batch, 10) LongTensor containing tokenized character features.

        Returns:
            final_logits: The combined output of Wide and Deep streams.
            aux1_logits: Output from Stage 1 (for deep supervision).
            aux2_logits: Output from Stage 2 (for deep supervision).
        """
        # 1. Process Embeddings
        # categorical: (B, 10) -> (B, 10, 32)
        emb = self.embedding(categorical)
        # Flatten embeddings: (B, 320)
        emb_flat = emb.view(emb.size(0), -1)

        # 2. Create Input Vector
        # Concatenate numerical and flattened embeddings: (B, 30 + 320) -> (B, 350)
        x_in = torch.cat([numeric, emb_flat], dim=1)

        # 3. Wide Path Execution
        wide_logits = self.wide_head(x_in)

        # 4. Deep Stream Execution

        # Initial Projection
        x = self.input_proj(x_in)
        x = self.input_bn(x)
        x = self.input_act(x)

        # Stage 1
        x = self.stage1(x)
        aux1_logits = self.aux1_head(x)

        # Transition Stage 1 -> Stage 2
        x = self.trans1(x)

        # Stage 2
        x = self.stage2(x)
        aux2_logits = self.aux2_head(x)

        # Transition Stage 2 -> Stage 3
        x = self.trans2(x)

        # Stage 3
        x = self.stage3(x)

        # Final Deep Output
        deep_logits = self.deep_head(x)

        # 5. Fusion
        # Sum the logits from the Deep Head and the Wide Path
        final_logits = deep_logits + wide_logits

        return final_logits, aux1_logits, aux2_logits
