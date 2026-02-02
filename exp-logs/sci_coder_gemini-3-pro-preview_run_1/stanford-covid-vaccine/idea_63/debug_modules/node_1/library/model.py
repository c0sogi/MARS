import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config, SinusoidalPositionalEmbedding, OrthogonalBiLSTM

# Alias the imported classes to match the prompt's naming convention
SinusoidalDistanceEmbedding = SinusoidalPositionalEmbedding
OrthogonalBiLSTMBlock = OrthogonalBiLSTM


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Used to aggregate outputs from the Stem and all Backbone blocks.
    """

    def __init__(self, num_inputs):
        super().__init__()
        # Initialize weights to zeros -> Uniform distribution after softmax
        self.weights = nn.Parameter(torch.zeros(num_inputs))

    def forward(self, inputs):
        """
        Args:
            inputs: List of tensors, each of shape (Batch, Seq_Len, Hidden_Dim)
        Returns:
            Tensor of shape (Batch, Seq_Len, Hidden_Dim)
        """
        # Stack inputs: (Num_Inputs, Batch, Seq_Len, Hidden_Dim)
        stacked = torch.stack(inputs, dim=0)

        # Compute softmax weights: (Num_Inputs, 1, 1, 1) to broadcast
        w = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1)

        # Weighted sum
        return (stacked * w).sum(dim=0)


class RNAModel(nn.Module):
    """
    Orthogonally-Initialized High-Capacity Wide-Stream BiLSTM Model.

    Architecture:
    1. Embeddings: Sequence (128) + Loop (64) + Distance (64) = 256 dim.
    2. Stem: BiLSTM projecting 256 -> 512. No dropout. Orthogonal Init.
    3. Backbone: 6 Blocks of OrthogonalBiLSTM (Pre-LN, BiLSTM, Dropout, Residual). Width 512.
    4. Aggregation: Scalar Mixture of Stem + 6 Blocks.
    5. Head: Linear Projection 512 -> 3 targets.
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        self.seq_emb = nn.Embedding(Config.SEQ_VOCAB_SIZE, Config.SEQ_EMBED_DIM)
        self.loop_emb = nn.Embedding(Config.LOOP_VOCAB_SIZE, Config.LOOP_EMBED_DIM)
        # Fixed Signed Sinusoidal Encodings
        self.dist_emb = SinusoidalDistanceEmbedding(Config.DIST_EMBED_DIM)

        input_dim = Config.SEQ_EMBED_DIM + Config.LOOP_EMBED_DIM + Config.DIST_EMBED_DIM

        # 2. Stem
        # Projects concatenated embeddings to the residual stream width (512)
        # Bidirectional: 256 hidden per direction
        self.stem = nn.LSTM(
            input_size=input_dim,
            hidden_size=Config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Backbone
        # 6 Blocks maintaining the 512 width
        self.blocks = nn.ModuleList(
            [
                OrthogonalBiLSTMBlock(
                    Config.HIDDEN_DIM, Config.HIDDEN_DIM, dropout=Config.DROPOUT
                )
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 4. Aggregation
        # Inputs: 1 (Stem) + 6 (Blocks) = 7
        self.mixture = ScalarMixture(Config.NUM_LAYERS + 1)

        # 5. Head
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

        # Apply Orthogonal Initialization to Stem
        self._init_stem()

    def _init_stem(self):
        """
        Explicitly applies Orthogonal Initialization to the Stem LSTM weights
        to ensure gradient stability in the wide network.
        """
        for name, param in self.stem.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, seq, loop, dist):
        # Embeddings
        s = self.seq_emb(seq)  # (B, L, 128)
        l = self.loop_emb(loop)  # (B, L, 64)
        d = self.dist_emb(dist)  # (B, L, 64)

        # Concatenation (Early Fusion)
        x = torch.cat([s, l, d], dim=-1)  # (B, L, 256)

        # Stem
        x, _ = self.stem(x)  # (B, L, 512)

        # Collect outputs for mixture (strictly excluding raw inputs)
        outputs = [x]

        # Backbone
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # Aggregation
        x_mixed = self.mixture(outputs)  # (B, L, 512)

        # Head
        logits = self.head(x_mixed)  # (B, L, 3)

        return logits
