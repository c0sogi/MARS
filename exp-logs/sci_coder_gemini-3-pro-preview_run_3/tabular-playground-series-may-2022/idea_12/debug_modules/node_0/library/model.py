import torch
import torch.nn as nn
import math
from library.config import Config


class SNNBlock(nn.Module):
    """
    Helper block for Self-Normalizing Networks.
    Consists of Linear -> SELU -> AlphaDropout.
    """

    def __init__(self, in_features, out_features, dropout_rate):
        super(SNNBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.SELU()
        self.dropout = nn.AlphaDropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.act(self.linear(x)))


class SelfNormalizingFunnelNet(nn.Module):
    """
    Self-Normalizing Funnel Network (SNN) architecture.
    Uses SELU activations and AlphaDropout to maintain stable mean/variance
    without explicit normalization layers (like Batch/Layer Norm).
    """

    def __init__(self, vocab_sizes, cont_dim):
        """
        Args:
            vocab_sizes (list[int]): List of vocabulary sizes for each categorical feature.
            cont_dim (int): Number of continuous features.
        """
        super(SelfNormalizingFunnelNet, self).__init__()

        # 1. Entity Embeddings
        # Create an embedding layer for each categorical feature
        # Dimension is fixed to 16 as per strategy
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=Config.EMBEDDING_DIM)
                for size in vocab_sizes
            ]
        )

        # Calculate total input dimension after flattening embeddings
        # Input = (Num_Categorical * Embedding_Dim) + Num_Continuous
        total_input_dim = (len(vocab_sizes) * Config.EMBEDDING_DIM) + cont_dim

        # 2. Backbone (Funnel Structure)
        # Decreasing layer widths defined in Config.HIDDEN_LAYERS
        layers = []
        in_dim = total_input_dim

        for hidden_dim in Config.HIDDEN_LAYERS:
            layers.append(SNNBlock(in_dim, hidden_dim, Config.DROPOUT_RATE))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 3. Output Head
        # Direct linear connection to output logit (no tapering)
        self.head = nn.Linear(in_dim, 1)

        # 4. Initialization
        # Explicitly apply LeCun Normal initialization required for SNNs
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Applies LeCun Normal initialization to Linear layers.
        This initializes weights from N(0, 1/fan_in), which is a mathematical
        requirement for the fixed-point dynamics of SELU activations.
        """
        if isinstance(m, nn.Linear):
            # LeCun Normal: N(0, 1/fan_in)
            nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(1.0 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            # Initialize embeddings with standard normal distribution
            # to match the expected input distribution of the SNN
            nn.init.normal_(m.weight, mean=0.0, std=1.0)

    def forward(self, x_cat, x_cont):
        """
        Forward pass of the network.

        Args:
            x_cat (torch.Tensor): Tensor of categorical indices [Batch, Num_Cat].
            x_cont (torch.Tensor): Tensor of continuous features [Batch, Num_Cont].

        Returns:
            torch.Tensor: Logits [Batch, 1].
        """
        # Process Embeddings
        embedded_features = []
        for i, emb_layer in enumerate(self.embeddings):
            # Extract the i-th categorical column and lookup embedding
            emb = emb_layer(x_cat[:, i])
            embedded_features.append(emb)

        # Concatenate all embeddings: [Batch, Num_Cat * Emb_Dim]
        x_emb = torch.cat(embedded_features, dim=1)

        # Early Fusion: Concatenate flattened embeddings with continuous features
        # [Batch, Total_Input_Dim]
        x = torch.cat([x_emb, x_cont], dim=1)

        # Pass through SNN Backbone
        x = self.backbone(x)

        # Output Head (returns logits)
        logits = self.head(x)

        return logits
