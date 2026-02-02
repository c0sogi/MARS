import numpy as np
import torch
import torch.nn as nn


class SNNModel(nn.Module):
    """
    Self-Normalizing Funnel Network (SNN) implementation.

    This architecture uses Scaled Exponential Linear Units (SELU) and AlphaDropout
    to induce self-normalizing properties, eliminating the need for Batch/Layer Normalization.
    It employs an 'Early Fusion' strategy where categorical embeddings are concatenated
    with continuous features before the first hidden layer.
    """

    def __init__(self, num_cont, vocab_sizes, embed_dim, hidden_layers, dropout_rate):
        """
        Args:
            num_cont (int): Number of continuous features.
            vocab_sizes (list[int]): List containing the vocabulary size for each categorical feature.
            embed_dim (int): Dimension of the entity embeddings.
            hidden_layers (list[int]): List of hidden layer widths (e.g., [512, 256, 128]).
            dropout_rate (float): Dropout rate for AlphaDropout.
        """
        super(SNNModel, self).__init__()

        # 1. Embeddings
        # Create a ModuleList of embeddings, one for each categorical feature
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # 2. Backbone Construction
        # Calculate total input dimension: continuous features + (num_categorical * embed_dim)
        input_dim = num_cont + (len(vocab_sizes) * embed_dim)

        layers = []
        in_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.SELU())
            layers.append(nn.AlphaDropout(p=dropout_rate))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # 3. Prediction Head
        # Projects the final representation to a single logit
        self.head = nn.Linear(in_dim, 1)

        # 4. Explicit Initialization
        self.initialize_weights()

    def initialize_weights(self):
        """
        Applies LeCun Normal initialization to Linear layers.

        Standard initialization methods (like Kaiming or Xavier) are not optimal for SELU.
        SNNs require weights to be sampled from a normal distribution with
        mean 0 and variance 1/fan_in to maintain fixed-point dynamics.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # LeCun Normal: std = sqrt(1 / fan_in)
                fan_in = m.weight.size(1)
                nn.init.normal_(m.weight, std=np.sqrt(1.0 / fan_in))

                # Biases should be initialized to zero
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_cont, x_cat):
        """
        Forward pass of the model.

        Args:
            x_cont (torch.Tensor): Continuous features, shape (batch_size, num_cont).
            x_cat (torch.Tensor): Categorical features, shape (batch_size, num_cat).
                                  Expected to be LongTensor.

        Returns:
            torch.Tensor: Logits, shape (batch_size, 1).
        """
        # 1. Process Embeddings
        embedded_features = []
        for i, emb_layer in enumerate(self.embeddings):
            # Extract the i-th categorical column and pass through its embedding layer
            embedded_features.append(emb_layer(x_cat[:, i]))

        # Flatten and concatenate all embeddings
        x_emb = torch.cat(embedded_features, dim=1)

        # 2. Early Fusion
        # Concatenate continuous features with the dense embedding vector
        x = torch.cat([x_cont, x_emb], dim=1)

        # 3. Backbone
        x = self.backbone(x)

        # 4. Head
        logits = self.head(x)

        return logits
