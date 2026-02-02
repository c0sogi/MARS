import torch
import torch.nn as nn


class ParallelStream(nn.Module):
    """
    A single independent stream for the MORPE ensemble.
    Maintains its own embedding layers and MLP backbone to ensure
    decorrelated feature representations.
    """

    def __init__(self, num_cont, vocab_sizes, embed_dim, hidden_layers, dropout_rate):
        super().__init__()

        # Independent Embeddings for this stream
        # We use a ModuleList to hold an embedding layer for each categorical feature
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v_size, embedding_dim=embed_dim)
                for v_size in vocab_sizes
            ]
        )

        # Calculate input dimension for the MLP
        # Input = Continuous Features + (Number of Categorical Features * Embedding Dimension)
        total_embed_dim = len(vocab_sizes) * embed_dim
        input_dim = num_cont + total_embed_dim

        # Construct the MLP Backbone (Funnel Architecture)
        layers = []
        in_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        # Final output layer (Binary Classification Logit)
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, cat_data, cont_data):
        """
        Forward pass for a single stream.

        Args:
            cat_data: Tensor of shape (Batch, Num_Cat_Cols) containing integer indices.
            cont_data: Tensor of shape (Batch, Num_Cont_Cols) containing normalized continuous features.

        Returns:
            Tensor of shape (Batch, 1) containing logits.
        """
        # 1. Lookup Embeddings
        # Iterate over columns of cat_data and apply the corresponding embedding layer
        emb_list = [emb(cat_data[:, i]) for i, emb in enumerate(self.embeddings)]

        # 2. Concatenate Embeddings
        x_emb = torch.cat(emb_list, dim=1)

        # 3. Early Fusion: Concatenate Embeddings with Continuous Features
        x = torch.cat([x_emb, cont_data], dim=1)

        # 4. Pass through MLP
        return self.mlp(x)


class MORPE(nn.Module):
    """
    Multi-Objective Regularized Parallel Ensemble (MORPE).
    Container class that holds multiple ParallelStream instances.
    """

    def __init__(self, vocab_sizes_list, num_cont, embed_dim, stream_configs):
        super().__init__()
        self.streams = nn.ModuleList()

        # Instantiate each stream based on the configuration
        for cfg in stream_configs:
            stream = ParallelStream(
                num_cont=num_cont,
                vocab_sizes=vocab_sizes_list,
                embed_dim=embed_dim,
                hidden_layers=cfg["layers"],
                dropout_rate=cfg["dropout"],
            )
            self.streams.append(stream)

    def forward(self, cat_data, cont_data):
        """
        Forward pass for the ensemble.

        Args:
            cat_data: Tensor of shape (Batch, Num_Cat_Cols)
            cont_data: Tensor of shape (Batch, Num_Cont_Cols)

        Returns:
            List of Tensors, where each Tensor is (Batch, 1) containing logits from a specific stream.
        """
        # Propagate input through all streams independently
        return [stream(cat_data, cont_data) for stream in self.streams]
