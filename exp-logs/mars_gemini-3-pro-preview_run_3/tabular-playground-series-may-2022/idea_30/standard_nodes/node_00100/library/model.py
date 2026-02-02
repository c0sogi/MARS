import torch
import torch.nn as nn
import library.config as config


class HPFEModel(nn.Module):
    def __init__(self, vocab_sizes, num_continuous):
        """
        Heterogeneous Parallel Funnel Ensemble (HPFE) Model.

        Constructs 5 independent neural network streams within a single computational graph.
        Each stream has its own embedding layers (fixed dimension) and MLP backbone
        as defined in library.config.STREAMS_CONFIG.

        Args:
            vocab_sizes (list[int]): List containing the vocabulary size for each categorical feature.
            num_continuous (int): The number of continuous features.
        """
        super(HPFEModel, self).__init__()

        self.streams = nn.ModuleList()

        # Iterate through the stream configurations defined in config.py
        for stream_cfg in config.STREAMS_CONFIG:
            emb_dim = stream_cfg["emb_dim"]
            hidden_layers = stream_cfg["hidden_layers"]
            dropout_rate = stream_cfg["dropout"]

            # --- 1. Independent Embedding Layers ---
            # Create a dedicated embedding layer for each categorical feature for this stream.
            # This allows each stream to learn its own representation (resolution) of the categories.
            embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=v, embedding_dim=emb_dim)
                    for v in vocab_sizes
                ]
            )

            # Calculate total input dimension for the MLP
            # Input = (Num_Categorical * Emb_Dim) + Num_Continuous
            num_categorical = len(vocab_sizes)
            input_dim = (num_categorical * emb_dim) + num_continuous

            # --- 2. MLP Backbone ---
            layers = []
            in_features = input_dim

            for h_dim in hidden_layers:
                layers.append(nn.Linear(in_features, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                in_features = h_dim

            # Final projection to a single logit (binary classification)
            layers.append(nn.Linear(in_features, 1))

            # Store components in a ModuleDict to keep them organized per stream
            stream_module = nn.ModuleDict(
                {"embeddings": embeddings, "backbone": nn.Sequential(*layers)}
            )

            self.streams.append(stream_module)

    def forward(self, continuous, categorical):
        """
        Forward pass processing inputs through all 5 streams in parallel.

        Args:
            continuous (torch.Tensor): Tensor of continuous features (Batch, Num_Continuous).
            categorical (torch.Tensor): Tensor of categorical indices (Batch, Num_Categorical).

        Returns:
            list[torch.Tensor]: A list of 5 tensors, where each tensor contains the
                                logits (unnormalized scores) from one stream.
                                Shape of each tensor: (Batch, 1).
        """
        outputs = []

        for stream in self.streams:
            embeddings_layers = stream["embeddings"]
            backbone = stream["backbone"]

            # --- 1. Process Embeddings ---
            embedded_list = []
            # Iterate over each categorical feature column and its corresponding embedding layer
            # We assume categorical tensor columns align with vocab_sizes order
            for i, emb_layer in enumerate(embeddings_layers):
                # categorical[:, i] selects the column of indices for feature i
                emb = emb_layer(categorical[:, i])
                embedded_list.append(emb)

            if embedded_list:
                # Concatenate all embeddings: (Batch, Num_Categorical * Emb_Dim)
                cat_embeddings = torch.cat(embedded_list, dim=1)
                # Concatenate with continuous features: (Batch, Total_Dim)
                x = torch.cat([cat_embeddings, continuous], dim=1)
            else:
                # Fallback if no categorical features exist
                x = continuous

            # --- 2. Process Backbone ---
            logit = backbone(x)
            outputs.append(logit)

        return outputs
