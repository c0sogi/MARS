import torch
import torch.nn as nn
from library.config import STREAM_CONFIGS, EMBEDDING_DIM


class DSRPEModel(nn.Module):
    """
    Deep Selective-Residual Parallel Ensemble (DSR-PE) Model.

    Architecture:
    - 5 Independent Streams (defined in library.config).
    - Each stream possesses its own independent Entity Embeddings.
    - Each stream implements a 'Selective Wide-and-Deep' topology:
        - Deep Path: Funnel MLP processing fused inputs (Flattened Embeddings + Continuous).
        - Wide Path: Linear Residual processing ONLY Continuous inputs.
    """

    def __init__(self, vocab_sizes, cont_dim):
        """
        Initialize the DSR-PE Model.

        Args:
            vocab_sizes (list or np.array): List of vocabulary sizes for each categorical feature.
            cont_dim (int): Number of continuous features (input dimension for Wide Path).
        """
        super(DSRPEModel, self).__init__()

        self.streams = nn.ModuleList()
        self.num_cat = len(vocab_sizes)

        # Calculate the input dimension for the Deep Path
        # It receives: Flattened Embeddings (Num_Cat * 16) + Continuous Features
        deep_input_dim = (self.num_cat * EMBEDDING_DIM) + cont_dim

        for config in STREAM_CONFIGS:
            stream = nn.ModuleDict()

            # 1. Independent Embeddings
            # We create a specific embedding layer for each categorical feature.
            # These are NOT shared between streams.
            embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=int(size), embedding_dim=EMBEDDING_DIM)
                    for size in vocab_sizes
                ]
            )
            stream["embeddings"] = embeddings

            # 2. Deep Path (The Funnel)
            # Processes fused data: Embeddings + Continuous
            layers = []
            in_dim = deep_input_dim

            for hidden_dim in config["hidden_layers"]:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=config["dropout"]))
                in_dim = hidden_dim

            # Final projection to scalar logit
            layers.append(nn.Linear(in_dim, 1))
            stream["deep_path"] = nn.Sequential(*layers)

            # 3. Wide Path (The Selective Residual)
            # Processes ONLY Continuous data via a linear projection
            stream["wide_path"] = nn.Linear(cont_dim, 1)

            self.streams.append(stream)

    def forward(self, cat_inputs, cont_inputs):
        """
        Forward pass for the ensemble.

        Args:
            cat_inputs (torch.Tensor): Categorical inputs [Batch, Num_Cat].
            cont_inputs (torch.Tensor): Continuous inputs [Batch, Cont_Dim].

        Returns:
            list[torch.Tensor]: A list of 5 tensors, each containing the logits [Batch, 1]
                                from one of the parallel streams.
        """
        outputs = []

        for stream in self.streams:
            # --- Embedding Lookup ---
            # Process each categorical feature through its specific embedding layer
            emb_vectors = []
            for i, emb_layer in enumerate(stream["embeddings"]):
                # cat_inputs[:, i] shape: [Batch] -> [Batch, Embed_Dim]
                emb_vectors.append(emb_layer(cat_inputs[:, i]))

            # Flatten embeddings: [Batch, Num_Cat * Embed_Dim]
            emb_flat = torch.cat(emb_vectors, dim=1)

            # --- Deep Path Execution ---
            # Input: Concatenation of Flattened Embeddings and Continuous Features
            deep_in = torch.cat([emb_flat, cont_inputs], dim=1)
            deep_out = stream["deep_path"](deep_in)

            # --- Wide Path Execution ---
            # Input: Only Continuous Features
            wide_out = stream["wide_path"](cont_inputs)

            # --- Stream Output ---
            # Sum of Deep and Wide paths
            stream_logit = deep_out + wide_out
            outputs.append(stream_logit)

        return outputs
