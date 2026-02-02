import torch
import torch.nn as nn
from library.config import Config


class SingleStream(nn.Module):
    """
    A single independent stream of the HPFE architecture.
    Implements a Standard Funnel MLP (Pure Deep).
    """

    def __init__(self, vocab_sizes, num_cont, hidden_layers, dropout_rate):
        super(SingleStream, self).__init__()

        # 1. Independent Entity Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=Config.EMBEDDING_DIM)
                for v in vocab_sizes
            ]
        )

        # Calculate input dimensions
        total_embed_dim = len(vocab_sizes) * Config.EMBEDDING_DIM
        deep_input_dim = total_embed_dim + num_cont

        # 2. Deep Path (The Funnel)
        layers = []
        in_dim = deep_input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        # Final projection to 1 output logit
        layers.append(nn.Linear(in_dim, 1))

        self.deep_mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat: LongTensor of shape (Batch, Num_Cat_Feats)
            x_cont: FloatTensor of shape (Batch, Num_Cont_Feats)
        """
        # Process Embeddings
        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            embeds.append(emb_layer(x_cat[:, i]))

        # Flatten and concatenate embeddings
        cat_embeds = torch.cat(embeds, dim=1)

        # Deep Path Input: Early Fusion (Embeddings + Continuous)
        deep_in = torch.cat([cat_embeds, x_cont], dim=1)

        # Output
        return self.deep_mlp(deep_in)


class SRPFEModel(nn.Module):
    """
    Selective-Residual Parallel Funnel Ensemble (SR-PFE).
    Synthesizes Heterogeneous Parallel Funnel backbone with Selective Wide-and-Deep topology.
    """

    def __init__(self, vocab_sizes, num_cont_features):
        super(SRPFEModel, self).__init__()

        self.streams = nn.ModuleList()

        # Instantiate 5 independent streams based on heterogeneous configuration
        # Config.STREAM_CONFIGS handles the variations in capacity (width) and regularization (dropout)
        for config in Config.STREAM_CONFIGS:
            stream = SingleStream(
                vocab_sizes=vocab_sizes,
                num_cont=num_cont_features,
                hidden_layers=config["hidden_layers"],
                dropout_rate=config["dropout"],
            )
            self.streams.append(stream)

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat: LongTensor (Batch, Num_Cat)
            x_cont: FloatTensor (Batch, Num_Cont)

        Returns:
            Tensor of shape (Batch, Num_Streams) containing logits for each stream.
        """
        stream_outputs = []

        # Execute all streams
        for stream in self.streams:
            stream_outputs.append(stream(x_cat, x_cont))

        # Concatenate outputs along the feature dimension
        # Result shape: (Batch, 5)
        # This allows independent loss calculation per stream
        return torch.cat(stream_outputs, dim=1)
