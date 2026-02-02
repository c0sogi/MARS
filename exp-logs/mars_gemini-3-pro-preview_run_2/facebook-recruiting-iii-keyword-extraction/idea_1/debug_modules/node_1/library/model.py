import torch
import torch.nn as nn
from library import config


class DAN(nn.Module):
    """
    Deep Averaging Network (DAN) for text classification.

    Architecture:
    1. Embedding Layer: Maps word indices to dense vectors.
    2. Global Average Pooling: Averages embeddings across the sequence length (ignoring padding).
    3. Deep Classifier: A series of Dense -> BatchNorm -> ReLU -> Dropout layers.
    4. Output: Logits for multi-label classification.
    """

    def __init__(
        self,
        vocab_size=config.MAX_VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dims=config.HIDDEN_DIMS,
        output_dim=config.OUTPUT_DIM,
        dropout_rate=config.DROPOUT_RATE,
        padding_idx=0,
    ):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the word embeddings.
            hidden_dims (list): List of integers defining the dimensions of hidden layers.
            output_dim (int): Number of output classes (tags).
            dropout_rate (float): Dropout probability.
            padding_idx (int): Index used for padding tokens (ignored in averaging).
        """
        super(DAN, self).__init__()

        self.padding_idx = padding_idx

        # 1. Embedding Layer
        # padding_idx ensures the vector at this index is initialized to zeros
        # and not updated during training.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # 2. Classifier Head
        layers = []
        input_dim = embedding_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = h_dim

        # Final projection layer
        layers.append(nn.Linear(input_dim, output_dim))

        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, lengths=None):
        """
        Forward pass of the DAN model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing token indices.
            lengths (torch.Tensor, optional): Tensor of shape (batch_size,) containing actual sequence lengths.

        Returns:
            torch.Tensor: Logits of shape (batch_size, output_dim).
        """
        # input_ids: (batch_size, seq_len)

        # Create a mask for non-padding tokens
        # Shape: (batch_size, seq_len)
        mask = (input_ids != self.padding_idx).float()

        # Get embeddings
        # Shape: (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(input_ids)

        # Explicitly zero out padding embeddings (redundant with padding_idx but safe)
        embeddings = embeddings * mask.unsqueeze(-1)

        # Sum embeddings along the sequence dimension
        # Shape: (batch_size, embedding_dim)
        summed_embeddings = embeddings.sum(dim=1)

        # Determine lengths for averaging
        if lengths is not None:
            # Use provided lengths
            lens = lengths.to(embeddings.device).unsqueeze(1).float()
        else:
            # Calculate lengths from mask
            lens = mask.sum(dim=1, keepdim=True)

        # Clamp lengths to avoid division by zero (though empty sequences shouldn't exist ideally)
        lens = lens.clamp(min=1.0)

        # Compute mean embedding
        # Shape: (batch_size, embedding_dim)
        averaged_embeddings = summed_embeddings / lens

        # Pass through classifier
        # Shape: (batch_size, output_dim)
        logits = self.classifier(averaged_embeddings)

        return logits
