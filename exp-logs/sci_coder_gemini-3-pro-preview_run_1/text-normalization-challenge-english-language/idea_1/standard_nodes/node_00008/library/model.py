import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class BiLSTMTagger(nn.Module):
    """
    Bi-directional LSTM Tagger for Sequence Labeling.

    Architecture:
    1. Embedding Layer: Converts token IDs to dense vectors.
    2. Bi-LSTM Layer: Processes sequence context in both directions.
    3. Dropout Layer: Regularization.
    4. Linear Layer: Projects LSTM outputs to class logits.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = Config.EMBEDDING_DIM,
        hidden_dim: int = Config.HIDDEN_DIM,
        num_layers: int = Config.NUM_LAYERS,
        dropout: float = Config.DROPOUT,
        padding_idx: int = Config.PAD_TOKEN_ID,
        bidirectional: bool = Config.BIDIRECTIONAL,
    ):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            num_classes (int): Number of target classes.
            embedding_dim (int): Dimension of token embeddings.
            hidden_dim (int): Dimension of LSTM hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability.
            padding_idx (int): Index used for padding in embeddings.
            bidirectional (bool): Whether to use a bi-directional LSTM.
        """
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # LSTM dropout is only applied between layers, so we disable it if num_layers=1
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        # Calculate output dimension of LSTM
        self.lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.lstm_output_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        seq_len: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing token IDs.
            seq_len (torch.Tensor, optional): Tensor of shape (batch_size) containing actual sequence lengths.
                                              Used for packing padded sequences.
            attention_mask (torch.Tensor, optional): Tensor of shape (batch_size, seq_len).
                                                     Not strictly needed if seq_len is provided for packing,
                                                     but included for interface compatibility.

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, num_classes).
        """
        # 1. Embedding
        # Shape: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)

        # 2. LSTM Processing
        if seq_len is not None:
            # Pack the sequence to ignore padding computations in LSTM
            # enforce_sorted=False allows unsorted batches (though slightly less efficient)
            packed_embedded = pack_padded_sequence(
                embedded, seq_len.cpu(), batch_first=True, enforce_sorted=False
            )

            packed_output, _ = self.lstm(packed_embedded)

            # Unpack back to padded tensor
            # total_length ensures the output matches input_ids length even if the longest sequence was shorter
            lstm_out, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=input_ids.size(1)
            )
        else:
            # Fallback if lengths are not provided (treats padding as normal tokens)
            lstm_out, _ = self.lstm(embedded)

        # 3. Dropout & Classification
        # Shape: (batch_size, seq_len, lstm_output_dim)
        out = self.dropout(lstm_out)

        # Shape: (batch_size, seq_len, num_classes)
        logits = self.classifier(out)

        return logits
