import torch
import torch.nn as nn
from library.config import Config


class BiLSTMDualHead(nn.Module):
    """
    Bi-Directional LSTM with Dual-Head Prediction.

    This model processes a sentence with a missing word and simultaneously predicts:
    1. The location of the missing word (Location Head).
    2. The identity of the missing word (Generation Head).

    Architecture:
    - Embedding Layer
    - Bi-Directional LSTM Encoder
    - Location Head (Linear -> 1)
    - Generation Head (Linear -> Vocab Size)
    """

    def __init__(
        self,
        vocab_size=None,
        embedding_dim=None,
        hidden_dim=None,
        lstm_layers=None,
        dropout=None,
    ):
        """
        Initializes the model architecture.

        Args:
            vocab_size (int, optional): Size of the vocabulary. Defaults to Config.VOCAB_SIZE.
            embedding_dim (int, optional): Dimension of embeddings. Defaults to Config.EMBEDDING_DIM.
            hidden_dim (int, optional): Dimension of LSTM hidden state. Defaults to Config.HIDDEN_DIM.
            lstm_layers (int, optional): Number of LSTM layers. Defaults to Config.LSTM_LAYERS.
            dropout (float, optional): Dropout probability. Defaults to Config.DROPOUT.
        """
        super(BiLSTMDualHead, self).__init__()

        # Use Config defaults if arguments are not provided
        self.vocab_size = vocab_size if vocab_size is not None else Config.VOCAB_SIZE
        self.embedding_dim = (
            embedding_dim if embedding_dim is not None else Config.EMBEDDING_DIM
        )
        self.hidden_dim = hidden_dim if hidden_dim is not None else Config.HIDDEN_DIM
        self.lstm_layers = (
            lstm_layers if lstm_layers is not None else Config.LSTM_LAYERS
        )
        self.dropout = dropout if dropout is not None else Config.DROPOUT

        # 1. Embedding Layer
        # Maps integer token IDs to dense vectors.
        # We assume padding_idx=0 based on the tokenizer implementation.
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
        )

        # 2. Bi-Directional LSTM Encoder
        # Processes the sequence in both directions to capture context.
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0.0,
        )

        # The output dimension of a BiLSTM is hidden_dim * 2 (forward + backward states)
        self.lstm_output_dim = self.hidden_dim * 2

        # 3. Location Head
        # Binary classification for each token: "Is the missing word immediately after this token?"
        # Output shape: (batch_size, seq_len, 1)
        self.location_head = nn.Linear(self.lstm_output_dim, 1)

        # 4. Generation Head
        # Multi-class classification for each token: "If the missing word is here, what is it?"
        # Output shape: (batch_size, seq_len, vocab_size)
        self.generation_head = nn.Linear(self.lstm_output_dim, self.vocab_size)

    def forward(self, input_ids):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing token IDs.

        Returns:
            tuple: (loc_logits, word_logits)
                - loc_logits (torch.Tensor): Unnormalized scores for gap location.
                  Shape: (batch_size, seq_len, 1)
                - word_logits (torch.Tensor): Unnormalized scores for word prediction.
                  Shape: (batch_size, seq_len, vocab_size)
        """
        # 1. Embed inputs
        # Shape: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)

        # 2. LSTM Encoding
        # lstm_out Shape: (batch_size, seq_len, hidden_dim * 2)
        # We ignore the hidden/cell states (h_n, c_n) as we need the full sequence output.
        lstm_out, _ = self.lstm(embedded)

        # 3. Location Prediction
        # Shape: (batch_size, seq_len, 1)
        loc_logits = self.location_head(lstm_out)

        # 4. Word Generation Prediction
        # Shape: (batch_size, seq_len, vocab_size)
        word_logits = self.generation_head(lstm_out)

        return loc_logits, word_logits
