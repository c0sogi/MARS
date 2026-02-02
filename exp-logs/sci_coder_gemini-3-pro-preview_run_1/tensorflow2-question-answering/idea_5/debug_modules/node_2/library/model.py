import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from library.config import Config


class QCBiGRU(nn.Module):
    """
    Question-Conditioned Bi-Directional GRU Model.

    Architecture:
    1. Embeddings: Shared embedding layer for Question and Candidate.
    2. Question Aggregation: Max Pooling over Question tokens.
    3. Conditioning: Concatenation of Question Vector + Candidate Embedding at each step.
    4. Encoder: Bi-Directional GRU processing the conditioned sequence.
    5. Long Answer Head: Classifier on concatenated final hidden states.
    6. Short Answer Head: Token-wise classifier for Start/End logits.
    """

    def __init__(self, vocab_size, embedding_matrix=None):
        super(QCBiGRU, self).__init__()

        self.hidden_size = Config.HIDDEN_SIZE
        self.embedding_dim = Config.EMBEDDING_DIM

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)

        # Load pre-trained embeddings if provided
        if embedding_matrix is not None:
            # Ensure it's a tensor
            if isinstance(embedding_matrix, np.ndarray):
                embedding_matrix = torch.from_numpy(embedding_matrix).float()

            # Verify shape matches config
            if embedding_matrix.shape == (vocab_size, self.embedding_dim):
                self.embedding.weight.data.copy_(embedding_matrix)
                # Freeze embeddings as per description ("fixed, pre-trained")
                self.embedding.weight.requires_grad = False
            else:
                print(
                    f"[Model] Warning: Embedding matrix shape {embedding_matrix.shape} "
                    f"does not match vocab/dim {(vocab_size, self.embedding_dim)}. "
                    f"Initializing randomly."
                )

        # 2. Bi-Directional GRU
        # Input size is embedding_dim (candidate token) + embedding_dim (question vector)
        self.gru_input_size = self.embedding_dim * 2

        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.hidden_size,
            num_layers=Config.NUM_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0,
        )

        # 3. Dropout
        self.dropout = nn.Dropout(Config.DROPOUT)

        # 4. Long Answer Classifier
        # Inputs: Concatenation of final forward and backward hidden states
        # Shape: hidden_size * 2 (bidirectional)
        self.long_answer_fc = nn.Linear(self.hidden_size * 2, 1)

        # 5. Short Answer Tagger
        # Inputs: Hidden states at each time step (forward + backward)
        # Outputs: 2 logits per token (start_logit, end_logit)
        self.short_answer_fc = nn.Linear(self.hidden_size * 2, 2)

    def forward(self, question_ids, candidate_ids):
        """
        Args:
            question_ids: (batch_size, max_q_len)
            candidate_ids: (batch_size, max_seq_len)

        Returns:
            long_prob: (batch_size, 1) - Probability of being a long answer
            start_logits: (batch_size, max_seq_len)
            end_logits: (batch_size, max_seq_len)
        """
        batch_size = candidate_ids.size(0)
        seq_len = candidate_ids.size(1)

        # --- Embeddings ---
        # (batch, q_len, emb_dim)
        q_embeds = self.embedding(question_ids)
        # (batch, seq_len, emb_dim)
        c_embeds = self.embedding(candidate_ids)

        # --- Question Aggregation (Max Pooling) ---
        # We take the max value across the sequence dimension (dim=1)
        # Result: (batch, emb_dim)
        q_vector, _ = torch.max(q_embeds, dim=1)

        # --- Conditioning ---
        # Expand q_vector to match candidate sequence length for concatenation
        # (batch, 1, emb_dim) -> (batch, seq_len, emb_dim)
        q_expanded = q_vector.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate candidate embeddings with question vector at every step
        # (batch, seq_len, emb_dim * 2)
        rnn_input = torch.cat([c_embeds, q_expanded], dim=2)

        # --- RNN Pass ---
        # outputs: (batch, seq_len, hidden_size * 2) containing hidden states for all steps
        # hidden: (num_layers * 2, batch, hidden_size) containing final hidden state
        outputs, hidden = self.gru(rnn_input)

        # --- Long Answer Classification ---
        # We need the final hidden states from the forward and backward directions of the last layer.
        # hidden layout: (num_layers * num_directions, batch, hidden_size)
        # With bidirectional=True, last two indices are the forward and backward states of the last layer.

        # Concatenate forward and backward final states
        # (batch, hidden_size * 2)
        global_rep = torch.cat((hidden[-2], hidden[-1]), dim=1)

        global_rep = self.dropout(global_rep)
        long_logits = self.long_answer_fc(global_rep)
        long_prob = torch.sigmoid(long_logits)

        # --- Short Answer Tagging ---
        # Process the full sequence output
        outputs = self.dropout(outputs)

        # Project to start/end logits
        # (batch, seq_len, 2)
        span_logits = self.short_answer_fc(outputs)

        # Split into start and end logits
        # (batch, seq_len)
        start_logits = span_logits[:, :, 0]
        end_logits = span_logits[:, :, 1]

        return long_prob, start_logits, end_logits


def get_model(load_weights=False):
    """
    Factory function to initialize the model.
    Loads vocab size and embedding matrix from cache if available.

    Args:
        load_weights (bool): If True, attempts to load trained weights from Config.MODEL_SAVE_PATH.

    Returns:
        model (QCBiGRU): Initialized model instance.
    """
    # Determine Vocab Size
    vocab_size = Config.MAX_VOCAB_SIZE
    if os.path.exists(Config.VOCAB_PATH):
        try:
            vocab_df = pd.read_parquet(Config.VOCAB_PATH)
            vocab_size = len(vocab_df)
        except Exception as e:
            print(f"[Model] Error loading vocab size from cache: {e}")

    # Load Embedding Matrix
    embedding_matrix = None
    if os.path.exists(Config.EMBEDDING_MATRIX_PATH):
        try:
            embedding_matrix = np.load(Config.EMBEDDING_MATRIX_PATH)
            print(
                f"[Model] Loaded embedding matrix with shape {embedding_matrix.shape}"
            )
        except Exception as e:
            print(f"[Model] Failed to load embedding matrix: {e}")

    # Initialize Model
    model = QCBiGRU(vocab_size, embedding_matrix)

    # Load Trained Weights
    if load_weights and os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"[Model] Loading model weights from {Config.MODEL_SAVE_PATH}")
        try:
            state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[Model] Failed to load model weights: {e}")

    return model
