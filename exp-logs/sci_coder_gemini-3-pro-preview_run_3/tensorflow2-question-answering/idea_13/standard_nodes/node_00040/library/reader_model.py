import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class QIRNReader(nn.Module):
    """
    Query-Initialized Recurrent Network (QIRN) for Short Answer Extraction.

    Architecture:
    1. Embedding Layer (Pre-trained)
    2. Question Encoder (Unidirectional LSTM) -> Extract final state
    3. State Projection (Map Q-state to P-LSTM initial state)
    4. Paragraph Reader (Bidirectional LSTM, initialized with Q-state)
    5. Output Heads (Linear layers for Start/End logits)
    """

    def __init__(self, embedding_matrix):
        """
        Args:
            embedding_matrix (numpy.ndarray): Pre-trained embedding matrix.
        """
        super(QIRNReader, self).__init__()

        # 1. Embedding Layer
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_tensor, freeze=False, padding_idx=0
        )

        input_dim = Config.EMBEDDING_DIM
        hidden_dim = Config.READER_HIDDEN_DIM

        # 2. Question Encoder (Unidirectional)
        # Compresses the question into a context vector
        self.q_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # 3. State Initialization Projections
        # The Q-LSTM is unidirectional (1 direction), P-LSTM is bidirectional (2 directions).
        # We project the final (1, B, H) state of Q to (2, B, H) for P.
        # We need separate projections for Hidden state (h) and Cell state (c).
        self.h_init_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.c_init_proj = nn.Linear(hidden_dim, 2 * hidden_dim)

        # 4. Paragraph Reader (Bidirectional)
        # Processes the paragraph conditioned on the question via initialization
        self.p_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 5. Output Heads
        # Input is 2 * hidden_dim because P-LSTM is bidirectional
        self.dropout = nn.Dropout(Config.READER_DROPOUT)
        self.start_head = nn.Linear(2 * hidden_dim, 1)
        self.end_head = nn.Linear(2 * hidden_dim, 1)

    def forward(self, q_indices, p_indices):
        """
        Args:
            q_indices (torch.Tensor): Question indices (Batch, Max_Q_Len)
            p_indices (torch.Tensor): Paragraph indices (Batch, Max_P_Len)

        Returns:
            start_logits (torch.Tensor): (Batch, Max_P_Len)
            end_logits (torch.Tensor): (Batch, Max_P_Len)
        """
        batch_size = q_indices.size(0)

        # --- 1. Compute Lengths ---
        # Calculate actual lengths for packing.
        # Clamp min length to 1 to avoid errors with empty sequences (though data loader should prevent this)
        q_lens = (q_indices != 0).sum(dim=1).cpu()
        q_lens = torch.clamp(q_lens, min=1)

        p_lens = (p_indices != 0).sum(dim=1).cpu()
        p_lens = torch.clamp(p_lens, min=1)

        # --- 2. Embeddings ---
        q_embed = self.embedding(q_indices)  # (B, Q_Len, E)
        p_embed = self.embedding(p_indices)  # (B, P_Len, E)

        q_embed = self.dropout(q_embed)
        p_embed = self.dropout(p_embed)

        # --- 3. Question Encoding ---
        # Pack sequence to handle padding correctly and extract the true final state
        q_packed = pack_padded_sequence(
            q_embed, q_lens, batch_first=True, enforce_sorted=False
        )
        _, (q_h_n, q_c_n) = self.q_lstm(q_packed)

        # q_h_n shape: (1, B, H) -> Squeeze to (B, H)
        q_h_n = q_h_n.squeeze(0)
        q_c_n = q_c_n.squeeze(0)

        # --- 4. State Initialization ---
        # Project Q states to match P-LSTM bidirectional requirements
        # (B, H) -> (B, 2*H)
        p_h_0_flat = self.h_init_proj(q_h_n)
        p_c_0_flat = self.c_init_proj(q_c_n)

        # Reshape to (2, B, H) for bidirectional LSTM input
        # The first dimension is num_layers * num_directions
        p_h_0 = p_h_0_flat.view(batch_size, 2, -1).permute(1, 0, 2).contiguous()
        p_c_0 = p_c_0_flat.view(batch_size, 2, -1).permute(1, 0, 2).contiguous()

        # --- 5. Paragraph Reading ---
        # Initialize P-LSTM with projected Q states
        p_packed = pack_padded_sequence(
            p_embed, p_lens, batch_first=True, enforce_sorted=False
        )
        p_out_packed, _ = self.p_lstm(p_packed, (p_h_0, p_c_0))

        # Unpack to get padded output tensor
        # p_out shape: (B, P_Len, 2*H)
        p_out, _ = pad_packed_sequence(
            p_out_packed, batch_first=True, total_length=p_indices.size(1)
        )

        p_out = self.dropout(p_out)

        # --- 6. Prediction Heads ---
        # (B, P_Len, 2*H) -> (B, P_Len, 1) -> (B, P_Len)
        start_logits = self.start_head(p_out).squeeze(-1)
        end_logits = self.end_head(p_out).squeeze(-1)

        # --- 7. Masking Padding ---
        # Set logits at padding positions to a very large negative number
        # so they have 0 probability after softmax.
        p_mask = p_indices == 0
        start_logits = start_logits.masked_fill(p_mask, -1e9)
        end_logits = end_logits.masked_fill(p_mask, -1e9)

        return start_logits, end_logits
