import torch
import torch.nn as nn
import math
from library.config import Config


class LayerNormLSTMCell(nn.Module):
    """
    A specific LSTM cell implementation that applies Layer Normalization
    to the internal gate activations (input, forget, cell, output)
    before non-linearities.

    This is critical for stabilizing the 'Width 512' regime as per the
    Internally-Normalized Wide-Stream Residual BiLSTM strategy.
    """

    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Linear projections for input-to-hidden and hidden-to-hidden
        # We project to 4 * hidden_size for (i, f, g, o) gates
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

        # Internal Layer Normalization
        # Applied to the sum of projections before activation
        self.ln = nn.LayerNorm(4 * hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        # Standard LSTM initialization
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_tensor, hidden_state):
        """
        Args:
            input_tensor: (Batch, Input_Size)
            hidden_state: Tuple (h_prev, c_prev), each (Batch, Hidden_Size)

        Returns:
            h_next, c_next: (Batch, Hidden_Size)
        """
        h_prev, c_prev = hidden_state

        # Compute gates
        # gates = LN(W_ih * x + W_hh * h_prev)
        gates = self.weight_ih(input_tensor) + self.weight_hh(h_prev)
        gates = self.ln(gates)

        # Split into input, forget, cell, output gates
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        # Apply non-linearities
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        # Update cell state and hidden state
        c_next = (f_gate * c_prev) + (i_gate * g_gate)
        h_next = o_gate * torch.tanh(c_next)

        return h_next, c_next


class LayerNormBiLSTM(nn.Module):
    """
    A wrapper module that constructs a bidirectional layer using
    LayerNormLSTMCell. It handles the forward and backward passes
    over the sequence and concatenates the results.
    """

    def __init__(self, input_size, hidden_size):
        super(LayerNormBiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forward and Backward cells
        self.cell_fwd = LayerNormLSTMCell(input_size, hidden_size)
        self.cell_bwd = LayerNormLSTMCell(input_size, hidden_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Input_Size)

        Returns:
            output: (Batch, Seq_Len, 2 * Hidden_Size)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # Initialize hidden and cell states
        h_fwd = torch.zeros(batch_size, self.hidden_size, device=device)
        c_fwd = torch.zeros(batch_size, self.hidden_size, device=device)

        h_bwd = torch.zeros(batch_size, self.hidden_size, device=device)
        c_bwd = torch.zeros(batch_size, self.hidden_size, device=device)

        # Storage for outputs
        outputs_fwd = []
        outputs_bwd = []

        # Forward pass
        for t in range(seq_len):
            h_fwd, c_fwd = self.cell_fwd(x[:, t, :], (h_fwd, c_fwd))
            outputs_fwd.append(h_fwd)

        # Backward pass
        for t in range(seq_len - 1, -1, -1):
            h_bwd, c_bwd = self.cell_bwd(x[:, t, :], (h_bwd, c_bwd))
            outputs_bwd.append(h_bwd)

        # Reverse backward outputs to match sequence order
        outputs_bwd = outputs_bwd[::-1]

        # Stack along sequence dimension
        outputs_fwd = torch.stack(outputs_fwd, dim=1)  # (Batch, Seq, Hidden)
        outputs_bwd = torch.stack(outputs_bwd, dim=1)  # (Batch, Seq, Hidden)

        # Concatenate
        output = torch.cat([outputs_fwd, outputs_bwd], dim=2)  # (Batch, Seq, 2*Hidden)

        return output


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Implements Fixed Sinusoidal Encodings for pairing distances.
    Preserves the sign to distinguish upstream/downstream dependencies.
    """

    def __init__(self, embedding_dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        # Precompute the division term for sinusoidal calculation
        # div_term = 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * -(math.log(10000.0) / embedding_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, positions):
        """
        Args:
            positions: Tensor of signed distances/positions.
                       Shape: (Batch, Seq_Len) or (Batch, Seq_Len, 1)

        Returns:
            embeddings: (Batch, Seq_Len, Embedding_Dim)
        """
        # Ensure input is float and has correct shape
        if positions.dim() == 3 and positions.size(-1) == 1:
            positions = positions.squeeze(-1)

        positions = positions.float()

        # Create output tensor
        # Shape: (Batch, Seq_Len, Embedding_Dim)
        pe = torch.zeros(
            positions.size(0),
            positions.size(1),
            self.embedding_dim,
            device=positions.device,
        )

        # Calculate sine and cosine
        # We use broadcasting: positions (B, S, 1) * div_term (1, 1, D/2)
        scaled_pos = positions.unsqueeze(-1) * self.div_term

        pe[..., 0::2] = torch.sin(scaled_pos)
        pe[..., 1::2] = torch.cos(scaled_pos)

        return pe
