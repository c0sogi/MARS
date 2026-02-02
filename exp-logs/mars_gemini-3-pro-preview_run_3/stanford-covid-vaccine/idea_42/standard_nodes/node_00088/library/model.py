import torch
import torch.nn as nn
from library.config import Config


class DecoupledStructuralInteraction(nn.Module):
    """
    Implements the Decoupled Structural Interaction Module with Bias-Driven Loop Refinement.

    Logic:
    1. Gather neighbor states h_j based on adjacency.
    2. Zero-Masking: Unpaired bases gather a zero vector.
    3. Message: m_ij = GELU(W_msg * h_j + b_msg).
       - For unpaired bases, h_j=0, so m_ij = GELU(b_msg) (Learned Bias).
    4. Gate: g_ij = Sigmoid(W_gate * [h_i; h_j]).
    5. Update: h_res = h_i + g_ij * m_ij.
    6. Post-Norm: LayerNorm(h_res).
    """

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.dim = dim

        # Message projection: Projects neighbor state to message
        # Input: dim (h_j), Output: dim
        self.msg_proj = nn.Linear(dim, dim)

        # Gate projection: Computes gate from joint context
        # Input: 2*dim ([h_i; h_j]), Output: dim
        self.gate_proj = nn.Linear(dim * 2, dim)

        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        Args:
            h (torch.Tensor): Hidden states (Batch, Seq, Dim).
            adj (torch.Tensor): Adjacency indices (Batch, Seq).
                                Values are indices of paired bases, or -1 if unpaired.
        Returns:
            torch.Tensor: Updated hidden states (Batch, Seq, Dim).
        """
        B, L, D = h.shape

        # --- 1. Zero-Masking & Gathering ---
        # Create a zero vector for unpaired positions to gather from.
        # Shape: (Batch, 1, Dim)
        zeros = torch.zeros(B, 1, D, device=h.device, dtype=h.dtype)

        # Append zeros to the end of the sequence dimension -> (B, L+1, D)
        h_padded = torch.cat([h, zeros], dim=1)

        # Adjust adjacency indices: Map -1 (unpaired) to L (index of zero vector)
        gather_idx = adj.clone()
        gather_idx[gather_idx == -1] = L

        # Expand indices for gathering across the feature dimension
        # Shape: (B, L, D)
        gather_idx_expanded = gather_idx.unsqueeze(-1).expand(-1, -1, D)

        # Gather neighbor states h_j
        # If paired, gets h[pair_idx]. If unpaired, gets zeros.
        h_j = torch.gather(h_padded, 1, gather_idx_expanded)

        # --- 2. Decoupled Message ---
        # m_ij = GELU(W * h_j + b)
        # For unpaired bases, h_j is 0, so this becomes GELU(bias).
        m_ij = self.act(self.msg_proj(h_j))

        # --- 3. Channel-Wise Gating ---
        # g_ij = Sigmoid(W * [h_i; h_j])
        # Concatenate current state and neighbor state
        cat_feat = torch.cat([h, h_j], dim=-1)
        g_ij = self.sigmoid(self.gate_proj(cat_feat))

        # --- 4. Residual Injection ---
        # h_res = h_i + g_ij * m_ij
        update = self.dropout(g_ij * m_ij)
        h_res = h + update

        # --- 5. Post-Normalization ---
        h_out = self.norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    Deep Bias-Refined Decoupled BiGRU Architecture.

    Structure:
    1. Convolutional Stem (1D Conv + GELU)
    2. Deep Backbone (4 Layers)
       - Layers 1-3: BiGRU -> Interaction -> LayerNorm
       - Layer 4: BiGRU
    3. Output Head (Linear)
    """

    def __init__(self):
        super().__init__()

        # --- 1. Convolutional Stem ---
        # Projects one-hot inputs (14 channels) to dense embedding (256 channels)
        # Aggregates local k-mers (k=3)
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.INPUT_CHANNELS,
                out_channels=Config.STEM_FILTERS,
                kernel_size=Config.STEM_KERNEL_SIZE,
                padding=Config.STEM_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
        )

        # --- 2. Deep Backbone ---
        self.layers = nn.ModuleList()
        self.interactions = nn.ModuleList()

        # We want the output of the BiGRU to match HIDDEN_DIM (384).
        # Since it's bidirectional, hidden_size per direction is HIDDEN_DIM // 2.
        rnn_hidden_size = Config.HIDDEN_DIM // 2

        for i in range(Config.NUM_LAYERS):
            # Determine input dimension for the GRU
            # Layer 0 takes stem output, others take previous layer output
            input_dim = Config.STEM_FILTERS if i == 0 else Config.HIDDEN_DIM

            # Bidirectional GRU
            gru = nn.GRU(
                input_size=input_dim,
                hidden_size=rnn_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.layers.append(gru)

            # Structural Interaction Module
            # Applied after every block EXCEPT the final one
            if i < Config.NUM_LAYERS - 1:
                inter = DecoupledStructuralInteraction(
                    dim=Config.HIDDEN_DIM, dropout=Config.DROPOUT
                )
                self.interactions.append(inter)
            else:
                self.interactions.append(None)

        # --- 3. Output Head ---
        # Projects final hidden state to 5 target values
        self.head = nn.Linear(Config.HIDDEN_DIM, 5)

    def forward(self, inputs, adj):
        """
        Args:
            inputs (torch.Tensor): (Batch, Seq, 14)
            adj (torch.Tensor): (Batch, Seq)

        Returns:
            torch.Tensor: Predictions (Batch, Seq, 5)
        """
        # --- Stem ---
        # Conv1d expects (Batch, Channels, Seq)
        x = inputs.transpose(1, 2)
        x = self.stem(x)
        # Transpose back to (Batch, Seq, Channels)
        x = x.transpose(1, 2)

        # --- Backbone ---
        for i in range(Config.NUM_LAYERS):
            # 1. BiGRU
            # x shape: (Batch, Seq, Input_Dim)
            # out shape: (Batch, Seq, 2*rnn_hidden) = (Batch, Seq, HIDDEN_DIM)
            x, _ = self.layers[i](x)

            # 2. Structural Interaction (if applicable)
            if self.interactions[i] is not None:
                x = self.interactions[i](x, adj)

        # --- Head ---
        out = self.head(x)

        return out
