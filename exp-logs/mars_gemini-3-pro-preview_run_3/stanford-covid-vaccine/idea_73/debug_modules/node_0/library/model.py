import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Hyperparameters


class InteractionModule(nn.Module):
    """
    Full-Rank GLU-Decoupled Interaction Module.
    Injects structural information by gathering neighbor features, computing a
    GLU-modulated message, and gating the injection via a full-rank stabilized network.
    """

    def __init__(self, hidden_dim):
        super(InteractionModule, self).__init__()
        self.hidden_dim = hidden_dim

        # GLU Message components: (W_c * h_j + b_c) * sigmoid(W_g * h_j + b_g)
        # We use two separate linear layers.
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # Full-Rank Stabilized Gate components
        # Input: [h_i; h_j] -> 2 * hidden_dim
        # Projection: Wide dimension (Full Rank = hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate_out = nn.Linear(hidden_dim, hidden_dim)

        # Final Post-Normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adjacency):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Hidden_Dim)
            adjacency: Tensor of shape (Batch, Seq_Len) with indices of paired bases.
                       -1 indicates unpaired.
        """
        B, L, D = x.shape

        # 1. Gather Neighbor Features (h_j)
        # Handle -1 indices by replacing them with 0 temporarily, then masking the result.
        # mask is True where base is paired (adj != -1)
        mask = (adjacency != -1).unsqueeze(-1).float()  # (B, L, 1)

        # Clamp -1 to 0 for valid gathering
        safe_indices = adjacency.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices for gather: (B, L, D)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, D)

        # Gather: neighbor_x[b, i, :] = x[b, safe_indices[b, i], :]
        neighbor_x = torch.gather(x, 1, gather_indices)

        # Apply mask: Force h_j = 0 for unpaired bases
        neighbor_x = neighbor_x * mask

        # 2. GLU Message (Bias-Refined)
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # For unpaired (neighbor_x=0), this becomes b_c * sigmoid(b_g)
        msg_content = self.W_c(neighbor_x)
        msg_gate = torch.sigmoid(self.W_g(neighbor_x))
        message = msg_content * msg_gate

        # 3. Full-Rank Stabilized Gate
        # Input: Concat [h_i, h_j]
        gate_input = torch.cat([x, neighbor_x], dim=-1)  # (B, L, 2*D)

        # Wide Projection -> LayerNorm -> GELU -> Linear -> Sigmoid
        z_wide = self.gate_proj(gate_input)
        z_norm = self.gate_norm(z_wide)
        z_act = F.gelu(z_norm)
        logits = self.gate_out(z_act)
        gate = torch.sigmoid(logits)

        # 4. Injection & Post-Normalization
        # h_res = h_i + gate * message
        h_res = x + gate * message
        h_out = self.out_norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    High-Capacity Full-Rank GLU-Decoupled BiGRU.
    Consists of a Conv1D stem, a 4-layer BiGRU backbone with interleaved
    interaction modules, and a final prediction head.
    """

    def __init__(self):
        super(RNAModel, self).__init__()

        # ------------------------------------------------------------------
        # Hyperparameters
        # ------------------------------------------------------------------
        self.input_dim = Hyperparameters.INPUT_DIM
        self.hidden_dim_per_dir = Hyperparameters.HIDDEN_DIM
        self.hidden_dim = (
            self.hidden_dim_per_dir * 2
            if Hyperparameters.BIDIRECTIONAL
            else self.hidden_dim_per_dir
        )
        self.num_layers = Hyperparameters.NUM_LAYERS
        self.num_targets = Hyperparameters.NUM_TARGETS
        self.dropout_rate = Hyperparameters.DROPOUT

        # ------------------------------------------------------------------
        # 1. Convolutional Stem
        # ------------------------------------------------------------------
        self.stem_conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=Hyperparameters.STEM_FILTERS,
            kernel_size=Hyperparameters.STEM_KERNEL_SIZE,
            padding=Hyperparameters.STEM_KERNEL_SIZE // 2,
        )
        self.stem_act = nn.GELU()
        self.stem_dropout = nn.Dropout(self.dropout_rate)

        # Project stem output to RNN hidden dimension if they differ
        self.stem_proj = nn.Linear(Hyperparameters.STEM_FILTERS, self.hidden_dim)

        # ------------------------------------------------------------------
        # 2. Backbone (BiGRU + Interaction)
        # ------------------------------------------------------------------
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        for i in range(self.num_layers):
            # BiGRU Layer
            gru = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim_per_dir,
                batch_first=True,
                bidirectional=Hyperparameters.BIDIRECTIONAL,
            )
            self.gru_layers.append(gru)

            # Interaction Module (except for the final block)
            if i < self.num_layers - 1:
                self.interaction_layers.append(InteractionModule(self.hidden_dim))
            else:
                self.interaction_layers.append(None)

        self.dropout = nn.Dropout(self.dropout_rate)

        # ------------------------------------------------------------------
        # 3. Output Head
        # ------------------------------------------------------------------
        self.head = nn.Linear(self.hidden_dim, self.num_targets)

    def forward(self, inputs, adjacency):
        """
        Args:
            inputs: (Batch, Seq_Len, Input_Dim)
            adjacency: (Batch, Seq_Len)
        Returns:
            outputs: (Batch, Seq_Len, Num_Targets)
        """
        # ------------------------------------------------------------------
        # Stem
        # ------------------------------------------------------------------
        # Permute for Conv1d: (B, C, L)
        x = inputs.transpose(1, 2)
        x = self.stem_conv(x)
        x = self.stem_act(x)
        x = self.stem_dropout(x)
        # Permute back: (B, L, C)
        x = x.transpose(1, 2)

        # Project to hidden dim
        x = self.stem_proj(x)

        # ------------------------------------------------------------------
        # Backbone
        # ------------------------------------------------------------------
        for i in range(self.num_layers):
            # GRU
            # x is (B, L, H)
            gru_out, _ = self.gru_layers[i](x)

            # Residual connection around GRU (optional but often helpful,
            # here we simply pass the GRU output forward as the new state
            # or add residual if dimensions match perfectly and we want ResRNN.
            # Given the "High-Capacity" description usually implies deep stacking,
            # we use the GRU output directly as the updated representation).
            # However, standard deep RNNs just pass output to next layer.
            x = gru_out

            # Apply Dropout
            x = self.dropout(x)

            # Interaction (if applicable)
            if self.interaction_layers[i] is not None:
                x = self.interaction_layers[i](x, adjacency)

        # ------------------------------------------------------------------
        # Head
        # ------------------------------------------------------------------
        outputs = self.head(x)

        return outputs
