import torch
import torch.nn as nn
from library import config


class MultiScaleConv1d(nn.Module):
    """
    Inception-style Multi-Scale 1D Convolutional Block.
    Applies convolutions with different kernel sizes in parallel to capture
    features at different temporal resolutions (noise vs. trends).
    """

    def __init__(self, in_channels, out_channels, kernels, dropout=0.0):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k, padding="same"),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for k in kernels
            ]
        )

    def forward(self, x):
        # x shape: (Batch, In_Channels, Seq_Len)
        # Apply each branch
        branch_outputs = [branch(x) for branch in self.branches]
        # Concatenate along the channel dimension
        # Output shape: (Batch, Out_Channels * Num_Kernels, Seq_Len)
        return torch.cat(branch_outputs, dim=1)


class PhysicsResidualModel(nn.Module):
    """
    Learnable Physics-Residual Multi-Scale CNN-LSTM.

    This architecture decouples the learning of linear physical laws from
    complex non-linear residuals.

    Branch 1: Deep Temporal Residual (CNN-LSTM)
        - Inputs: Continuous control signals + Learnable Embeddings for R/C.
        - Stem: Multi-Scale Conv1d to extract local temporal features.
        - Backbone: Bidirectional LSTM to model long-term dependencies.
        - Output: A residual pressure correction.

    Branch 2: Linear Physics Adapter
        - Inputs: Physics-derived features (Flow Interaction, Volume Interaction).
        - Layer: A simple Linear layer without activation.
        - Output: A baseline pressure estimate based on the Equation of Motion.
        - Purpose: Explicitly learns the scaling coefficients (Resistance, Elastance)
          to convert proxy units into pressure units.

    Fusion:
        - Final Pressure = Residual + Baseline
    """

    def __init__(self):
        super().__init__()

        # --- Embeddings for Lung Attributes ---
        # R and C are categorical settings (3 unique values each).
        # We learn a dense vector representation for them.
        self.r_embedding = nn.Embedding(3, config.EMBEDDING_DIM)
        self.c_embedding = nn.Embedding(3, config.EMBEDDING_DIM)

        # --- Deep Branch: Stem (Multi-Scale CNN) ---
        # Input dimension = Continuous Features + R_Embedding + C_Embedding
        cnn_in_dim = len(config.CONTINUOUS_FEATURES) + (2 * config.EMBEDDING_DIM)

        self.cnn_stem = MultiScaleConv1d(
            in_channels=cnn_in_dim,
            out_channels=config.CNN_FILTERS,
            kernels=config.CNN_KERNELS,
            dropout=config.CNN_DROPOUT,
        )

        # --- Deep Branch: Backbone (LSTM) ---
        # Input to LSTM is the concatenated output of CNN branches
        lstm_in_dim = config.CNN_FILTERS * len(config.CNN_KERNELS)

        self.lstm_backbone = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            dropout=config.LSTM_DROPOUT if config.LSTM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL,
            batch_first=True,
        )

        # --- Deep Branch: Projection Head ---
        lstm_out_dim = config.LSTM_HIDDEN_SIZE * (2 if config.BIDIRECTIONAL else 1)
        self.residual_head = nn.Linear(lstm_out_dim, 1)

        # --- Physics Branch: Linear Adapter ---
        # Takes specific physics features and learns a linear combination
        self.physics_adapter = nn.Linear(len(config.PHYSICS_FEATURES), 1, bias=True)

        # Initialize physics adapter with positive weights intuitively (optional, but helps convergence)
        # R (Flow) and 1/C (Volume) contributions to pressure are generally positive.
        with torch.no_grad():
            self.physics_adapter.weight.fill_(0.1)
            self.physics_adapter.bias.zero_()

    def forward(self, x_cont, x_cat, x_phys, **kwargs):
        """
        Args:
            x_cont (Tensor): Continuous features (Batch, Seq, Feat_Cont)
            x_cat (Tensor): Categorical indices (Batch, Seq, 2) -> [R_idx, C_idx]
            x_phys (Tensor): Physics features (Batch, Seq, Feat_Phys)
            **kwargs: Ignored (handles extra args like u_out, ids)

        Returns:
            Tensor: Predicted pressure (Batch, Seq)
        """
        batch_size, seq_len, _ = x_cont.shape

        # 1. Embed Categorical Features
        # x_cat[:, :, 0] is R_idx, x_cat[:, :, 1] is C_idx
        r_emb = self.r_embedding(x_cat[:, :, 0])  # (Batch, Seq, Emb_Dim)
        c_emb = self.c_embedding(x_cat[:, :, 1])  # (Batch, Seq, Emb_Dim)

        # 2. Prepare Input for Deep Branch
        # Concatenate continuous features with embeddings
        deep_input = torch.cat([x_cont, r_emb, c_emb], dim=-1)  # (Batch, Seq, C_in)

        # 3. CNN Stem
        # Permute to (Batch, Channels, Seq) for Conv1d
        deep_input = deep_input.permute(0, 2, 1)
        cnn_out = self.cnn_stem(deep_input)  # (Batch, C_out, Seq)

        # 4. LSTM Backbone
        # Permute back to (Batch, Seq, Channels) for LSTM
        lstm_input = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm_backbone(lstm_input)  # (Batch, Seq, Hidden*Dirs)

        # 5. Calculate Residual Pressure
        pressure_residual = self.residual_head(lstm_out)  # (Batch, Seq, 1)

        # 6. Calculate Baseline Pressure (Physics Branch)
        # x_phys contains [flow_interaction, volume_interaction]
        pressure_baseline = self.physics_adapter(x_phys)  # (Batch, Seq, 1)

        # 7. Fusion
        # Element-wise sum of the learned residual and the physics baseline
        output = pressure_residual + pressure_baseline

        # Remove last dimension to return (Batch, Seq)
        return output.squeeze(-1)
