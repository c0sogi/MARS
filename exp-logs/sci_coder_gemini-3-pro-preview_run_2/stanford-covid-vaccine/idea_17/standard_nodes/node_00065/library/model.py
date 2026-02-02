import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedDenseBlock(nn.Module):
    """
    A single dilated residual block utilizing dense connections.
    Follows the Pre-Activation design: BN -> ReLU -> Conv -> Dropout.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DilatedDenseBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.act = nn.ReLU()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.bn(x)
        out = self.act(out)
        out = self.conv(out)
        out = self.dropout(out)
        return out


class GatedFusionUnit(nn.Module):
    """
    Performs non-linear gated fusion of local and partner features.
    Mechanically similar to a GRU cell update step.

    f_fused = z * c + (1 - z) * f_local
    where:
      z = sigmoid(W_z * [f_local, f_partner])
      c = tanh(W_c * [f_local, f_partner])
    """

    def __init__(self, dim):
        super(GatedFusionUnit, self).__init__()
        # Input dim is 2 * dim (concatenation of local and partner)
        self.gate_conv = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.cand_conv = nn.Conv1d(dim * 2, dim, kernel_size=1)

    def forward(self, h_local, h_partner):
        # h_local, h_partner: (Batch, Channels, SeqLen)

        # Concatenate along channel dimension
        combined = torch.cat([h_local, h_partner], dim=1)

        # Compute gate z (0 to 1) and candidate c (-1 to 1)
        z = torch.sigmoid(self.gate_conv(combined))
        c = torch.tanh(self.cand_conv(combined))

        # Gated fusion
        # If z is 1, we fully trust the new candidate (interaction)
        # If z is 0, we keep the local feature (residual)
        h_fused = c * z + h_local * (1 - z)
        return h_fused


class GatedDenseNet(nn.Module):
    """
    Gated Dense-Context Hybrid Network.

    Architecture:
    1. Input Stem (Conv1d)
    2. Dense Dilated TCN Backbone (DenseNet-style connections)
    3. Feature Compression (1x1 Conv)
    4. Gated Structural Fusion (Gather + Gated Unit)
    5. Global Aggregation (BiGRU)
    6. Output Head (Linear)
    """

    def __init__(self):
        super(GatedDenseNet, self).__init__()

        # 1. Input Stem
        # Projects 18 input channels to the growth rate
        self.stem = nn.Conv1d(
            Config.INPUT_CHANNELS, Config.TCN_GROWTH_RATE, kernel_size=1
        )

        # 2. Dense Dilated Backbone
        self.dense_blocks = nn.ModuleList()
        current_dim = Config.TCN_GROWTH_RATE

        for dilation in Config.TCN_DILATIONS:
            block = DilatedDenseBlock(
                in_channels=current_dim,
                out_channels=Config.TCN_GROWTH_RATE,
                kernel_size=Config.TCN_KERNEL_SIZE,
                dilation=dilation,
                dropout=Config.DROPOUT,
            )
            self.dense_blocks.append(block)
            # In DenseNet, channels grow by the growth rate at each step
            current_dim += Config.TCN_GROWTH_RATE

        # 3. Compression for Fusion
        # Compress the accumulated dense features to a manageable latent dimension
        self.compressor = nn.Conv1d(current_dim, Config.LATENT_DIM, kernel_size=1)
        self.compressor_bn = nn.BatchNorm1d(Config.LATENT_DIM)
        self.compressor_act = nn.ReLU()

        # 4. Gated Structural Fusion
        self.fusion_unit = GatedFusionUnit(Config.LATENT_DIM)

        # 5. Global Aggregation (BiGRU)
        # Input: LATENT_DIM
        # Output: 2 * GRU_HIDDEN_DIM. Config ensures this equals LATENT_DIM
        self.gru = nn.GRU(
            input_size=Config.LATENT_DIM,
            hidden_size=Config.GRU_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 6. Output Head
        # Projects latent features to the 5 target variables
        # Input dim matches GRU output dim (LATENT_DIM)
        self.head = nn.Linear(Config.LATENT_DIM, 5)

    def forward(self, x, partner_indices):
        """
        Args:
            x: Input features (Batch, Channels=18, SeqLen=107)
            partner_indices: Indices of paired bases (Batch, SeqLen=107)
        """
        # --- 1. Stem ---
        out = self.stem(x)

        # --- 2. Dense Backbone ---
        # We maintain a list of features to concatenate (Dense Connectivity)
        features = [out]

        for block in self.dense_blocks:
            # Concatenate all previous features along channel dimension
            dense_input = torch.cat(features, dim=1)
            block_out = block(dense_input)
            features.append(block_out)

        # Final Dense Representation (Concatenate all)
        dense_out = torch.cat(features, dim=1)

        # --- 3. Compression ---
        latent = self.compressor(dense_out)
        latent = self.compressor_bn(latent)
        h_local = self.compressor_act(latent)  # Shape: (Batch, Latent, SeqLen)

        # --- 4. Gated Structural Fusion ---
        # We need to gather the latent features of the partner bases.
        # partner_indices shape: (Batch, SeqLen)
        # h_local shape: (Batch, Channels, SeqLen)

        batch_size, channels, seq_len = h_local.shape

        # Expand indices to match channel dimension for gather
        # (Batch, 1, SeqLen) -> (Batch, Channels, SeqLen)
        indices_expanded = partner_indices.unsqueeze(1).expand(-1, channels, -1)

        # Gather features: output[b, c, i] = input[b, c, indices_expanded[b, c, i]]
        h_partner = torch.gather(h_local, 2, indices_expanded)

        # Apply Gated Fusion
        h_fused = self.fusion_unit(h_local, h_partner)  # Shape: (Batch, Latent, SeqLen)

        # --- 5. Global Aggregation (BiGRU) ---
        # GRU expects (Batch, SeqLen, Input_Size)
        gru_input = h_fused.permute(0, 2, 1)

        # gru_out: (Batch, SeqLen, Num_Directions * Hidden_Size)
        gru_out, _ = self.gru(gru_input)

        # --- 6. Head ---
        # Linear layer applies to the last dimension
        logits = self.head(gru_out)  # Shape: (Batch, SeqLen, 5)

        return logits
