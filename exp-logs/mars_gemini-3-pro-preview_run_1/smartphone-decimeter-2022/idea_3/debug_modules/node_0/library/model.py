import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SatelliteEncoder(nn.Module):
    """
    Processes features for each satellite independently using an MLP.
    Shared weights across all satellites and time steps.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SatelliteEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Max_Sats, Feat_Dim)
        return self.net(x)


class DeepSetLayer(nn.Module):
    """
    Aggregates satellite embeddings into a single epoch embedding.
    Uses permutation-invariant Max-Pooling.
    """

    def __init__(self):
        super(DeepSetLayer, self).__init__()

    def forward(self, sat_embeddings, mask):
        """
        Args:
            sat_embeddings: (Batch, Seq_Len, Max_Sats, Embed_Dim)
            mask: (Batch, Seq_Len, Max_Sats) - 1 for valid, 0 for padding
        Returns:
            epoch_embeddings: (Batch, Seq_Len, Embed_Dim)
        """
        # Expand mask for broadcasting: (Batch, Seq_Len, Max_Sats, 1)
        mask_expanded = mask.unsqueeze(-1)

        # Mask out invalid satellites by setting them to a very small number before max pooling
        # We use -1e9 instead of -inf to avoid NaNs in gradients if all are masked (though unlikely)
        masked_embeddings = sat_embeddings.masked_fill(mask_expanded == 0, -1e9)

        # Max pooling over the satellite dimension (dim=2)
        # values: (Batch, Seq_Len, Embed_Dim)
        epoch_embeddings, _ = torch.max(masked_embeddings, dim=2)

        # Handle case where all satellites are masked (though data loader should prevent this)
        # If mask sum is 0, result is -1e9, we can zero it out or leave it.
        # Ideally, valid data always has at least one satellite.

        return epoch_embeddings


class TemporalBlock(nn.Module):
    """
    A single residual block for the TCN backbone.
    Uses 1D Convolutions with padding to maintain sequence length.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # For 'same' padding with dilation
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.bn2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, Channels, Seq_Len)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalBackbone(nn.Module):
    """
    TCN Backbone consisting of stacked TemporalBlocks.
    """

    def __init__(
        self, input_dim, num_channels, kernel_size=3, num_layers=4, dropout=0.2
    ):
        super(TemporalBackbone, self).__init__()
        layers = []
        num_levels = num_layers

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_dim if i == 0 else num_channels
            out_channels = num_channels

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Input: (Batch, Seq_Len, Features)
        # Conv1d expects (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1)
        y = self.network(x)
        # Permute back to (Batch, Seq_Len, Channels)
        y = y.permute(0, 2, 1)
        return y


class GnssDeepSetTCN(nn.Module):
    """
    Hierarchical Deep-Set TCN for GNSS location prediction.

    Architecture:
    1. Satellite Encoder (MLP): Process each satellite -> Embedding
    2. Deep Set Layer: Aggregate satellite embeddings -> Epoch Embedding
    3. Global Context: Concatenate global features (e.g., Altitude)
    4. Temporal Backbone (TCN): Process sequence of epochs -> Temporal Features
    5. Output Head: Predict residuals (Delta Lat, Delta Lon)
    """

    def __init__(self):
        super(GnssDeepSetTCN, self).__init__()

        # 1. Satellite Feature Encoder
        # Input dim is determined by the number of satellite features defined in Config
        # We define a dummy dataset to get the feature names count
        # Cn0, El, SinAz, CosAz, PrUnc (5) + Constellation OneHot (7) = 12
        sat_input_dim = 12

        self.sat_encoder = SatelliteEncoder(
            input_dim=sat_input_dim,
            hidden_dim=Config.SAT_HIDDEN_DIM,
            output_dim=Config.SAT_EMBEDDING_DIM,
        )

        # 2. Deep Set Aggregation
        self.deep_set_layer = DeepSetLayer()

        # 3. Temporal Backbone
        # Input to TCN = Sat Embedding + Global Features
        tcn_input_dim = Config.SAT_EMBEDDING_DIM + len(Config.GLOBAL_FEATURES)

        self.tcn = TemporalBackbone(
            input_dim=tcn_input_dim,
            num_channels=Config.TCN_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            num_layers=Config.TCN_LAYERS,
            dropout=Config.TCN_DROPOUT,
        )

        # 4. Output Head
        self.head = nn.Sequential(
            nn.Linear(Config.TCN_CHANNELS, 32),
            nn.ReLU(),
            nn.Linear(32, Config.OUTPUT_DIM),  # Predicts (dLat, dLon)
        )

    def forward(self, sat_features, global_features, masks):
        """
        Args:
            sat_features: (Batch, Seq_Len, Max_Sats, Sat_Feat_Dim)
            global_features: (Batch, Seq_Len, Global_Feat_Dim)
            masks: (Batch, Seq_Len, Max_Sats)

        Returns:
            residuals: (Batch, Seq_Len, 2)
        """
        # 1. Encode Satellites
        # (B, L, N, F) -> (B, L, N, Emb)
        sat_emb = self.sat_encoder(sat_features)

        # 2. Aggregate Satellites (Deep Set)
        # (B, L, N, Emb) -> (B, L, Emb)
        epoch_emb = self.deep_set_layer(sat_emb, masks)

        # 3. Fuse Global Context
        # (B, L, Emb) cat (B, L, Glob) -> (B, L, Emb + Glob)
        combined_emb = torch.cat([epoch_emb, global_features], dim=2)

        # 4. Temporal Processing
        # (B, L, Feats) -> (B, L, Channels)
        temporal_feats = self.tcn(combined_emb)

        # 5. Prediction
        # (B, L, Channels) -> (B, L, 2)
        residuals = self.head(temporal_feats)

        return residuals
