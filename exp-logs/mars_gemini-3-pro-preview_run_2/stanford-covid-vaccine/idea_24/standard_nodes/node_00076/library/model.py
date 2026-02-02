import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseDilatedBlock(nn.Module):
    """
    A single dilated convolution block used in the Dense-Context Backbone.
    It receives the concatenated history of features and outputs a fixed number
    of new features (growth_rate), which are then added to the history.
    """

    def __init__(self, in_channels, out_channels, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=Config.KERNEL_SIZE,
            padding=dilation,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class NonLinearBottleneck(nn.Module):
    """
    Deep Non-Linear Bottleneck to compress the high-dimensional dense history
    into a semantic latent representation.
    Structure: Conv1d(In -> 128) -> BN -> GELU -> Dropout -> Conv1d(128 -> Latent)
    """

    def __init__(self, in_channels, latent_dim, dropout):
        super().__init__()
        inter_dim = 128
        self.project_1 = nn.Conv1d(in_channels, inter_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(inter_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.project_2 = nn.Conv1d(inter_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        x = self.project_1(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.project_2(x)
        return x


class StructuralInteraction(nn.Module):
    """
    Symmetric Structural Interaction Layer.
    Gathers latent features from the paired base (partner) and concatenates them
    with the local features. Unpaired bases are handled via zero-masking to
    ensure orthogonal representation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, partner_indices):
        """
        Args:
            x: Latent features of shape (Batch, Channels, Length)
            partner_indices: Tensor of shape (Batch, Length) containing indices of partners or -1.
        """
        B, C, L = x.shape

        # 1. Create Mask for Unpaired Bases
        # partner_indices is -1 for unpaired bases.
        mask = (partner_indices != -1).unsqueeze(1)  # Shape: (B, 1, L)

        # 2. Prepare Indices for Gather
        # Replace -1 with 0 temporarily to prevent index out of bounds errors during gather.
        # The values gathered from index 0 for unpaired bases will be masked out subsequently.
        safe_indices = partner_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices to match channel dimension for gathering
        # We gather along dimension 2 (Length)
        gather_indices = safe_indices.unsqueeze(1).expand(-1, C, -1)  # Shape: (B, C, L)

        # 3. Gather Partner Features
        partner_features = torch.gather(x, 2, gather_indices)

        # 4. Apply Zero-Mask
        partner_features = partner_features * mask.float()

        # 5. Concatenate Local and Partner Features
        # Output channels = 2 * input_channels
        out = torch.cat([x, partner_features], dim=1)

        return out


class BiGRUHead(nn.Module):
    """
    Bidirectional GRU for Global Aggregation.
    Captures long-range dependencies and global constraints.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (Batch, Channels, Length) -> Permute to (Batch, Length, Channels) for GRU
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        # out: (Batch, Length, 2*Hidden) -> Permute back to (Batch, Channels, Length)
        out = out.permute(0, 2, 1)
        return out


class RNA_Model(nn.Module):
    """
    Regularized Non-Linear Dense-Context Network.

    Architecture:
    1. Input Embedding (Partner-Aware)
    2. Dense Dilated TCN Backbone (Multi-scale history)
    3. Deep Non-Linear Bottleneck (Latent Projection)
    4. Symmetric Structural Interaction (Partner Fusion)
    5. BiGRU Head (Global Context)
    6. Linear Classifier
    """

    def __init__(self):
        super().__init__()

        # 1. Input Projection
        self.embedding = nn.Conv1d(
            Config.INPUT_CHANNELS, Config.GROWTH_RATE, kernel_size=1
        )

        # 2. Dense Dilated Backbone
        self.blocks = nn.ModuleList()
        current_channels = Config.GROWTH_RATE

        for d in Config.DILATIONS:
            block = DenseDilatedBlock(
                in_channels=current_channels,
                out_channels=Config.GROWTH_RATE,
                dilation=d,
                dropout=Config.DROPOUT,
            )
            self.blocks.append(block)
            # In a DenseNet, input size for the next block increases by the growth rate
            current_channels += Config.GROWTH_RATE

        # 3. Deep Non-Linear Bottleneck
        # Projects the full concatenated history to the latent dimension
        self.neck = NonLinearBottleneck(
            in_channels=current_channels,
            latent_dim=Config.LATENT_DIM,
            dropout=Config.DROPOUT,
        )

        # 4. Structural Interaction
        self.interaction = StructuralInteraction()

        # 5. Global Aggregation (BiGRU)
        # Input is Local (64) + Partner (64) = 128
        gru_input_dim = 2 * Config.LATENT_DIM
        # Hidden size is set to input // 2 so that bidirectional output matches input dim
        gru_hidden_dim = gru_input_dim // 2
        self.gru = BiGRUHead(gru_input_dim, gru_hidden_dim)

        # 6. Output Head
        # Projects to the 5 target variables
        self.classifier = nn.Linear(gru_input_dim, Config.NUM_TARGETS)

    def forward(self, inputs, partner_indices):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input features of shape (Batch, Length, Channels).
            partner_indices (torch.Tensor): Partner indices of shape (Batch, Length).

        Returns:
            torch.Tensor: Predictions of shape (Batch, Length, Num_Targets).
        """
        # Permute inputs to (Batch, Channels, Length) for Conv1d operations
        x = inputs.permute(0, 2, 1)

        # Initial Embedding
        x = self.embedding(x)

        # Dense Backbone Processing
        # We maintain a list of feature maps to implement dense connections
        features = [x]

        for block in self.blocks:
            # Concatenate all prior feature maps
            in_tensor = torch.cat(features, dim=1)
            # Compute new features
            out = block(in_tensor)
            # Append new features to history
            features.append(out)

        # Concatenate full history for the neck
        total_history = torch.cat(features, dim=1)

        # Non-Linear Compression to Latent Space
        latent = self.neck(total_history)

        # Symmetric Structural Interaction
        interacted = self.interaction(latent, partner_indices)

        # Global Context Aggregation
        context = self.gru(interacted)

        # Final Classification
        # Permute back to (Batch, Length, Channels) for Linear layer
        context = context.permute(0, 2, 1)
        logits = self.classifier(context)

        return logits
