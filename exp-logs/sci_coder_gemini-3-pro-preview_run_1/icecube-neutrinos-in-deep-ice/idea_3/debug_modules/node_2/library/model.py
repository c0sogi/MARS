import torch
import torch.nn as nn
from library.config import Config


class SequenceBranch(nn.Module):
    """
    Processes the event sequence using a Bidirectional GRU.
    Input: (Batch, Seq_Len, N_Features)
    Output: Final hidden state (Batch, Hidden_Dim * Directions)
    """

    def __init__(self):
        super(SequenceBranch, self).__init__()

        self.input_dim = Config.N_SEQ_FEATURES
        self.hidden_dim = Config.GRU_HIDDEN_DIM
        self.num_layers = Config.GRU_NUM_LAYERS
        self.bidirectional = Config.GRU_BIDIRECTIONAL
        self.dropout_rate = Config.GRU_DROPOUT

        # Dropout is only applied between layers, so it's 0 if num_layers=1
        gru_dropout = self.dropout_rate if self.num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=gru_dropout,
        )

        self.output_dim = self.hidden_dim * (2 if self.bidirectional else 1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)

        # GRU returns: output, h_n
        # output: (Batch, Seq_Len, Num_Dirs * Hidden_Dim)
        # h_n: (Num_Layers * Num_Dirs, Batch, Hidden_Dim)
        _, h_n = self.gru(x)

        # We want the final hidden state representing the whole sequence.
        # For bidirectional GRU, h_n contains the final states for forward and backward passes.
        # Structure of h_n: (layer_1_dir_1, layer_1_dir_2, layer_2_dir_1, layer_2_dir_2, ...)

        if self.bidirectional:
            # Extract the last layer's hidden states
            # The last two elements of the first dimension correspond to the last layer
            # h_n[-2] is forward direction of last layer
            # h_n[-1] is backward direction of last layer
            final_state = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            # Extract the last layer's hidden state
            final_state = h_n[-1]

        return final_state


class FeatureBranch(nn.Module):
    """
    Processes the engineered features using an MLP with residual connections.
    Input: (Batch, N_Manual_Features)
    Output: (Batch, MLP_Hidden_Dim)
    """

    def __init__(self):
        super(FeatureBranch, self).__init__()

        self.input_dim = Config.N_MANUAL_FEATURES
        self.hidden_dim = Config.MLP_HIDDEN_DIM

        # Initial projection
        self.fc_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn_in = nn.BatchNorm1d(self.hidden_dim)
        self.act = nn.ReLU()

        # Residual Block 1
        self.res1_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.res1_bn = nn.BatchNorm1d(self.hidden_dim)

        # Residual Block 2
        self.res2_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.res2_bn = nn.BatchNorm1d(self.hidden_dim)

        self.output_dim = self.hidden_dim

    def forward(self, x):
        # Initial projection
        out = self.fc_in(x)
        out = self.bn_in(out)
        out = self.act(out)

        # Residual Block 1
        identity = out
        out = self.res1_fc(out)
        out = self.res1_bn(out)
        out = self.act(out)
        out = out + identity  # Add residual

        # Residual Block 2
        identity = out
        out = self.res2_fc(out)
        out = self.res2_bn(out)
        out = self.act(out)
        out = out + identity  # Add residual

        return out


class FusionHead(nn.Module):
    """
    Fuses the outputs of the sequence and feature branches and predicts the 3D vector.
    """

    def __init__(self, input_dim):
        super(FusionHead, self).__init__()

        self.hidden_dim = Config.FUSION_HIDDEN_DIM
        self.dropout_rate = Config.DROPOUT_RATE

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)

        # Output is a 3D vector (x, y, z)
        self.fc_out = nn.Linear(self.hidden_dim, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class HybridRecurrentDenseNet(nn.Module):
    """
    Main model architecture combining GRU sequence processing and MLP feature processing.
    """

    def __init__(self):
        super(HybridRecurrentDenseNet, self).__init__()

        self.seq_branch = SequenceBranch()
        self.feat_branch = FeatureBranch()

        fusion_input_dim = self.seq_branch.output_dim + self.feat_branch.output_dim
        self.head = FusionHead(fusion_input_dim)

    def forward(self, seq, features):
        """
        Args:
            seq: Tensor of shape (Batch, Seq_Len, N_Seq_Features)
            features: Tensor of shape (Batch, N_Manual_Features)
        Returns:
            Tensor of shape (Batch, 3) representing the direction vector (x, y, z)
        """
        seq_out = self.seq_branch(seq)
        feat_out = self.feat_branch(features)

        # Concatenate representations
        concat = torch.cat([seq_out, feat_out], dim=1)

        # Predict direction
        output = self.head(concat)

        return output
