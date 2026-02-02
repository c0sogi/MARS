import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Multi-Task Recurrent Encoder.
    Backbone: Bi-Directional LSTM.
    Heads: Classification (21 classes) and Boundary (1 class).
    """

    def __init__(self):
        super(BiLSTMEncoder, self).__init__()

        self.input_dim = Config.INPUT_DIM
        self.hidden_size = Config.LSTM_HIDDEN_SIZE
        self.num_layers = Config.LSTM_NUM_LAYERS
        self.dropout = Config.LSTM_DROPOUT
        self.num_classes = Config.NUM_CLASSES

        # Bi-LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        # Projection Heads
        # Input to FC is hidden_size * 2 (bidirectional)
        self.fc_cls = nn.Linear(self.hidden_size * 2, self.num_classes)
        self.fc_bnd = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x):
        """
        Args:
            x: (B, T, InputDim)
        Returns:
            cls_logits: (B, T, NumClasses)
            bnd_logits: (B, T, 1)
        """
        # LSTM output: (B, T, Hidden*2)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)

        cls_logits = self.fc_cls(out)
        bnd_logits = self.fc_bnd(out)

        return cls_logits, bnd_logits


class DilatedResidualLayer(nn.Module):
    """
    Building block for the TCN.
    Dilated Conv1D -> ReLU -> Dropout -> Conv1D 1x1 -> Residual Add
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()

        # Padding calculation to maintain temporal dimension
        # For kernel_size=k, dilation=d, padding = (k-1)*d // 2 (assuming odd k)
        # Here we use padding='same' logic manually if needed, or rely on PyTorch padding
        # Config says kernel_size=3. padding = (3-1)*d / 2 = d.
        padding = dilation

        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        # Cite solution_lesson_node_00075: Instance Normalization improves TCN refinement
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv_dilated(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    """
    Stage 2 & 3: Boundary-Informed Refinement.
    Backbone: Stack of Dilated Residual Layers (MS-TCN style).
    """

    def __init__(self, input_dim):
        super(SingleStageTCN, self).__init__()

        self.num_classes = Config.NUM_CLASSES
        self.num_layers = len(Config.TCN_DILATIONS)
        self.num_channels = Config.TCN_NUM_CHANNELS[0]  # Assuming uniform channels
        self.kernel_size = Config.TCN_KERNEL_SIZE
        self.dropout = Config.TCN_DROPOUT
        self.dilations = Config.TCN_DILATIONS

        # Input projection
        self.conv_in = nn.Conv1d(input_dim, self.num_channels, 1)

        # Dilated Residual Stack
        layers = []
        for i in range(self.num_layers):
            layers.append(
                DilatedResidualLayer(
                    self.num_channels, self.kernel_size, self.dilations[i], self.dropout
                )
            )
        self.layers = nn.Sequential(*layers)

        # Output Heads
        self.conv_cls = nn.Conv1d(self.num_channels, self.num_classes, 1)
        self.conv_bnd = nn.Conv1d(self.num_channels, 1, 1)

    def forward(self, x):
        """
        Args:
            x: (B, T, InputDim) - Input probabilities from previous stage
        Returns:
            cls_logits: (B, T, NumClasses)
            bnd_logits: (B, T, 1)
        """
        # Permute for Conv1d: (B, C, T)
        out = x.permute(0, 2, 1)

        out = self.conv_in(out)
        out = self.layers(out)

        cls_logits = self.conv_cls(out)
        bnd_logits = self.conv_bnd(out)

        # Permute back: (B, T, C)
        cls_logits = cls_logits.permute(0, 2, 1)
        bnd_logits = bnd_logits.permute(0, 2, 1)

        return cls_logits, bnd_logits


class SBMD_CRCN(nn.Module):
    """
    Supervised Boundary-Aware Masked Dual-Stage Cascaded Recurrent-Convolutional Network.
    Connects Stage 1 (LSTM) -> Stage 2 (TCN) -> Stage 3 (TCN) with inter-stage masking.
    """

    def __init__(self):
        super(SBMD_CRCN, self).__init__()

        # Stage 1
        self.stage1 = BiLSTMEncoder()

        # Stage 2 Input Dim: NumClasses (Prob) + 1 (Boundary Prob)
        tcn_input_dim = Config.NUM_CLASSES + 1

        # Stage 2
        self.stage2 = SingleStageTCN(input_dim=tcn_input_dim)

        # Stage 3
        self.stage3 = SingleStageTCN(input_dim=tcn_input_dim)

    def forward(self, x, mask):
        """
        Args:
            x: (B, T, InputDim) - Raw features
            mask: (B, T) - Sequence mask (1 for valid, 0 for padding)
        Returns:
            dict containing logits for all stages/heads.
        """
        # Expand mask for multiplication: (B, T, 1)
        mask_expanded = mask.unsqueeze(-1)

        # --- Stage 1 ---
        s1_cls_logits, s1_bnd_logits = self.stage1(x)

        # Prepare input for Stage 2
        # Apply Softmax/Sigmoid to get probabilities
        s1_cls_probs = F.softmax(s1_cls_logits, dim=-1)
        s1_bnd_probs = torch.sigmoid(s1_bnd_logits)

        # Concatenate: (B, T, C+1)
        s1_out = torch.cat([s1_cls_probs, s1_bnd_probs], dim=-1)

        # Inter-Stage Masking
        s1_out_masked = s1_out * mask_expanded

        # --- Stage 2 ---
        s2_cls_logits, s2_bnd_logits = self.stage2(s1_out_masked)

        # Prepare input for Stage 3
        s2_cls_probs = F.softmax(s2_cls_logits, dim=-1)
        s2_bnd_probs = torch.sigmoid(s2_bnd_logits)

        s2_out = torch.cat([s2_cls_probs, s2_bnd_probs], dim=-1)

        # Inter-Stage Masking
        s2_out_masked = s2_out * mask_expanded

        # --- Stage 3 ---
        s3_cls_logits, s3_bnd_logits = self.stage3(s2_out_masked)

        return {
            "stage1_cls": s1_cls_logits,
            "stage1_bnd": s1_bnd_logits,
            "stage2_cls": s2_cls_logits,
            "stage2_bnd": s2_bnd_logits,
            "stage3_cls": s3_cls_logits,
            "stage3_bnd": s3_bnd_logits,
        }
