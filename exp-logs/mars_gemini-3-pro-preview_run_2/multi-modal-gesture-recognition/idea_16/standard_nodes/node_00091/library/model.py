import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualLayer(nn.Module):
    """
    Standard Dilated Residual Layer for MS-TCN.
    Conv1d -> InstanceNorm -> ReLU -> Dropout -> Conv1x1 -> Residual
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size, padding=self.padding, dilation=dilation
        )
        # InstanceNorm is critical for parameter efficiency (Cite solution_lesson_node_00063)
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        out = self.conv_dilated(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv_1x1(out)
        return x + out


class BiLSTMLatentEncoder(nn.Module):
    """
    Stage 1: Recurrent Encoder.
    Backbone: Bi-LSTM
    Heads: Class Probabilities (Softmax) only.
    Removed latent transition head to avoid noise injection (Cite solution_lesson_node_00073).
    """

    def __init__(self):
        super(BiLSTMLatentEncoder, self).__init__()

        self.input_dim = Config.INPUT_DIM
        self.hidden_size = Config.LSTM_HIDDEN_SIZE
        self.num_layers = Config.LSTM_NUM_LAYERS
        self.num_classes = Config.NUM_CLASSES

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Bidirectional output size is hidden_size * 2
        lstm_out_dim = self.hidden_size * 2

        # Head 1: Classification
        self.cls_head = nn.Linear(lstm_out_dim, self.num_classes)

    def forward(self, x, mask=None):
        lstm_out, _ = self.lstm(x)
        cls_logits = self.cls_head(lstm_out)  # (B, T, C)
        cls_probs = F.softmax(cls_logits, dim=2)
        return cls_probs


class MSTCN(nn.Module):
    """
    Multi-Stage Temporal Convolutional Network.
    Used for Stage 2 (Refinement) and Stage 3 (Sharpening).
    Standard architecture with InstanceNorm.
    """

    def __init__(self, in_channels, out_channels):
        super(MSTCN, self).__init__()

        self.hidden_channels = Config.TCN_NUM_CHANNELS
        self.num_layers = Config.TCN_NUM_LAYERS
        self.kernel_size = Config.TCN_KERNEL_SIZE
        self.dropout = Config.DROPOUT

        # Input Projection
        self.conv_in = nn.Conv1d(in_channels, self.hidden_channels, 1)

        # Stack of Dilated Residual Layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = 2**i
            self.layers.append(
                DilatedResidualLayer(
                    self.hidden_channels, self.kernel_size, dilation, self.dropout
                )
            )

        # Output Projection
        self.conv_out = nn.Conv1d(self.hidden_channels, out_channels, 1)

    def forward(self, x, mask=None):
        # x: (Batch, in_channels, Time)
        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)
        out = F.softmax(out, dim=1)
        return out


class GLT_CRCN(nn.Module):
    """
    Cascaded Recurrent-Convolutional Network (CRCN).
    Stage 1: Bi-LSTM Encoder -> [P_cls]
    Stage 2: MS-TCN Refinement -> [P'_cls]
    Stage 3: MS-TCN Sharpening -> [P''_cls]
    """

    def __init__(self):
        super(GLT_CRCN, self).__init__()

        self.stage1 = BiLSTMLatentEncoder()

        # Stage 2 Input: 21 Classes
        # Stage 2 Output: 21 Classes
        self.stage2 = MSTCN(
            in_channels=Config.NUM_CLASSES, out_channels=Config.NUM_CLASSES
        )

        # Stage 3 Input: 21 Classes
        # Stage 3 Output: 21 Classes
        self.stage3 = MSTCN(
            in_channels=Config.NUM_CLASSES, out_channels=Config.NUM_CLASSES
        )

    def forward(self, x, mask):
        # ---------------------------------------------------------------------
        # Stage 1: Recurrent Encoder
        # ---------------------------------------------------------------------
        s1_cls = self.stage1(x)  # (B, T, C)

        # ---------------------------------------------------------------------
        # Inter-Stage Masking
        # ---------------------------------------------------------------------
        mask_expanded = mask.unsqueeze(-1).float()
        s1_masked = s1_cls * mask_expanded

        # Transpose for TCN: (B, C, T)
        s1_masked_t = s1_masked.permute(0, 2, 1)

        # ---------------------------------------------------------------------
        # Stage 2: Refinement
        # ---------------------------------------------------------------------
        s2_out_t = self.stage2(s1_masked_t)  # (B, C, T)
        s2_out = s2_out_t.permute(0, 2, 1)
        s2_masked = s2_out * mask_expanded
        s2_masked_t = s2_masked.permute(0, 2, 1)

        # ---------------------------------------------------------------------
        # Stage 3: Sharpening
        # ---------------------------------------------------------------------
        s3_out_t = self.stage3(s2_masked_t)  # (B, C, T)
        s3_out = s3_out_t.permute(0, 2, 1)  # (B, T, C)
        s3_out = s3_out * mask_expanded

        return {
            "stage1_cls": s1_cls,
            "stage2_cls": s2_out,
            "stage3_cls": s3_out,
        }
