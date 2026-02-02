import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualLayer(nn.Module):
    """
    Standard Dilated Residual Layer for MS-TCN.
    Includes Dilated Conv, InstanceNorm, ReLU, Dropout, and Residual Connection.
    Cite Lesson 84, 63: InstanceNorm is critical for TCN efficiency and stability.
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
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
    Stage 1: Latent-Transition Recurrent Encoder.
    Backbone: Bi-LSTM
    Heads:
        1. Class Probabilities (Softmax)
        2. Latent Transition Signal (Sigmoid)
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

        # Head 2: Latent Transition (1 channel)
        self.trans_head = nn.Linear(lstm_out_dim, 1)

    def forward(self, x, mask=None):
        # x: (Batch, Time, Features)

        # Pack padded sequence if mask is provided for efficiency (optional but good practice)
        # However, for simplicity and compatibility with TCN downstream, we use raw output.
        # We rely on the mask provided to the loss function and the inter-stage masking.

        lstm_out, _ = self.lstm(x)

        # Predictions
        cls_logits = self.cls_head(lstm_out)  # (B, T, C)
        trans_logits = self.trans_head(lstm_out)  # (B, T, 1)

        # Apply activations
        cls_probs = F.softmax(cls_logits, dim=2)
        trans_probs = torch.sigmoid(trans_logits)

        return cls_probs, trans_probs


class MSTCN(nn.Module):
    """
    Multi-Stage Temporal Convolutional Network with Instance Normalization.
    Used for Stage 2 (Refinement) and Stage 3 (Sharpening).
    Cite Lesson 84: Replaced Gated TCN with Dilated TCN + InstanceNorm.
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

        # Apply activations
        if out.shape[1] == Config.NUM_CLASSES:
            # Stage 3 case: only class probs
            out = F.softmax(out, dim=1)
        else:
            # Stage 2 case: classes + transition
            # Split
            cls_logits = out[:, : Config.NUM_CLASSES, :]
            trans_logits = out[:, Config.NUM_CLASSES :, :]

            cls_probs = F.softmax(cls_logits, dim=1)
            trans_probs = torch.sigmoid(trans_logits)
            out = torch.cat([cls_probs, trans_probs], dim=1)

        return out


class GLT_CRCN(nn.Module):
    """
    Latent-Transition Cascaded Recurrent-Convolutional Network.
    Stage 1: Bi-LSTM Encoder -> [P_cls, P_trans]
    Stage 2: MS-TCN Refinement -> [P'_cls, P'_trans]
    Stage 3: MS-TCN Sharpening -> [P''_cls]
    """

    def __init__(self):
        super(GLT_CRCN, self).__init__()

        self.stage1 = BiLSTMLatentEncoder()

        # Stage 2 Input: 21 Classes + 1 Transition = 22
        # Stage 2 Output: 21 Classes + 1 Transition = 22
        self.stage2 = MSTCN(
            in_channels=Config.NUM_CLASSES + 1, out_channels=Config.NUM_CLASSES + 1
        )

        # Stage 3 Input: 22 (Output of Stage 2)
        # Stage 3 Output: 21 Classes
        self.stage3 = MSTCN(
            in_channels=Config.NUM_CLASSES + 1, out_channels=Config.NUM_CLASSES
        )

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, Time, Features)
            mask: (Batch, Time) - Boolean mask where True indicates valid frames.
        Returns:
            Dictionary containing outputs from all stages.
        """
        # ---------------------------------------------------------------------
        # Stage 1: Recurrent Encoder
        # ---------------------------------------------------------------------
        s1_cls, s1_trans = self.stage1(x)  # (B, T, C), (B, T, 1)

        # Concatenate for next stage
        # Shape: (B, T, C+1)
        s1_out = torch.cat([s1_cls, s1_trans], dim=2)

        # ---------------------------------------------------------------------
        # Inter-Stage Masking
        # ---------------------------------------------------------------------
        # Zero out padding to prevent noise propagation
        # Mask shape (B, T) -> (B, T, 1)
        mask_expanded = mask.unsqueeze(-1).float()
        s1_masked = s1_out * mask_expanded

        # Transpose for TCN: (B, C+1, T)
        s1_masked_t = s1_masked.permute(0, 2, 1)

        # ---------------------------------------------------------------------
        # Stage 2: Gated Refinement
        # ---------------------------------------------------------------------
        s2_out_t = self.stage2(s1_masked_t)  # (B, C+1, T)

        # Mask again (though TCN should handle it if padding is correct, explicit is safer)
        # Transpose back to (B, T, C+1) for masking
        s2_out = s2_out_t.permute(0, 2, 1)
        s2_masked = s2_out * mask_expanded

        # Transpose back for Stage 3
        s2_masked_t = s2_masked.permute(0, 2, 1)

        # ---------------------------------------------------------------------
        # Stage 3: Gated Sharpening
        # ---------------------------------------------------------------------
        s3_out_t = self.stage3(s2_masked_t)  # (B, C, T)
        s3_out = s3_out_t.permute(0, 2, 1)  # (B, T, C)

        # Final masking not strictly needed for loss but good for return
        s3_out = s3_out * mask_expanded

        # ---------------------------------------------------------------------
        # Extract Components for Output
        # ---------------------------------------------------------------------
        # Stage 2 components
        s2_cls = s2_out[:, :, : Config.NUM_CLASSES]
        s2_trans = s2_out[:, :, Config.NUM_CLASSES :]

        return {
            "stage1_cls": s1_cls,
            "stage1_trans": s1_trans,
            "stage2_cls": s2_cls,
            "stage2_trans": s2_trans,
            "stage3_cls": s3_out,
        }
