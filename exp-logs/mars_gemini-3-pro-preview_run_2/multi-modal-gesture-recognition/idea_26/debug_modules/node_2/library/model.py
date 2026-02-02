import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class GeometricFeatureExtractor(nn.Module):
    """
    Computes geometric features (Velocity, Bone Vectors) from raw skeleton data
    and concatenates them with audio features.
    """

    def __init__(self):
        super(GeometricFeatureExtractor, self).__init__()

        # Define Upper Body Bone Connections (Parent, Child) indices relative to the 12-joint subset
        # 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head
        # 4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 7:HandLeft
        # 8:ShoulderRight, 9:ElbowRight, 10:WristRight, 11:HandRight
        self.bones = [
            (0, 1),
            (1, 2),
            (2, 3),  # Spine chain
            (2, 4),
            (4, 5),
            (5, 6),
            (6, 7),  # Left arm
            (2, 8),
            (8, 9),
            (9, 10),
            (10, 11),  # Right arm
        ]
        self.num_bones = len(self.bones)  # 11
        self.num_joints = len(config.UPPER_BODY_JOINTS)  # 12

        # Input dimensions
        # Pos (12*3) + Vel (12*3) + Bones (11*3) + Audio (13)
        self.input_dim = (
            (self.num_joints * 3)
            + (self.num_joints * 3)
            + (self.num_bones * 3)
            + config.HYPERPARAMS["audio_n_mfcc"]
        )

    def forward(self, skeleton, audio):
        """
        Args:
            skeleton: (B, T, 12, 3) Raw positions
            audio: (B, T, 13) MFCC features
        Returns:
            features: (B, T, input_dim)
        """
        B, T, J, C = skeleton.shape

        # 1. Normalize Skeleton (Center around HipCenter/Index 0 and Scale)
        # HipCenter is at index 0
        hip_center = skeleton[:, :, 0:1, :]  # (B, T, 1, 3)
        skeleton_norm = (skeleton - hip_center) * config.SKELETON_SCALE

        # Flatten Positions: (B, T, 36)
        pos_feat = skeleton_norm.reshape(B, T, -1)

        # 2. Compute Velocity: P_t - P_{t-1}
        # Pad first frame with 0
        padded_pos = torch.cat(
            [
                torch.zeros(B, 1, J, C, device=skeleton.device),
                skeleton_norm[:, :-1, :, :],
            ],
            dim=1,
        )
        vel_feat = (skeleton_norm - padded_pos).reshape(B, T, -1)

        # 3. Compute Bone Vectors
        bone_list = []
        for p_idx, c_idx in self.bones:
            # Vector from Parent to Child
            bone_vec = skeleton_norm[:, :, c_idx, :] - skeleton_norm[:, :, p_idx, :]
            bone_list.append(bone_vec)
        bone_feat = torch.cat(bone_list, dim=-1)  # (B, T, 33)

        # 4. Concatenate all
        # Audio is (B, T, 13)
        features = torch.cat([pos_feat, vel_feat, bone_feat, audio], dim=-1)

        return features


class SimplifiedGatedBlock(nn.Module):
    """
    Dilated Convolutional Block with Gated Activation.
    Structure: Input -> Dilated Conv -> Split(Tanh, Sigmoid) -> Mult -> Dropout -> + Input
    Crucially: No 1x1 output projection in the residual path.
    """

    def __init__(self, hidden_dim, kernel_size, dilation, dropout):
        super(SimplifiedGatedBlock, self).__init__()

        self.hidden_dim = hidden_dim

        # Padding to maintain temporal dimension (assuming centered/non-causal for offline)
        # For kernel 3: dilation 1 -> pad 1, dilation 2 -> pad 2, etc.
        padding = (kernel_size - 1) * dilation // 2

        # Convolution produces 2 * hidden_dim channels (for filter and gate)
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        Returns:
            out: (B, C, T)
        """
        residual = x

        # Dilated Conv
        out = self.conv(x)

        # Split into Filter and Gate
        filter_out, gate_out = torch.split(out, self.hidden_dim, dim=1)

        # Gated Activation
        z = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        # Dropout
        z = self.dropout(z)

        # Residual Connection (Direct)
        return residual + z


class GeometricRecurrentEncoder(nn.Module):
    """
    Stage 1: Bi-LSTM Encoder with geometric feature expansion.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GeometricRecurrentEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=config.HYPERPARAMS["lstm_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=(
                config.HYPERPARAMS["dropout"]
                if config.HYPERPARAMS["lstm_layers"] > 1
                else 0
            ),
        )

        # Project LSTM output (hidden_dim * 2) to heads
        lstm_out_dim = hidden_dim * 2

        self.head_cls = nn.Linear(lstm_out_dim, num_classes)
        self.head_bnd = nn.Linear(lstm_out_dim, 1)
        self.head_fg = nn.Linear(lstm_out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, T, Input_Dim)
        Returns:
            dict containing logits for cls, bnd, fg
        """
        # LSTM
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)

        # Heads
        logits_cls = self.head_cls(lstm_out)  # (B, T, C)
        logits_bnd = self.head_bnd(lstm_out)  # (B, T, 1)
        logits_fg = self.head_fg(lstm_out)  # (B, T, 1)

        return {"cls": logits_cls, "bnd": logits_bnd, "fg": logits_fg}


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: MS-TCN style refinement using Simplified Gated Blocks.
    """

    def __init__(
        self, input_dim, hidden_dim, num_layers, kernel_size, dilations, dropout
    ):
        super(RefinementStage, self).__init__()

        # Input Projection: (C + 1 + 1) -> Hidden
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Stack of Gated Blocks
        self.layers = nn.ModuleList(
            [
                SimplifiedGatedBlock(hidden_dim, kernel_size, dilation, dropout)
                for dilation in dilations
            ]
        )

        # Output Projection: Hidden -> (C + 1 + 1)
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

        # Split output back to heads
        self.num_classes = config.NUM_CLASSES

    def forward(self, x, mask):
        """
        Args:
            x: (B, Input_Dim, T) - Concatenated probs/logits from prev stage
            mask: (B, 1, T)
        Returns:
            dict containing logits
        """
        # Mask input
        out = x * mask

        # Project
        out = self.input_proj(out)

        # Apply layers
        for layer in self.layers:
            out = layer(out)
            out = out * mask  # Apply mask after every layer to keep padding zero

        # Project back
        out = self.output_proj(out)

        # Split (Channel dimension is 1)
        # Input was [cls (21), bnd (1), fg (1)]
        logits_cls = out[:, : self.num_classes, :]
        logits_bnd = out[:, self.num_classes : self.num_classes + 1, :]
        logits_fg = out[:, self.num_classes + 1 :, :]

        # Transpose back to (B, T, C) for consistency
        return {
            "cls": logits_cls.transpose(1, 2),
            "bnd": logits_bnd.transpose(1, 2),
            "fg": logits_fg.transpose(1, 2),
        }


class HGGCRCN(nn.Module):
    """
    Hierarchical Geometric Gated-Cascaded Recurrent-Convolutional Network.

    Pipeline:
    1. Geometric Feature Extraction
    2. Stage 1: Bi-LSTM Encoder -> [Cls, Bnd, Fg]
    3. Stage 2: Refinement (TCN) -> [Cls, Bnd, Fg]
    4. Stage 3: Sharpening (TCN) -> [Cls, Bnd, Fg]
    """

    def __init__(self):
        super(HGGCRCN, self).__init__()

        self.feature_extractor = GeometricFeatureExtractor()

        # Hyperparameters
        hidden_dim = config.HYPERPARAMS["hidden_dim"]
        num_classes = config.NUM_CLASSES
        tcn_layers = config.HYPERPARAMS["tcn_layers"]
        kernel_size = config.HYPERPARAMS["tcn_kernel_size"]
        dilations = config.HYPERPARAMS["tcn_dilations"]
        dropout = config.HYPERPARAMS["dropout"]

        # Stage 1
        self.stage1 = GeometricRecurrentEncoder(
            input_dim=self.feature_extractor.input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

        # Dimensions passed between stages: Cls + Bnd + Fg
        inter_stage_dim = num_classes + 1 + 1

        # Stage 2
        self.stage2 = RefinementStage(
            input_dim=inter_stage_dim,
            hidden_dim=hidden_dim,
            num_layers=tcn_layers,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )

        # Stage 3
        self.stage3 = RefinementStage(
            input_dim=inter_stage_dim,
            hidden_dim=hidden_dim,
            num_layers=tcn_layers,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )

    def forward(self, skeleton, audio, mask):
        """
        Args:
            skeleton: (B, T, 12, 3)
            audio: (B, T, 13)
            mask: (B, T)
        Returns:
            stage_outputs: List of dicts
        """
        # 1. Feature Extraction
        features = self.feature_extractor(skeleton, audio)  # (B, T, FeatDim)

        # 2. Stage 1 (LSTM)
        s1_out = self.stage1(features)

        # Prepare input for Stage 2
        # Concatenate Softmax/Sigmoid probabilities or Logits?
        # Usually TCN refinement works on Logits or Softmax.
        # Deep Supervision usually computes loss on logits.
        # However, passing probabilities to next stage is standard in MS-TCN.
        # Let's pass Probabilities to provide normalized signals to the gates.

        s1_probs_cls = F.softmax(s1_out["cls"], dim=2)
        s1_probs_bnd = torch.sigmoid(s1_out["bnd"])
        s1_probs_fg = torch.sigmoid(s1_out["fg"])

        # Concat: (B, T, C+2)
        s1_combined = torch.cat([s1_probs_cls, s1_probs_bnd, s1_probs_fg], dim=2)

        # Transpose for Conv1d: (B, C+2, T)
        s1_input_next = s1_combined.transpose(1, 2)

        # Mask for Conv1d: (B, 1, T)
        mask_conv = mask.unsqueeze(1)

        # 3. Stage 2 (Refinement)
        s2_out = self.stage2(s1_input_next, mask_conv)

        # Prepare input for Stage 3
        s2_probs_cls = F.softmax(s2_out["cls"], dim=2)
        s2_probs_bnd = torch.sigmoid(s2_out["bnd"])
        s2_probs_fg = torch.sigmoid(s2_out["fg"])

        s2_combined = torch.cat([s2_probs_cls, s2_probs_bnd, s2_probs_fg], dim=2)
        s2_input_next = s2_combined.transpose(1, 2)

        # 4. Stage 3 (Sharpening)
        s3_out = self.stage3(s2_input_next, mask_conv)

        return [s1_out, s2_out, s3_out]
