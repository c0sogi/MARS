import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ProjectionBlock(nn.Module):
    """
    A projection block consisting of:
    Linear Projection -> Temporal Convolution (1D) -> Activation (ReLU) -> Dropout.
    """

    def __init__(self, input_dim, output_dim, kernel_size, dropout):
        super(ProjectionBlock, self).__init__()

        # Linear Projection: Implemented as Conv1d with kernel_size=1
        self.linear_proj = nn.Conv1d(input_dim, output_dim, kernel_size=1)

        # Temporal Convolution
        # Padding is calculated to maintain temporal dimension: (k - 1) // 2
        padding = (kernel_size - 1) // 2
        self.temporal_conv = nn.Conv1d(
            output_dim, output_dim, kernel_size=kernel_size, padding=padding, groups=1
        )

        self.activation = nn.ReLU()
        # Cite Lesson 00013: Robustness via Channel Masking.
        # Using Dropout1d (Spatial Dropout) to drop entire channels during training,
        # which acts as a learnable version of random channel masking.
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        x = self.linear_proj(x)
        x = self.temporal_conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MultiStreamGRU(nn.Module):
    """
    Multi-Stream GRU Network.
    Implements independent feature stems for Skeleton and Audio, followed by fusion and GRU.
    Relies on global normalization provided by the DataLoader.
    Cite Lesson 00021: Decoupling Heterogeneous Modalities.
    Cite Lesson 00019: Simpler RNN (removing refinement).
    """

    def __init__(self):
        super(MultiStreamGRU, self).__init__()

        # 1. Projection Blocks (Independent Stems)
        self.proj_skeleton = ProjectionBlock(
            Config.INPUT_DIM_SKELETON,
            Config.PROJECTION_DIM,
            Config.STEM_KERNEL_SIZE,
            Config.DROPOUT,
        )
        self.proj_audio = ProjectionBlock(
            Config.INPUT_DIM_AUDIO,
            Config.PROJECTION_DIM,
            Config.STEM_KERNEL_SIZE,
            Config.DROPOUT,
        )

        # 2. Fusion
        # Concatenation results in 2 * PROJECTION_DIM channels
        fusion_dim = Config.PROJECTION_DIM * 2
        # Cite Lesson 00011: LayerNorm after fusion
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # 3. Sequence Modeling
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.GRU_LAYERS > 1 else 0.0,
        )

        # 4. Classifier Head
        gru_output_dim = (
            Config.HIDDEN_DIM * 2 if Config.BIDIRECTIONAL else Config.HIDDEN_DIM
        )
        self.classifier = nn.Linear(gru_output_dim, Config.NUM_CLASSES)

    def forward(self, skeleton, audio):
        """
        Args:
            skeleton: (Batch, Time, InputDim_Skel)
            audio: (Batch, Time, InputDim_Audio)
        Returns:
            logits: (Batch, Time, NumClasses)
        """
        # Permute to (Batch, Channels, Time) for Conv1d
        skel_t = skeleton.transpose(1, 2)
        audio_t = audio.transpose(1, 2)

        # Projection (Note: No InstanceNorm here, relying on Global Norm)
        skel_feat = self.proj_skeleton(skel_t)
        audio_feat = self.proj_audio(audio_t)

        # Fusion
        # Concatenate along channel dimension
        fused = torch.cat([skel_feat, audio_feat], dim=1)

        # Permute back to (Batch, Time, Channels) for LayerNorm and GRU
        fused = fused.transpose(1, 2)
        fused = self.fusion_norm(fused)

        # Sequence Modeling
        gru_out, _ = self.gru(fused)

        # Classification
        logits = self.classifier(gru_out)

        return logits
