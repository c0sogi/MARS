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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        x = self.linear_proj(x)
        x = self.temporal_conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class FeatureStem(nn.Module):
    """
    Stage 1: Generation Stage.
    Processes raw multi-modal inputs into initial frame-wise predictions.
    """

    def __init__(self):
        super(FeatureStem, self).__init__()

        # 1. Modality Normalization
        # Using InstanceNorm1d with affine=False to enforce Z-score standardization (0 mean, 1 var)
        # per sample, per channel, independently.
        self.norm_skeleton = nn.InstanceNorm1d(Config.INPUT_DIM_SKELETON, affine=False)
        self.norm_audio = nn.InstanceNorm1d(Config.INPUT_DIM_AUDIO, affine=False)

        # 2. Projection Blocks
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

        # 3. Fusion
        # Concatenation results in 2 * PROJECTION_DIM channels
        fusion_dim = Config.PROJECTION_DIM * 2
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # 4. Sequence Modeling
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.GRU_LAYERS > 1 else 0.0,
        )

        # 5. Classifier Head (Stage 1 Output)
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
        # Permute to (Batch, Channels, Time) for Conv1d and InstanceNorm
        skel_t = skeleton.transpose(1, 2)
        audio_t = audio.transpose(1, 2)

        # Modality Normalization
        skel_t = self.norm_skeleton(skel_t)
        audio_t = self.norm_audio(audio_t)

        # Projection
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


class RefinementModule(nn.Module):
    """
    Stage 2: Refinement Stage.
    Cleans and corrects Stage 1 predictions using temporal context.
    """

    def __init__(self):
        super(RefinementModule, self).__init__()

        layers = []
        input_dim = Config.NUM_CLASSES
        hidden_dim = Config.PROJECTION_DIM  # Lightweight intermediate dimension

        # Stack of Dilated Convolutions
        for i in range(Config.REFINE_LAYERS):
            dilation = 2**i
            kernel_size = Config.REFINE_KERNEL_SIZE
            padding = (kernel_size - 1) * dilation // 2

            # Input dim for first layer is NumClasses, else HiddenDim
            in_c = input_dim if i == 0 else hidden_dim
            # Output dim for last layer is NumClasses, else HiddenDim
            out_c = input_dim if i == (Config.REFINE_LAYERS - 1) else hidden_dim

            conv = nn.Conv1d(
                in_c, out_c, kernel_size=kernel_size, padding=padding, dilation=dilation
            )

            layers.append(conv)

            # Activation and Dropout for intermediate layers
            if i < (Config.REFINE_LAYERS - 1):
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(Config.DROPOUT))

        self.net = nn.Sequential(*layers)

    def forward(self, stage1_logits):
        """
        Args:
            stage1_logits: (Batch, Time, NumClasses)
        Returns:
            stage2_logits: (Batch, Time, NumClasses)
        """
        # Permute to (Batch, Channels, Time) for Conv1d
        x = stage1_logits.transpose(1, 2)

        # Apply refinement network
        # We add a residual connection if dimensions match, but here we are transforming
        # logits to logits, so essentially learning a residual correction
        correction = self.net(x)

        # Permute back
        out = correction.transpose(1, 2)

        # Residual connection: Output = Input + Correction
        # This helps gradient flow and implies the module learns "what to fix"
        return stage1_logits + out


class MSRN(nn.Module):
    """
    Multi-Stage Refinement Network.
    Combines FeatureStem and RefinementModule.
    """

    def __init__(self):
        super(MSRN, self).__init__()
        self.stem = FeatureStem()
        self.refinement = RefinementModule()

    def forward(self, skeleton, audio):
        """
        Args:
            skeleton: (Batch, Time, InputDim_Skel)
            audio: (Batch, Time, InputDim_Audio)
        Returns:
            stage1_logits: (Batch, Time, NumClasses)
            stage2_logits: (Batch, Time, NumClasses)
        """
        # Stage 1: Generation
        stage1_logits = self.stem(skeleton, audio)

        # Stage 2: Refinement
        # We pass the logits directly. The refinement module learns to smooth them.
        stage2_logits = self.refinement(stage1_logits)

        return stage1_logits, stage2_logits
