import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class InceptionBlock1D(nn.Module):
    """
    Multi-Scale 1D Convolutional Block (Inception-style).
    Processes input with parallel branches of different kernel sizes.
    """

    def __init__(self, in_channels, out_channels, kernels=[3, 5, 7, 9]):
        super().__init__()
        self.branches = nn.ModuleList()

        # Ensure out_channels is divisible by number of branches
        assert (
            out_channels % len(kernels) == 0
        ), "out_channels must be divisible by number of kernels"
        branch_channels = out_channels // len(kernels)

        for k in kernels:
            # Padding to maintain temporal dimension: k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        branch_channels,
                        kernel_size=k,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        # x: (Batch, In_Channels, Time)
        branch_outputs = [branch(x) for branch in self.branches]
        return torch.cat(branch_outputs, dim=1)


class EEGEncoder(nn.Module):
    """
    Stream A: Raw EEG Encoder using Multi-Scale 1D CNNs.
    Extracts phase and frequency-specific features from the time-series.
    """

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.EEG_CHANNELS
        self.base_filters = config.EEG_BASE_FILTERS
        self.kernels = config.EEG_KERNELS
        self.hidden_dim = config.FUSION_HIDDEN_DIM

        # Stage 1: (B, 20, 5000) -> (B, 32, 5000) -> Pool -> (B, 32, 1250)
        self.stage1 = nn.Sequential(
            InceptionBlock1D(self.in_channels, self.base_filters, self.kernels),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        # Stage 2: (B, 32, 1250) -> (B, 64, 1250) -> Pool -> (B, 64, 312)
        self.stage2 = nn.Sequential(
            InceptionBlock1D(self.base_filters, self.base_filters * 2, self.kernels),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        # Stage 3: (B, 64, 312) -> (B, 128, 312) -> Pool -> (B, 128, 78)
        self.stage3 = nn.Sequential(
            InceptionBlock1D(
                self.base_filters * 2, self.base_filters * 4, self.kernels
            ),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        # Stage 4: (B, 128, 78) -> (B, 256, 78) -> Pool -> (B, 256, 39)
        self.stage4 = nn.Sequential(
            InceptionBlock1D(
                self.base_filters * 4, self.base_filters * 8, self.kernels
            ),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Final Projection to Fusion Dimension
        # Input: (B, 256, 39) -> Output: (B, FusionDim, 39)
        self.project = nn.Conv1d(self.base_filters * 8, self.hidden_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, 20, 5000)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.project(x)

        # Permute to (Batch, Time, Features) for Attention
        x = x.permute(0, 2, 1)
        return x


class SpecEncoder(nn.Module):
    """
    Stream B: Coordinate-Aware Spectrogram Encoder.
    Uses EfficientNet-B1 to process 5-channel spectrogram inputs.
    """

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.SPEC_CHANNELS
        self.backbone_name = config.SPEC_BACKBONE
        self.hidden_dim = config.FUSION_HIDDEN_DIM

        # Create Backbone (EfficientNet B1)
        # global_pool='' ensures we get the spatial feature map (B, C, H, W)
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=config.SPEC_PRETRAINED,
            in_chans=self.in_channels,
            num_classes=0,
            global_pool="",
        )

        # Determine output channels of the backbone
        # Run a dummy pass to find out feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, self.in_channels, 256, 256)
            out = self.backbone(dummy)
            self.backbone_out_channels = out.shape[1]

        # Projection layer to match Fusion Dimension
        self.project = nn.Conv2d(
            self.backbone_out_channels, self.hidden_dim, kernel_size=1
        )

    def forward(self, x):
        # x: (B, 5, 512, 512)
        x = self.backbone(x)  # (B, 1280, 16, 16) for B1 on 512x512
        x = self.project(x)  # (B, HiddenDim, 16, 16)

        # Flatten spatial dimensions -> Sequence
        # (B, D, H, W) -> (B, D, H*W) -> (B, H*W, D)
        B, D, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        return x


class BidirectionalFusionNet(nn.Module):
    """
    Bidirectional Coordinate-Guided Fusion Network.
    Fuses EEG and Spectrogram features using symmetric cross-attention.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # Encoders
        self.eeg_encoder = EEGEncoder(config)
        self.spec_encoder = SpecEncoder(config)

        dim = config.FUSION_HIDDEN_DIM
        heads = config.ATTENTION_HEADS
        dropout = config.DROPOUT_RATE

        # Bidirectional Cross-Attention
        # Head 1: Waveform-Guided Context (EEG queries Spec)
        self.attn_eeg_query = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )

        # Head 2: Context-Guided Waveform (Spec queries EEG)
        self.attn_spec_query = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )

        # Classification Head
        # Input is concatenation of pooled outputs from both attentions (Dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, config.NUM_CLASSES),
        )

    def forward(self, eeg, spec, **kwargs):
        # 1. Encode Streams
        # eeg_feats: (B, T_eeg, D)
        eeg_feats = self.eeg_encoder(eeg)

        # spec_feats: (B, T_spec, D)
        spec_feats = self.spec_encoder(spec)

        # 2. Bidirectional Cross-Attention

        # Head 1: EEG queries Spec (Q=EEG, K=Spec, V=Spec)
        # "Contextualize the waveform with spectrogram history"
        # Output: (B, T_eeg, D)
        attn1_out, _ = self.attn_eeg_query(
            query=eeg_feats, key=spec_feats, value=spec_feats
        )

        # Head 2: Spec queries EEG (Q=Spec, K=EEG, V=EEG)
        # "Verify spectral patterns against phase details"
        # Output: (B, T_spec, D)
        attn2_out, _ = self.attn_spec_query(
            query=spec_feats, key=eeg_feats, value=eeg_feats
        )

        # 3. Global Pooling
        # Average over the sequence dimension
        pool1 = torch.mean(attn1_out, dim=1)  # (B, D)
        pool2 = torch.mean(attn2_out, dim=1)  # (B, D)

        # 4. Fusion
        fused = torch.cat([pool1, pool2], dim=1)  # (B, 2*D)

        # 5. Classification
        logits = self.classifier(fused)

        return logits
