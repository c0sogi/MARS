import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class InceptionBlock1D(nn.Module):
    """
    Multi-Scale 1D Convolutional Block (Inception Style).
    Extracts features using multiple kernel sizes in parallel to capture
    different frequency components.
    """

    def __init__(self, in_channels, out_channels, kernels=[3, 5, 7, 9]):
        super().__init__()
        self.branches = nn.ModuleList()

        # Distribute output channels evenly across branches
        channels_per_branch = out_channels // len(kernels)
        self.remainder = out_channels % len(kernels)

        for i, k in enumerate(kernels):
            # Add remainder channels to the first branch if division isn't exact
            out_c = channels_per_branch + (1 if i < self.remainder else 0)

            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_c, kernel_size=k, padding="same", bias=False
                    ),
                    nn.BatchNorm1d(out_c),
                    nn.SiLU(),
                )
            )

    def forward(self, x):
        # Concatenate branch outputs along channel dimension
        return torch.cat([b(x) for b in self.branches], dim=1)


class EEGEncoder(nn.Module):
    """
    Encodes raw EEG signals into a sequence of feature vectors.
    Input: (Batch, Channels, Time) -> (B, 20, 5000)
    Output: (Batch, Seq_Len, D_Model)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Configuration
        in_channels = config.EEG_CHANNELS
        base_channels = config.EEG_CNN_CHANNELS
        kernels = config.EEG_KERNELS
        d_model = config.D_MODEL

        # Layer 1: Low-level features
        self.block1 = InceptionBlock1D(in_channels, base_channels * 4, kernels)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Layer 2: Mid-level features
        self.block2 = InceptionBlock1D(base_channels * 4, base_channels * 4, kernels)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Layer 3: High-level features
        self.block3 = InceptionBlock1D(base_channels * 4, base_channels * 8, kernels)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Projection to Transformer dimension
        # Input channels: base_channels * 8 (e.g., 32*8 = 256)
        self.proj = nn.Conv1d(base_channels * 8, d_model, kernel_size=1)

        # Positional Embedding for the EEG sequence
        # Sequence length calculation: 5000 -> /4 -> 1250 -> /4 -> 312 -> /2 -> 156
        self.seq_len = 156
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)

    def forward(self, x):
        # x: (B, 20, 5000)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))

        # Project to d_model
        x = self.proj(x)  # (B, d_model, seq_len)

        # Permute for Transformer: (B, Seq_Len, D_Model)
        x = x.permute(0, 2, 1)

        # Add positional embedding
        if x.size(1) != self.seq_len:
            # Handle potential shape mismatch due to padding/rounding
            x = F.interpolate(
                x.permute(0, 2, 1), size=self.seq_len, mode="linear"
            ).permute(0, 2, 1)

        x = x + self.pos_embed
        return x


class SpectrogramEncoder(nn.Module):
    """
    Encodes Spectrogram images into a sequence of context tokens.
    Input: (Batch, 1, Freq, Time) -> (B, 1, 512, 512)
    Output: (Batch, Seq_Len, D_Model)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.D_MODEL

        # Adapter: 1 channel (grayscale) -> 3 channels (RGB) for pretrained backbone
        self.adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        # Backbone: EfficientNet-B2
        # features_only=True returns a list of feature maps
        self.backbone = timm.create_model(
            config.SPEC_BACKBONE,
            pretrained=True,
            features_only=True,
            out_indices=(3,),  # Use deep features (stride 32 usually)
        )

        # Determine backbone output channels dynamically
        dummy = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            feats = self.backbone(dummy)[0]
            # Shape is likely (1, 1408, 16, 16) for B2 at index 3/4
            out_channels = feats.shape[1]
            self.h_feat = feats.shape[2]
            self.w_feat = feats.shape[3]

        # Projection to Transformer dimension
        self.proj = nn.Conv2d(out_channels, d_model, kernel_size=1)

        # Relative Time/Freq Embeddings
        # Since the spectrogram is a fixed window centered on the event,
        # the spatial position (H, W) corresponds to relative time and frequency.
        # We flatten H*W into a sequence.
        self.num_tokens = self.h_feat * self.w_feat
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)

    def forward(self, x):
        # x: (B, 1, 512, 512)
        x = self.adapter(x)

        # Extract features
        feats = self.backbone(x)[0]  # (B, C, H, W)

        # Project to d_model
        x = self.proj(feats)  # (B, d_model, H, W)

        # Flatten to sequence: (B, d_model, H*W) -> (B, H*W, d_model)
        x = x.flatten(2).transpose(1, 2)

        # Add learned relative embeddings
        x = x + self.pos_embed

        return x


class TimeRelativeTransformer(nn.Module):
    """
    Main Architecture: Time-Relative Transformer Decoder Network.
    Fuses EEG Queries with Spectrogram Context.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # Encoders
        self.eeg_encoder = EEGEncoder(config)
        self.spec_encoder = SpectrogramEncoder(config)

        # Fusion: Transformer Decoder
        # EEG acts as Target (Query), Spec acts as Memory (Key/Value)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.fusion_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.NUM_DECODER_LAYERS
        )

        # Classifier Head
        self.head = nn.Sequential(
            nn.LayerNorm(config.D_MODEL), nn.Linear(config.D_MODEL, config.NUM_CLASSES)
        )

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, eeg, spec):
        """
        Args:
            eeg: (Batch, 20, 5000) - Raw EEG
            spec: (Batch, 1, 512, 512) - Spectrogram
        Returns:
            probs: (Batch, 6) - Class probabilities
        """
        # 1. Encode EEG (Queries)
        # Shape: (B, Seq_Len_EEG, D)
        eeg_feats = self.eeg_encoder(eeg)

        # 2. Encode Spectrogram (Keys/Values)
        # Shape: (B, Seq_Len_Spec, D)
        spec_feats = self.spec_encoder(spec)

        # 3. Fusion via Transformer Decoder
        # tgt = eeg_feats, memory = spec_feats
        # Output: (B, Seq_Len_EEG, D)
        decoded = self.fusion_decoder(tgt=eeg_feats, memory=spec_feats)

        # 4. Pooling
        # Global Average Pooling over the EEG sequence time dimension
        pooled = decoded.mean(dim=1)  # (B, D)

        # 5. Classification
        logits = self.head(pooled)  # (B, 6)

        # 6. Output Probabilities (Required for KL Div Loss)
        probs = F.softmax(logits, dim=1)

        return probs
