import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class Inception1D(nn.Module):
    """
    Multi-Scale 1D Convolutional Block.
    Applies parallel convolutions with different kernel sizes to capture
    features at various frequencies/scales.
    """

    def __init__(self, in_channels, out_channels, kernels=[3, 5, 7, 9]):
        super().__init__()
        self.branches = nn.ModuleList()

        # Calculate output channels per branch to sum up to out_channels
        # We reserve some channels for the bottleneck/reduction if needed,
        # but here we simply split evenly or project at the end.
        # Strategy: Each branch produces out_channels // 4 features.
        branch_channels = out_channels // len(kernels)

        for k in kernels:
            # Padding = (k - 1) // 2 to maintain temporal dimension
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    branch_channels,
                    kernel_size=k,
                    padding=(k - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

        # If division wasn't exact, we might have a mismatch, so we add a final 1x1 conv
        self.concat_channels = branch_channels * len(kernels)
        self.project = nn.Sequential(
            nn.Conv1d(self.concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, C, L)
        outputs = [branch(x) for branch in self.branches]
        out = torch.cat(outputs, dim=1)
        out = self.project(out)
        return out


class EEGEncoder(nn.Module):
    """
    Encodes raw EEG signals using a stack of Inception1D blocks.
    Input: (B, 20, 5000) -> Output: (B, Seq_Len, d_model)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Configuration
        in_channels = config.EEG_CHANNELS
        filters = config.EEG_FILTERS  # e.g., [64, 128, 256, 512]
        kernels = config.EEG_KERNELS
        d_model = config.D_MODEL

        layers = []
        current_channels = in_channels

        for out_channels in filters:
            layers.append(Inception1D(current_channels, out_channels, kernels))
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(0.1))
            current_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        # Final projection to d_model
        self.fc = nn.Conv1d(current_channels, d_model, kernel_size=1)

    def forward(self, x):
        # x: (B, L, C) -> Permute to (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)

        features = self.encoder(x)

        # Project to d_model
        features = self.fc(features)

        # Permute back to (B, L, d_model) for Transformer
        features = features.permute(0, 2, 1)
        return features


class SpectrogramEncoder(nn.Module):
    """
    Encodes Spectrograms using EfficientNet-B1.
    Input: (B, 5, 512, 512) -> Output: (B, Seq_Len, d_model)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Create backbone
        # We use efficientnet_b1 as per description
        self.backbone = timm.create_model(
            config.BACKBONE_2D,
            pretrained=config.PRETRAINED_2D,
            features_only=True,
            out_indices=(4,),  # Get features from the last stage
        )

        # Modify first layer to accept 5 channels (4 regions + 1 coord map)
        # Standard EfficientNet has 3 input channels
        original_stem = self.backbone.conv_stem
        new_stem = nn.Conv2d(
            config.SPEC_CHANNELS,
            original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=False,
        )

        # Initialize weights: Copy first 3 channels, average for the rest
        with torch.no_grad():
            new_stem.weight[:, :3] = original_stem.weight
            # Initialize remaining channels with average of RGB weights
            avg_weight = torch.mean(original_stem.weight, dim=1, keepdim=True)
            new_stem.weight[:, 3:] = avg_weight.repeat(
                1, config.SPEC_CHANNELS - 3, 1, 1
            )

        self.backbone.conv_stem = new_stem

        # Determine output feature dimension
        # EfficientNet-B1 usually outputs 1280 channels at the last stage
        dummy_input = torch.randn(1, config.SPEC_CHANNELS, 512, 512)
        with torch.no_grad():
            features = self.backbone(dummy_input)[0]
            out_channels = features.shape[1]

        self.proj = nn.Linear(out_channels, config.D_MODEL)

    def forward(self, x):
        # x: (B, 5, 512, 512)

        # Extract features: (B, C, H, W)
        # features_only=True returns a list of feature maps
        features = self.backbone(x)[0]

        # Flatten spatial dimensions: (B, C, H*W)
        b, c, h, w = features.shape
        features = features.flatten(2)

        # Permute to (B, Seq_Len, C)
        features = features.transpose(1, 2)

        # Project to d_model
        features = self.proj(features)

        return features


class AsymmetricCoordinateNet(nn.Module):
    """
    Asymmetric Coordinate-Injected Transformer Network.
    Fuses Raw EEG (Query) and Spectrogram (Key/Value) using a Transformer Decoder.
    """

    def __init__(self, config: Config = Config):
        super().__init__()
        self.config = config

        # Encoders
        self.eeg_encoder = EEGEncoder(config)
        self.spec_encoder = SpectrogramEncoder(config)

        # Positional Encoding for EEG Sequence
        # Since EEG is processed by CNNs, local order is kept, but global position
        # in the transformer sequence needs explicit encoding.
        # We use a learnable embedding.
        # Approx sequence length: 5000 / (2^4) = 312
        max_len = (
            config.EEG_SEQ_LEN // (2 ** len(config.EEG_FILTERS)) + 50
        )  # Safety margin
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, config.D_MODEL) * 0.02
        )

        # Fusion: Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            batch_first=True,
            norm_first=True,  # Pre-Norm usually stabilizes training
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.NUM_DECODER_LAYERS
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL),
            nn.LayerNorm(config.D_MODEL),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.D_MODEL, config.NUM_CLASSES),
        )

    def forward(self, eeg, spec, targets=None):
        # eeg: (B, Seq_Len, Channels) -> (B, 20, 5000)
        # spec: (B, 5, 512, 512)

        # 1. Encode Streams
        # EEG Features (Target/Query): (B, T_eeg, D)
        eeg_feats = self.eeg_encoder(eeg)

        # Spec Features (Memory/Key/Value): (B, T_spec, D)
        spec_feats = self.spec_encoder(spec)

        # 2. Add Positional Encoding to EEG
        # Truncate pos_embedding to match current batch sequence length
        seq_len = eeg_feats.shape[1]
        eeg_feats = eeg_feats + self.pos_embedding[:, :seq_len, :]

        # 3. Transformer Fusion
        # tgt = EEG, memory = Spec
        # EEG queries the Spectrogram context
        fused_feats = self.decoder(tgt=eeg_feats, memory=spec_feats)

        # 4. Global Pooling
        # Average over the temporal dimension of the EEG sequence
        pooled_out = torch.mean(fused_feats, dim=1)

        # 5. Classification
        logits = self.classifier(pooled_out)

        # Return logits (CrossEntropy/KLDiv expects logits or log_probs usually,
        # but for inference we want probs. We'll return logits here and handle softmax outside
        # or inside loss function if needed. However, the task description implies
        # predicting probabilities for submission.)

        # For consistency with typical PyTorch training loops using KLDivLoss (which expects log-probs),
        # we will return raw logits. The training loop handles LogSoftmax.
        # For inference, we apply Softmax.

        return logits
