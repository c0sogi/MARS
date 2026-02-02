import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config

# -----------------------------------------------------------------------------
# Stage 1: Multi-Task Anatomical Segmentor (U-Net)
# -----------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class AnatomicalSegmentor(nn.Module):
    """
    Stage 1: U-Net for Segmentation and Global Context Extraction.
    Backbone: EfficientNet-B0
    Outputs:
        - Segmentation Mask (1 channel)
        - Anatomical Presence Logits (8 classes: Background + C1-C7)
        - Global Context Vector (1280 dim)
    """

    def __init__(self, pretrained=True):
        super(AnatomicalSegmentor, self).__init__()

        # Encoder (EfficientNet-B0)
        # features_only=True returns a list of feature maps
        self.encoder = timm.create_model(
            Config.SEG_BACKBONE,
            features_only=True,
            pretrained=pretrained,
            in_chans=1,  # Grayscale input
        )

        # Get channel counts from encoder
        # Typical EffNet-B0 indices:
        # 0: stride 2 (16 ch), 1: stride 4 (24 ch), 2: stride 8 (40 ch),
        # 3: stride 16 (112 ch), 4: stride 32 (320 ch)
        enc_channels = self.encoder.feature_info.channels()

        # Decoder
        # We upsample from deepest (idx 4) back to input resolution
        self.up1 = nn.ConvTranspose2d(
            enc_channels[4], enc_channels[3], kernel_size=2, stride=2
        )
        self.conv1 = DoubleConv(enc_channels[3] + enc_channels[3], enc_channels[3])

        self.up2 = nn.ConvTranspose2d(
            enc_channels[3], enc_channels[2], kernel_size=2, stride=2
        )
        self.conv2 = DoubleConv(enc_channels[2] + enc_channels[2], enc_channels[2])

        self.up3 = nn.ConvTranspose2d(
            enc_channels[2], enc_channels[1], kernel_size=2, stride=2
        )
        self.conv3 = DoubleConv(enc_channels[1] + enc_channels[1], enc_channels[1])

        self.up4 = nn.ConvTranspose2d(
            enc_channels[1], enc_channels[0], kernel_size=2, stride=2
        )
        self.conv4 = DoubleConv(enc_channels[0] + enc_channels[0], enc_channels[0])

        # Final upsample to recover original resolution (stride 2 -> 1)
        self.up_final = nn.ConvTranspose2d(enc_channels[0], 16, kernel_size=2, stride=2)
        self.conv_final = nn.Conv2d(16, 1, kernel_size=1)  # Binary mask

        # Global Context & Anatomical Classification
        # Project bottleneck (320) to Config.RNN_INPUT_DIM component (1280)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.context_projector = nn.Linear(enc_channels[4], 1280)
        self.anatomical_head = nn.Linear(1280, Config.SEG_CLASSES)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # features indices: 0(s2), 1(s4), 2(s8), 3(s16), 4(s32)

        x0, x1, x2, x3, x4 = features

        # Global Context Branch
        global_feat = self.global_pool(x4).flatten(1)
        global_context = self.context_projector(global_feat)  # 1280 dim
        anatomical_logits = self.anatomical_head(global_context)

        # Decoder Branch
        x = self.up1(x4)
        # Handle slight shape mismatch due to padding if necessary, but 256x256 is power of 2
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x0], dim=1)
        x = self.conv4(x)

        x = self.up_final(x)
        mask_logits = self.conv_final(x)

        return mask_logits, anatomical_logits, global_context


# -----------------------------------------------------------------------------
# Stage 2: Mask-Guided High-Resolution Encoder (2.5D CNN)
# -----------------------------------------------------------------------------


class FractureEncoder(nn.Module):
    """
    Stage 2: 2.5D CNN for Slice-Level Fracture Feature Extraction.
    Backbone: EfficientNet-V2-S
    Input: 4 Channels (3 RGB Slices + 1 Bone Mask)
    Output: Local Fracture Embedding (1280 dim)
    """

    def __init__(self, pretrained=True):
        super(FractureEncoder, self).__init__()

        # Load backbone with no classifier
        self.backbone = timm.create_model(
            Config.CLS_BACKBONE,
            pretrained=pretrained,
            num_classes=0,  # Returns pooled features
            in_chans=Config.IN_CHANNELS_CLS,  # 4 channels
        )

        # Verify output dimension
        # EfficientNet-V2-S usually outputs 1280
        self.output_dim = self.backbone.num_features
        if self.output_dim != Config.CLS_EMBED_DIM:
            # Add projection if mismatch (though EffNetV2-S matches 1280)
            self.projector = nn.Linear(self.output_dim, Config.CLS_EMBED_DIM)
        else:
            self.projector = nn.Identity()

    def forward(self, x):
        # x shape: (B, 4, H, W)
        features = self.backbone(x)
        features = self.projector(features)
        return features


# -----------------------------------------------------------------------------
# Stage 3: Hybrid-Feature Recurrent Aggregator (Bi-GRU)
# -----------------------------------------------------------------------------


class AnatomicallyBiasedAttention(nn.Module):
    """
    Attention mechanism that pools sequence features for a specific vertebra,
    biased by the anatomical probability profile.
    """

    def __init__(self, hidden_size):
        super(AnatomicallyBiasedAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, rnn_output, anatomical_prob):
        """
        Args:
            rnn_output: (B, T, Hidden)
            anatomical_prob: (B, T) - Probability of specific vertebra k at each step
        Returns:
            context_vector: (B, Hidden)
        """
        # Raw attention scores from RNN state: (B, T, 1)
        scores = self.attention(rnn_output)

        # Bias the scores with anatomical probability
        # We add log(prob) to the score.
        # If prob is near 0, log(prob) is large negative -> score drops -> weight ~ 0.
        # Add epsilon for numerical stability.
        epsilon = 1e-7
        biased_scores = scores.squeeze(-1) + torch.log(anatomical_prob + epsilon)

        # Softmax over time dimension
        weights = F.softmax(biased_scores, dim=1).unsqueeze(-1)  # (B, T, 1)

        # Weighted sum
        context = torch.sum(rnn_output * weights, dim=1)  # (B, Hidden)

        return context


class HCHRNAggregator(nn.Module):
    """
    Stage 3: Bi-GRU with Anatomically-Biased Attention.
    Inputs: Sequence of (Local Embed + Global Context + Anatomical Probs)
    Outputs: Logits for C1-C7 and patient_overall
    """

    def __init__(self):
        super(HCHRNAggregator, self).__init__()

        self.input_dim = Config.RNN_INPUT_DIM
        self.hidden_size = Config.RNN_HIDDEN_SIZE

        # Bi-Directional GRU
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.RNN_BIDIRECTIONAL,
            dropout=Config.RNN_DROPOUT if Config.RNN_LAYERS > 1 else 0,
        )

        # Effective hidden size (bidirectional = 2x)
        self.feature_dim = (
            self.hidden_size * 2 if Config.RNN_BIDIRECTIONAL else self.hidden_size
        )

        # Attention Modules for C1-C7
        # We create a ModuleList of 7 attention heads
        self.attention_heads = nn.ModuleList(
            [AnatomicallyBiasedAttention(self.feature_dim) for _ in range(7)]
        )

        # Classifiers for each vertebra
        self.vertebrae_classifiers = nn.ModuleList(
            [nn.Linear(self.feature_dim, 1) for _ in range(7)]
        )

        # Patient Level Classifier
        # Concatenates all 7 context vectors
        self.patient_classifier = nn.Linear(self.feature_dim * 7, 1)

    def forward(self, features, anatomical_probs):
        """
        Args:
            features: (B, T, Input_Dim) - Concatenated Local + Global + Probs
            anatomical_probs: (B, T, 8) - Raw probabilities from Stage 1 (0=BG, 1..7=C1..C7)
        """
        # RNN Forward
        self.gru.flatten_parameters()
        rnn_output, _ = self.gru(features)  # (B, T, feature_dim)

        vertebrae_logits = []
        vertebrae_contexts = []

        # Process each vertebra (C1 to C7)
        for i in range(7):
            # Extract probability for C(i+1). Index 0 is background, so C1 is index 1.
            prob_k = anatomical_probs[:, :, i + 1]

            # Get context vector via biased attention
            context_k = self.attention_heads[i](rnn_output, prob_k)
            vertebrae_contexts.append(context_k)

            # Predict fracture for this vertebra
            logits_k = self.vertebrae_classifiers[i](context_k)
            vertebrae_logits.append(logits_k)

        # Stack vertebrae logits (B, 7)
        vertebrae_logits = torch.cat(vertebrae_logits, dim=1)

        # Patient Overall Prediction
        # Concat all contexts: (B, feature_dim * 7)
        patient_feat = torch.cat(vertebrae_contexts, dim=1)
        patient_logit = self.patient_classifier(patient_feat)

        # Final Output: [C1, C2, ..., C7, patient_overall]
        final_logits = torch.cat([vertebrae_logits, patient_logit], dim=1)

        return final_logits
