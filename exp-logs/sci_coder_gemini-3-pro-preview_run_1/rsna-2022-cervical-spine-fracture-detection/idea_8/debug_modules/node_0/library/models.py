import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class UNetLocalizer(nn.Module):
    """
    Stage 1: Multi-Class Anatomical Localizer (2D U-Net).
    Uses an EfficientNet-B0 encoder and a custom decoder for segmentation.
    """

    def __init__(
        self,
        in_channels=Config.STAGE1_IN_CHANNELS,
        num_classes=Config.STAGE1_NUM_CLASSES,
    ):
        super(UNetLocalizer, self).__init__()

        # Encoder: EfficientNet-B0
        # features_only=True returns a list of feature maps from different stages
        self.encoder = timm.create_model(
            Config.STAGE1_BACKBONE,
            features_only=True,
            pretrained=True,
            in_chans=in_channels,
        )

        # Get channel counts from the encoder
        # Indices: 0 (stride 2), 1 (stride 4), 2 (stride 8), 3 (stride 16), 4 (stride 32)
        encoder_channels = self.encoder.feature_info.channels()

        # Decoder
        # We upsample from the deepest layer (idx 4) back to input resolution

        # Block 4: Up from stride 32 to 16
        self.up4 = self._up_block(encoder_channels[4], encoder_channels[3])
        # Block 3: Up from stride 16 to 8
        self.up3 = self._up_block(encoder_channels[3], encoder_channels[2])
        # Block 2: Up from stride 8 to 4
        self.up2 = self._up_block(encoder_channels[2], encoder_channels[1])
        # Block 1: Up from stride 4 to 2
        self.up1 = self._up_block(encoder_channels[1], encoder_channels[0])
        # Block 0: Up from stride 2 to 1 (original resolution)
        # Note: encoder_channels[0] is usually stride 2. We need a final upsample.
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0], 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final classification layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def _up_block(self, in_channels, skip_channels):
        """
        Creates an upsampling block with skip connection concatenation.
        """
        out_channels = skip_channels
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * 2, out_channels, kernel_size=3, padding=1
            ),  # *2 for concat
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder path
        features = self.encoder(x)
        # features list indices: 0 (s2), 1 (s4), 2 (s8), 3 (s16), 4 (s32)

        e0, e1, e2, e3, e4 = features

        # Decoder path with skip connections
        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e0], dim=1)

        d0 = self.up0(d1)

        # Final prediction
        out = self.final_conv(d0)
        return out


class DetailEncoder(nn.Module):
    """
    Stage 2: Mask-Conditioned Detail Encoder (2.5D CNN).
    Uses EfficientNet-V2-S to extract a feature vector from a 4-channel input.
    """

    def __init__(self, in_channels=Config.STAGE2_IN_CHANNELS):
        super(DetailEncoder, self).__init__()

        # Load backbone with num_classes=0 to get pooled features
        self.backbone = timm.create_model(
            Config.STAGE2_BACKBONE,
            pretrained=True,
            num_classes=0,  # Removes classifier, returns global pool
            in_chans=in_channels,
        )

        # The output dimension is determined by the backbone (1280 for effnetv2_s)
        self.feature_dim = self.backbone.num_features

    def forward(self, x):
        # x shape: (Batch, 4, H, W)
        # Output shape: (Batch, Feature_Dim)
        features = self.backbone(x)
        return features


class SoftAnatomicalAttention(nn.Module):
    """
    Computes context vectors for each vertebrae class using attention weighted by
    the anatomical map.
    """

    def __init__(self, input_dim, hidden_dim, num_classes=7):
        super(SoftAnatomicalAttention, self).__init__()
        self.num_classes = num_classes

        # Learnable attention projection
        # We compute a score for each class at each timestep
        self.attn_proj = nn.Linear(input_dim, num_classes)

    def forward(self, rnn_output, anatomical_map):
        """
        Args:
            rnn_output: (Batch, Seq_Len, Hidden_Dim * 2) - BiGRU output
            anatomical_map: (Batch, Seq_Len, 7) - Probability map for C1-C7

        Returns:
            context_vectors: (Batch, 7, Hidden_Dim * 2)
        """
        # 1. Compute raw attention scores from RNN hidden states
        # Shape: (Batch, Seq_Len, 7)
        raw_scores = self.attn_proj(rnn_output)

        # 2. Combine with Anatomical Map
        # We want to attend to slices where the model is confident the vertebrae exists (anatomical_map is high)
        # AND where the RNN finds relevant features (raw_scores is high).
        # We use the anatomical map as a multiplicative gate (or log-space bias).
        # Using multiplication in probability space is equivalent to addition in log space.
        # Here we do: Score = Raw_Score * Anatomical_Map (simple gating)
        # But to ensure gradients flow well, let's do:
        # Attention_Weight = Softmax(Raw_Score) * Anatomical_Map -> Then Normalize

        # Let's use a stable approach:
        # We trust the anatomical map to localize. We trust the attention to pick specific slices within that locality.
        # exp_scores = exp(raw_scores)
        # weighted_scores = exp_scores * anatomical_map
        # alpha = weighted_scores / sum(weighted_scores)

        exp_scores = torch.exp(
            raw_scores - torch.max(raw_scores, dim=1, keepdim=True)[0]
        )  # Stable exp

        # anatomical_map shape: (Batch, Seq_Len, 7)
        # Ensure anatomical map is valid (it should be from Stage 1 softmax/sigmoid)
        # We assume anatomical_map is probabilities [0, 1]

        weighted_scores = exp_scores * anatomical_map

        # Add epsilon to avoid division by zero if a vertebrae is not found in the scan
        normalization = torch.sum(weighted_scores, dim=1, keepdim=True) + 1e-7

        # Alpha shape: (Batch, Seq_Len, 7)
        alpha = weighted_scores / normalization

        # 3. Compute Context Vectors
        # We need to aggregate rnn_output (B, T, H) using alpha (B, T, 7)
        # Result should be (B, 7, H)

        # Transpose alpha to (B, 7, T)
        alpha_t = alpha.permute(0, 2, 1)

        # Matrix multiplication: (B, 7, T) x (B, T, H) -> (B, 7, H)
        context_vectors = torch.bmm(alpha_t, rnn_output)

        return context_vectors


class HierarchicalRNN(nn.Module):
    """
    Stage 3: Hierarchical Anatomical Aggregator (Bi-GRU).
    """

    def __init__(
        self,
        input_dim=Config.STAGE3_INPUT_DIM,
        hidden_dim=Config.STAGE3_HIDDEN_DIM,
        num_layers=Config.STAGE3_NUM_LAYERS,
        dropout=Config.STAGE3_DROPOUT,
    ):
        super(HierarchicalRNN, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        gru_output_dim = hidden_dim * 2

        # Soft Anatomical Attention
        self.attention = SoftAnatomicalAttention(gru_output_dim, hidden_dim)

        # Vertebrae Heads (C1-C7)
        # Each head takes a specific context vector
        self.vertebrae_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gru_output_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                )
                for _ in range(7)
            ]
        )

        # Patient Head
        # Takes concatenation of all 7 context vectors
        # Input dim = 7 * gru_output_dim
        self.patient_head = nn.Sequential(
            nn.Linear(7 * gru_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, features, anatomical_map):
        """
        Args:
            features: (Batch, Seq_Len, Feature_Dim) - From Stage 2
            anatomical_map: (Batch, Seq_Len, 7) - From Stage 1

        Returns:
            logits: (Batch, 8) - [C1, C2, ..., C7, Patient]
        """
        # Concatenate inputs: Visual Features + Anatomical Map
        # Shape: (Batch, Seq_Len, Feature_Dim + 7)
        x = torch.cat([features, anatomical_map], dim=2)

        # Pass through Bi-GRU
        # rnn_out: (Batch, Seq_Len, Hidden_Dim * 2)
        rnn_out, _ = self.gru(x)

        # Compute Context Vectors via Soft Anatomical Attention
        # context: (Batch, 7, Hidden_Dim * 2)
        context = self.attention(rnn_out, anatomical_map)

        # Vertebrae Predictions
        vert_logits = []
        for i in range(7):
            # Select context for vertebrae i: (Batch, Hidden_Dim * 2)
            ctx_i = context[:, i, :]
            # Predict
            logit_i = self.vertebrae_heads[i](ctx_i)
            vert_logits.append(logit_i)

        # Stack logits: (Batch, 7)
        vert_logits = torch.cat(vert_logits, dim=1)

        # Patient Prediction
        # Flatten context: (Batch, 7 * Hidden_Dim * 2)
        patient_ctx = context.view(context.size(0), -1)
        patient_logit = self.patient_head(patient_ctx)

        # Combine all logits: (Batch, 8)
        # Order: C1...C7, Patient
        all_logits = torch.cat([vert_logits, patient_logit], dim=1)

        return all_logits, torch.sigmoid(all_logits)
