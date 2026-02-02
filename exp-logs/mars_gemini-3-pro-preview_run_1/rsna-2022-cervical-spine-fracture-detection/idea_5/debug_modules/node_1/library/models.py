import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from library.config import Config


# -----------------------------------------------------------------------------
# Stage 1: Anatomical Localizer & Segmentor (2D U-Net with ResNet18 Encoder)
# -----------------------------------------------------------------------------
class UNetLocalizer(nn.Module):
    """
    U-Net architecture with a ResNet18 encoder.
    Predicts pixel-wise masks for Background + C1-C7 (8 classes).
    """

    def __init__(self, num_classes=Config.SEG_NUM_CLASSES, pretrained=True):
        super(UNetLocalizer, self).__init__()

        # Load ResNet18 Encoder
        # We use torchvision's implementation
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.encoder = models.resnet18(weights=weights)

        # Modify first layer to accept 1 channel (Grayscale DICOM)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias,
        )

        # Initialize the new 1-channel weights by averaging the original 3-channel weights
        if pretrained:
            with torch.no_grad():
                self.encoder.conv1.weight.data = original_conv.weight.data.mean(
                    dim=1, keepdim=True
                )

        # Encoder Layers (accessing by name for skip connections)
        self.enc_conv1 = self.encoder.conv1
        self.enc_bn1 = self.encoder.bn1
        self.enc_relu = self.encoder.relu
        self.enc_maxpool = self.encoder.maxpool
        self.enc_layer1 = self.encoder.layer1  # 64 channels
        self.enc_layer2 = self.encoder.layer2  # 128 channels
        self.enc_layer3 = self.encoder.layer3  # 256 channels
        self.enc_layer4 = self.encoder.layer4  # 512 channels

        # Decoder Layers
        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(256, 128)
        self.up2 = self._up_block(128, 64)
        self.up1 = self._up_block(64, 64)

        # Final Classifier
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _up_block(self, in_channels, out_channels):
        """
        Simple upsampling block: Bilinear Upsample -> Conv -> BN -> ReLU
        Note: Input to this block will be cat(upsampled, skip), so in_channels needs to handle that.
        Here we define the conv part. The concatenation happens in forward.
        """
        # The input to the conv will be in_channels (from skip) + in_channels (from upsampled prev layer)
        # Wait, standard U-Net:
        # Layer 4 out: 512. Up -> 512. Concat with Layer 3 (256). Total 768.
        # To keep it simple and lightweight:
        # We will reduce channels after concatenation.

        return nn.Sequential(
            nn.Conv2d(
                in_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = self.enc_relu(x)
        x1 = x  # 64, H/2, W/2

        x = self.enc_maxpool(x)
        x2 = self.enc_layer1(x)  # 64, H/4, W/4
        x3 = self.enc_layer2(x2)  # 128, H/8, W/8
        x4 = self.enc_layer3(x3)  # 256, H/16, W/16
        x5 = self.enc_layer4(x4)  # 512, H/32, W/32

        # Decoder
        # Up 4
        d5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=True)
        # x4 is 256 ch, d5 is 512 ch. Concat -> 768.
        # self.up4 expects in=512 (from up) + out=256 (from skip).
        # Wait, my _up_block definition: in_channels is from 'up', out_channels is 'skip' dim.
        d4 = self.up4(torch.cat([d5, x4], dim=1))

        # Up 3
        d4_up = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        d3 = self.up3(torch.cat([d4_up, x3], dim=1))

        # Up 2
        d3_up = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        d2 = self.up2(torch.cat([d3_up, x2], dim=1))

        # Up 1
        d2_up = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        # x1 is 64 ch (before maxpool). d2_up is 64 ch.
        d1 = self.up1(torch.cat([d2_up, x1], dim=1))

        # Final upsample to original size if needed (x1 is H/2)
        d1_up = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=True)

        logits = self.final_conv(d1_up)

        return logits


# -----------------------------------------------------------------------------
# Stage 2: Mask-Conditioned Focus Encoder (2.5D CNN)
# -----------------------------------------------------------------------------
class FractureEncoder(nn.Module):
    """
    EfficientNetV2-S backbone.
    Input: 4 channels (3 RGB slices + 1 Binary Mask).
    Output: Feature vector (Config.ENC_FEATURE_DIM).
    """

    def __init__(self, backbone_name=Config.ENC_BACKBONE, pretrained=True):
        super(FractureEncoder, self).__init__()

        # Create model using timm
        # in_chans=4 allows timm to automatically adapt the first layer weights
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # No classification head, just features
            in_chans=Config.ENC_IN_CHANNELS,
            global_pool="",  # We will handle pooling or return spatial maps if needed.
            # However, for sequence modeling, we usually want a 1D vector per slice.
            # Setting global_pool='' returns (B, C, H, W).
            # Setting global_pool='avg' returns (B, C).
        )

        # Check output features dimension
        # We force global pooling to get a vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Determine feature dim dynamically
        dummy_input = torch.randn(1, Config.ENC_IN_CHANNELS, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            # if global_pool='', features is (1, C, H, W)
            if len(features.shape) == 4:
                features = self.global_pool(features)
                features = self.flatten(features)
            self.feature_dim = features.shape[1]

    def forward(self, x):
        """
        x: (B, 4, H, W)
        Returns: (B, Feature_Dim)
        """
        x = self.backbone(x)

        if len(x.shape) == 4:
            x = self.global_pool(x)
            x = self.flatten(x)

        return x


# -----------------------------------------------------------------------------
# Stage 3: Anatomically-Embedded Transformer Aggregator
# -----------------------------------------------------------------------------
class AnatomicalTransformer(nn.Module):
    """
    Transformer Encoder that aggregates slice features.
    Uses Anatomical Embeddings (C1-C7) to provide semantic positional information.
    """

    def __init__(self):
        super(AnatomicalTransformer, self).__init__()

        self.hidden_dim = Config.AGG_HIDDEN_DIM
        self.input_dim = Config.ENC_FEATURE_DIM

        # 1. Feature Projection
        self.feature_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.layer_norm_input = nn.LayerNorm(self.hidden_dim)

        # 2. Anatomical Embedding
        # 0=Background/Unknown, 1-7=C1-C7. Total 8 embeddings.
        self.anat_embedding = nn.Embedding(8, self.hidden_dim)

        # 3. Positional Embedding (Learnable)
        # For relative position in the stack (0 to MaxSeqLen)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, Config.AGG_MAX_SEQ_LEN, self.hidden_dim)
        )

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=Config.AGG_NUM_HEADS,
            dim_feedforward=self.hidden_dim * 4,
            dropout=Config.AGG_DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.AGG_NUM_LAYERS
        )

        # 5. Classification Head
        # Predicts 8 logits: C1, C2, C3, C4, C5, C6, C7, Patient_Overall
        self.classifier = nn.Linear(self.hidden_dim, 8)

        # Dropout
        self.dropout = nn.Dropout(Config.AGG_DROPOUT)

    def forward(self, x, anat_ids, mask=None):
        """
        x: (B, Seq_Len, Input_Dim) - Slice features
        anat_ids: (B, Seq_Len) - Integer IDs (0-7)
        mask: (B, Seq_Len) - 1 for valid slices, 0 for padding
        """
        B, L, _ = x.shape

        # Project features
        x = self.feature_proj(x)  # (B, L, Hidden)
        x = self.layer_norm_input(x)

        # Add Anatomical Embeddings
        anat_emb = self.anat_embedding(anat_ids)  # (B, L, Hidden)
        x = x + anat_emb

        # Add Positional Embeddings
        # Slice to current sequence length
        pos_emb = self.pos_embedding[:, :L, :]
        x = x + pos_emb

        x = self.dropout(x)

        # Create padding mask for Transformer
        # PyTorch Transformer src_key_padding_mask expects True for padded positions
        if mask is not None:
            # mask is 1 for valid, 0 for pad.
            # We need boolean mask where True = Pad.
            src_key_padding_mask = mask == 0
        else:
            src_key_padding_mask = None

        # Transformer Pass
        # Output: (B, L, Hidden)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Aggregation
        # We use Global Average Pooling over valid slices
        if mask is not None:
            # Zero out padded positions to be safe
            mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
            x = x * mask_expanded
            sum_features = x.sum(dim=1)  # (B, Hidden)
            count_features = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            pooled = sum_features / count_features
        else:
            pooled = x.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)  # (B, 8)

        return logits
