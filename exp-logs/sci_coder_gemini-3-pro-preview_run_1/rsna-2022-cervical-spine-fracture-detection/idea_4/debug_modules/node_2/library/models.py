import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from library.config import Config

# =========================================================================
# Stage 1: Segmentation U-Net (ResNet18 Backbone)
# =========================================================================


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


class SegmentationUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(SegmentationUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Load ResNet18 encoder
        # We use standard weights if available, otherwise no weights (though pretrained is preferred)
        # Using 'DEFAULT' for best available weights in newer torchvision versions, or True for older
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except AttributeError:
            weights = True  # Fallback for older torchvision

        resnet = models.resnet18(weights=weights)

        # Encoder layers
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Decoder layers
        # Layer 4 output is 512 channels.
        # We upsample and concat with Layer 3 (256).
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(512, 256)  # 256 (up) + 256 (skip) = 512 in

        # Upsample and concat with Layer 2 (128)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)  # 128 (up) + 128 (skip) = 256 in

        # Upsample and concat with Layer 1 (64)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)  # 64 (up) + 64 (skip) = 128 in

        # Upsample and concat with Initial Conv (64) - Note: Inc is before pooling
        # Initial conv output size is H/2, W/2. Layer 1 is H/4, W/4.
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)  # 64 (up) + 64 (skip) = 128 in

        # Final upsample to original size
        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)  # 64, H/2, W/2
        x1 = self.pool(x0)  # 64, H/4, W/4
        x2 = self.layer1(x1)  # 64, H/4, W/4
        x3 = self.layer2(x2)  # 128, H/8, W/8
        x4 = self.layer3(x3)  # 256, H/16, W/16
        x5 = self.layer4(x4)  # 512, H/32, W/32

        # Decoder
        u1 = self.up1(x5)
        # Resize if necessary due to odd padding in encoder
        if u1.shape != x4.shape:
            u1 = F.interpolate(
                u1, size=x4.shape[2:], mode="bilinear", align_corners=True
            )
        d1 = self.dec1(torch.cat([u1, x4], dim=1))

        u2 = self.up2(d1)
        if u2.shape != x3.shape:
            u2 = F.interpolate(
                u2, size=x3.shape[2:], mode="bilinear", align_corners=True
            )
        d2 = self.dec2(torch.cat([u2, x3], dim=1))

        u3 = self.up3(d2)
        if u3.shape != x2.shape:
            u3 = F.interpolate(
                u3, size=x2.shape[2:], mode="bilinear", align_corners=True
            )
        d3 = self.dec3(torch.cat([u3, x2], dim=1))

        u4 = self.up4(d3)
        if u4.shape != x0.shape:
            u4 = F.interpolate(
                u4, size=x0.shape[2:], mode="bilinear", align_corners=True
            )
        d4 = self.dec4(torch.cat([u4, x0], dim=1))

        u5 = self.up5(d4)
        # Final interpolation to input size
        if u5.shape[2:] != x.shape[2:]:
            u5 = F.interpolate(
                u5, size=x.shape[2:], mode="bilinear", align_corners=True
            )

        logits = self.outc(u5)
        return logits


# =========================================================================
# Stage 2: Mask-Conditioned Slice Encoder (2.5D CNN)
# =========================================================================


class MaskConditionedCNN(nn.Module):
    def __init__(self, backbone_name=Config.ENCODER_BACKBONE, pretrained=True):
        super(MaskConditionedCNN, self).__init__()

        # Create backbone with 4 input channels (3 RGB + 1 Mask)
        # num_classes=0 means we get the feature vector, not logits
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=Config.ENCODER_IN_CHANNELS,
            num_classes=0,
        )

        # Get feature dimension dynamically
        # Create a dummy input to check output shape
        with torch.no_grad():
            dummy = torch.randn(1, Config.ENCODER_IN_CHANNELS, 256, 256)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]

        # Projection head for dimensionality reduction (optional but good for RNN)
        # Config suggests ENCODER_HIDDEN_DIM = 256
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, Config.ENCODER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classification head for Phase 2 (Slice-level binary classification)
        self.classifier = nn.Linear(Config.ENCODER_HIDDEN_DIM, 1)

    def forward_features(self, x):
        """Returns the projected feature vector."""
        x = self.backbone(x)
        x = self.projection(x)
        return x

    def forward(self, x):
        """Returns logits for binary classification (Phase 2 training)."""
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits


# =========================================================================
# Stage 3: Attentional Sequence Aggregator (Bi-GRU + Attention)
# =========================================================================


class GlobalAttention(nn.Module):
    """
    Computes a weighted average of sequence vectors based on learned attention scores.
    """

    def __init__(self, input_dim):
        super(GlobalAttention, self).__init__()
        # Simple attention: Linear projection to scalar score
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        """
        x: (Batch, Seq_Len, Input_Dim)
        mask: (Batch, Seq_Len) - Optional mask for padding (1=valid, 0=pad)
        """
        # Calculate scores: (Batch, Seq_Len, 1)
        scores = self.attention_layer(x)

        if mask is not None:
            # Mask padding positions with large negative value
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        # Softmax over sequence dimension
        weights = F.softmax(scores, dim=1)  # (Batch, Seq_Len, 1)

        # Weighted sum: (Batch, Input_Dim)
        context = torch.sum(x * weights, dim=1)

        return context, weights


class AttentionalRNN(nn.Module):
    def __init__(
        self,
        input_dim=Config.ENCODER_HIDDEN_DIM,
        hidden_dim=Config.SEQ_HIDDEN_DIM,
        num_layers=Config.SEQ_NUM_LAYERS,
    ):
        super(AttentionalRNN, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=Config.SEQ_DROPOUT if num_layers > 1 else 0,
        )

        # Bidirectional doubles the hidden dimension
        gru_out_dim = hidden_dim * 2

        self.attention = GlobalAttention(gru_out_dim)

        # Final classification head for 8 targets
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 8),  # C1-C7 + patient_overall
        )

    def forward(self, x, mask=None):
        """
        x: (Batch, Seq_Len, Feature_Dim)
        mask: (Batch, Seq_Len)
        """
        # Pass through GRU
        # output: (Batch, Seq_Len, Hidden_Dim * 2)
        gru_out, _ = self.gru(x)

        # Aggregation via Attention
        # context: (Batch, Hidden_Dim * 2)
        context, attn_weights = self.attention(gru_out, mask)

        # Classification
        logits = self.classifier(context)

        return logits
