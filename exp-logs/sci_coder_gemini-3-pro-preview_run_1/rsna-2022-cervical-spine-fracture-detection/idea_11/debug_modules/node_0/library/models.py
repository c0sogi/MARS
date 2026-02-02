import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class ConvBlock(nn.Module):
    """
    Helper module for UNet Decoder.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetLocalizer(nn.Module):
    """
    Stage 1: Multi-Class Anatomical Localizer (2D U-Net).
    Uses a ResNet18 backbone for the encoder.
    Outputs: 8 channels (Background + C1-C7).
    """

    def __init__(self, num_classes=Config.STAGE1_NUM_CLASSES, pretrained=True):
        super().__init__()

        # Encoder: ResNet18
        # We need to access intermediate layers for skip connections.
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.encoder = models.resnet18(weights=weights)

        # Modify first layer to accept 1 channel (DICOM) instead of 3
        self.encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Decoder Layers
        # Input to decoder is layer4 output (512 channels)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256 + 256, 256)  # Cat with layer3 (256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128)  # Cat with layer2 (128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)  # Cat with layer1 (64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # Skip connection from before maxpool (which is after relu, 64ch)
        self.dec1 = ConvBlock(64 + 64, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(32, 32)  # Final resolution recovery

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)  # Stride 2, 64ch

        x1 = self.encoder.maxpool(x0)  # Stride 4, 64ch
        x1 = self.encoder.layer1(x1)  # Stride 4, 64ch

        x2 = self.encoder.layer2(x1)  # Stride 8, 128ch
        x3 = self.encoder.layer3(x2)  # Stride 16, 256ch
        x4 = self.encoder.layer4(x3)  # Stride 32, 512ch

        # Decoder
        d4 = self.up4(x4)  # -> 256ch, Stride 16
        if d4.size()[2:] != x3.size()[2:]:
            d4 = F.interpolate(
                d4, size=x3.size()[2:], mode="bilinear", align_corners=True
            )
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)  # -> 128ch, Stride 8
        if d3.size()[2:] != x2.size()[2:]:
            d3 = F.interpolate(
                d3, size=x2.size()[2:], mode="bilinear", align_corners=True
            )
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)  # -> 64ch, Stride 4
        if d2.size()[2:] != x1.size()[2:]:
            d2 = F.interpolate(
                d2, size=x1.size()[2:], mode="bilinear", align_corners=True
            )
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)  # -> 64ch, Stride 2
        if d1.size()[2:] != x0.size()[2:]:
            d1 = F.interpolate(
                d1, size=x0.size()[2:], mode="bilinear", align_corners=True
            )
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)  # -> 32ch, Stride 1
        if d0.size()[2:] != x.size()[2:]:
            d0 = F.interpolate(
                d0, size=x.size()[2:], mode="bilinear", align_corners=True
            )
        d0 = self.dec0(d0)

        logits = self.final_conv(d0)
        return logits


class DualStreamEncoder(nn.Module):
    """
    Stage 2: Dual-Stream Feature Encoder.
    Branch A (Local): High-res crop + Mask (2 channels).
    Branch B (Global): Resized full slice (1 channel).
    """

    def __init__(self, backbone_name=Config.STAGE2_BACKBONE, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None

        # --- Local Branch ---
        self.local_backbone = models.resnet18(weights=weights)
        # Input: Image (1) + Mask (1) = 2 channels
        self.local_backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.local_backbone.fc = nn.Identity()

        # --- Global Branch ---
        self.global_backbone = models.resnet18(weights=weights)
        # Input: Image (1) = 1 channel
        self.global_backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.global_backbone.fc = nn.Identity()

        # Output dimension of ResNet18 before FC is 512
        self.feature_dim = 512 + 512

    def forward(self, x_local, x_global):
        """
        x_local: (B, 2, H, W)
        x_global: (B, 1, H, W)
        """
        feat_local = self.local_backbone(x_local)
        feat_global = self.global_backbone(x_global)

        # Fusion
        fused = torch.cat([feat_local, feat_global], dim=1)  # (B, 1024)
        return fused


class AnatomicalGRU(nn.Module):
    """
    Stage 3: Anatomically-Indexed Recurrent Aggregator.
    Input: Sequence of fused features + Anatomical Profiles.
    """

    def __init__(
        self,
        input_dim=1024 + 8,  # 1024 visual + 8 anatomical
        hidden_dim=Config.STAGE3_HIDDEN_DIM,
        num_layers=Config.STAGE3_NUM_LAYERS,
        dropout=Config.STAGE3_DROPOUT,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bi-Directional GRU
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Output dim of GRU is 2 * hidden_dim
        self.gru_out_dim = hidden_dim * 2

        # Attention Mechanism for Soft Anatomical Pooling
        # We learn a projection from GRU state to a score for each of the 7 vertebrae
        self.attention = nn.Linear(self.gru_out_dim, 7)

        # Classification Heads
        # 7 Vertebrae Heads
        # Each takes the specific context vector for that vertebra
        self.vert_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.gru_out_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                )
                for _ in range(7)
            ]
        )

        # Patient Head
        # Takes concatenated context vectors (7 * gru_out_dim)
        self.patient_head = nn.Sequential(
            nn.Linear(self.gru_out_dim * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x, lengths=None):
        """
        x: (B, T, input_dim).
           Assumes last 8 channels are anatomical probabilities (Background, C1..C7).
        lengths: (B,) Sequence lengths for packing.
        """
        # Extract Anatomical Profiles (Probabilities)
        # Assuming last 8 columns are [Background, C1, C2, ..., C7]
        anat_probs = x[:, :, -8:]

        # GRU Processing
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            gru_out_packed, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out_packed, batch_first=True
            )
        else:
            gru_out, _ = self.gru(x)

        # gru_out: (B, T, 2*hidden)

        # --- Soft Anatomical Pooling ---
        # 1. Compute learned attention scores
        attn_scores = self.attention(gru_out)  # (B, T, 7)

        # 2. Get Anatomical Guidance (Log Probs)
        # We care about C1-C7, which are indices 1-7 in the 8-class profile
        guidance = anat_probs[:, :, 1:]  # (B, T, 7)

        # Clamp for log stability
        guidance = torch.clamp(guidance, min=1e-7, max=1.0)
        log_guidance = torch.log(guidance)

        # 3. Combine
        combined_scores = attn_scores + log_guidance

        # 4. Masking for padding (if lengths provided)
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1)  # (B, T, 1)
            combined_scores = combined_scores.masked_fill(~mask, -1e9)

        # 5. Softmax over Time dimension
        weights = F.softmax(combined_scores, dim=1)  # (B, T, 7)

        # 6. Weighted Sum (Einstein summation: btd, btk -> bkd)
        context_vectors = torch.einsum("btd,btk->bkd", gru_out, weights)

        # --- Classification ---

        # Vertebrae Predictions
        vert_preds = []
        for k in range(7):
            ctx = context_vectors[:, k, :]  # (B, D)
            logit = self.vert_heads[k](ctx)
            vert_preds.append(logit)

        vert_preds = torch.cat(vert_preds, dim=1)  # (B, 7)

        # Patient Prediction
        patient_ctx = context_vectors.view(x.size(0), -1)
        patient_pred = self.patient_head(patient_ctx)  # (B, 1)

        # Concatenate all predictions: C1..C7, Patient
        all_preds = torch.cat([vert_preds, patient_pred], dim=1)

        return all_preds
