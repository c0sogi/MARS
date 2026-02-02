import torch
import torch.nn as nn
import torchvision
import math
from library.config import Config


class ResNetEncoder(nn.Module):
    """
    ResNet-18 based feature extractor with anisotropic downsampling.
    Converts (N, 1, H, W) -> (N, 512, W/4).
    """

    def __init__(self, backbone_name="resnet18"):
        super().__init__()
        if backbone_name == "resnet18":
            # Use standard ResNet18; weights=None for a clean baseline/offline safety
            self.backbone = torchvision.models.resnet18(weights=None)
        else:
            raise NotImplementedError(
                "Only resnet18 is supported for this implementation."
            )

        # 1. Modify input layer for 1-channel grayscale images
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # 2. Modify strides for Anisotropic Downsampling
        # We want to collapse Height (128 -> 1) but preserve Width (W -> W/4).
        # Standard ResNet downsamples by 2 in both dims at: conv1, maxpool, layer2, layer3, layer4.
        # Total downsample: 2^5 = 32.
        # We keep conv1/maxpool strides (total 4x).
        # We change layers 2, 3, 4 to stride (2, 1) -> Downsample H by 2, W by 1.
        # Final H downsample: 2(conv1)*2(pool)*2(l2)*2(l3)*2(l4) = 32.
        # Final W downsample: 2(conv1)*2(pool)*1(l2)*1(l3)*1(l4) = 4.

        # Layer 2
        self.backbone.layer2[0].conv1.stride = (2, 1)
        if self.backbone.layer2[0].downsample is not None:
            self.backbone.layer2[0].downsample[0].stride = (2, 1)

        # Layer 3
        self.backbone.layer3[0].conv1.stride = (2, 1)
        if self.backbone.layer3[0].downsample is not None:
            self.backbone.layer3[0].downsample[0].stride = (2, 1)

        # Layer 4
        self.backbone.layer4[0].conv1.stride = (2, 1)
        if self.backbone.layer4[0].downsample is not None:
            self.backbone.layer4[0].downsample[0].stride = (2, 1)

    def forward(self, x):
        # x: (N, 1, 128, W)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # x: (N, 512, 4, W/4)
        # Collapse the remaining vertical dimension (4 pixels)
        x = x.mean(dim=2)
        # x: (N, 512, W/4)
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for Transformer.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer to save in state_dict but not train
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (N, Seq_Len, D_Model)
        seq_len = x.size(1)
        # Handle cases where input might be wider than initialized max_len (rare but possible with dynamic padding)
        if seq_len > self.pe.size(1):
            x = x[:, : self.pe.size(1), :]
            pe_slice = self.pe
        else:
            pe_slice = self.pe[:, :seq_len, :]

        x = x + pe_slice
        return self.dropout(x)


class CNNTransformerCTC(nn.Module):
    """
    End-to-End Model: CNN Encoder + Transformer Context + CTC Head.
    """

    def __init__(self):
        super().__init__()

        # 1. Visual Encoder
        self.encoder = ResNetEncoder(Config.CNN_BACKBONE)

        # 2. Feature Projection
        # ResNet18 output channels = 512. Project to Transformer D_MODEL.
        self.projection = nn.Linear(512, Config.D_MODEL)

        # 3. Positional Encoding
        # Max width buffer. 2500 covers images up to 10,000 pixels wide (stride 4).
        self.pos_encoder = PositionalEncoding(
            Config.D_MODEL, max_len=2500, dropout=Config.DROPOUT
        )

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.NUM_ENCODER_LAYERS
        )

        # 5. CTC Classification Head
        self.fc = nn.Linear(Config.D_MODEL, Config.VOCAB_SIZE)

    def forward(self, images):
        """
        Args:
            images (torch.Tensor): Batch of images (N, 1, H, W)

        Returns:
            logits (torch.Tensor): CTC logits (N, Seq_Len, Vocab_Size)
        """
        # CNN Feature Extraction
        # (N, 512, W_seq)
        features = self.encoder(images)

        # Permute for Transformer (Batch, Seq, Feature)
        # (N, W_seq, 512)
        features = features.permute(0, 2, 1)

        # Project to embedding dim
        # (N, W_seq, D_MODEL)
        x = self.projection(features)

        # Add Positional Info
        x = self.pos_encoder(x)

        # Apply Transformer Context
        # (N, W_seq, D_MODEL)
        x = self.transformer(x)

        # Project to Vocabulary
        # (N, W_seq, VOCAB_SIZE)
        logits = self.fc(x)

        return logits
