import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: (Conv -> BN -> ReLU) * 2
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


class AttentionGate(nn.Module):
    """
    Attention Gate to filter features from the skip connection (x)
    using the coarser gating signal (g) from the decoder.
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # W_g: Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        # W_x: Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        # psi: Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Upsample g1 to match x1 spatial dimensions
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(
                g1, size=x1.size()[2:], mode="bilinear", align_corners=False
            )

        # Add and activate
        psi = self.relu(g1 + x1)

        # Compute attention map
        psi = self.psi(psi)

        # Filter x
        return x * psi


class DecoderBlock(nn.Module):
    """
    Decoder block with optional Attention Gate.
    Upsamples input, gates the skip connection, concatenates, and refines.
    """

    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # Upsampling layer
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        # Attention Gate
        if use_attention:
            self.att = AttentionGate(
                F_g=in_channels, F_l=skip_channels, F_int=in_channels // 2
            )

        # Convolutional block after concatenation
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        # x: Input from deeper layer (coarse)
        # skip: Skip connection from encoder (fine)

        if self.use_attention:
            # Apply attention to skip connection using x as gate
            skip = self.att(g=x, x=skip)

        # Upsample x
        x = self.up(x)

        # Handle padding/size mismatch
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(
                x, size=skip.size()[2:], mode="bilinear", align_corners=False
            )

        # Concatenate
        x = torch.cat([x, skip], dim=1)

        # Refine
        x = self.conv(x)
        return x


class AttentionUNet25D(nn.Module):
    """
    2.5D U-Net with EfficientNet Encoder and Attention Gates.
    """

    def __init__(self):
        super().__init__()

        # Encoder: EfficientNet-B0
        # features_only=True returns intermediate feature maps
        self.encoder = timm.create_model(
            Config.ENCODER_NAME,
            pretrained=(Config.ENCODER_WEIGHTS == "imagenet"),
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Determine channel counts dynamically
        with torch.no_grad():
            dummy = torch.randn(1, Config.IN_CHANNELS, 256, 256)
            features = self.encoder(dummy)
            dims = [f.shape[1] for f in features]
            # Expected dims for B0: [16, 24, 40, 112, 320]

        # Decoder Path
        # We start from the deepest feature map (dims[4]) and work up

        # Block 4: Input dims[4] (320), Skip dims[3] (112) -> Out 256
        self.dec4 = DecoderBlock(
            dims[4], dims[3], 256, use_attention=Config.USE_ATTENTION_GATES
        )

        # Block 3: Input 256, Skip dims[2] (40) -> Out 128
        self.dec3 = DecoderBlock(
            256, dims[2], 128, use_attention=Config.USE_ATTENTION_GATES
        )

        # Block 2: Input 128, Skip dims[1] (24) -> Out 64
        self.dec2 = DecoderBlock(
            128, dims[1], 64, use_attention=Config.USE_ATTENTION_GATES
        )

        # Block 1: Input 64, Skip dims[0] (16) -> Out 32
        self.dec1 = DecoderBlock(
            64, dims[0], 32, use_attention=Config.USE_ATTENTION_GATES
        )

        # Final Upsampling Block (Stride 2 -> Stride 1)
        # Input 32 -> Upsample -> Conv -> Classes
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, Config.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc_feats = self.encoder(x)
        # enc_feats indices: 0 (1/2), 1 (1/4), 2 (1/8), 3 (1/16), 4 (1/32)

        # Decoder
        d4 = self.dec4(enc_feats[4], enc_feats[3])
        d3 = self.dec3(d4, enc_feats[2])
        d2 = self.dec2(d3, enc_feats[1])
        d1 = self.dec1(d2, enc_feats[0])

        # Final Output
        out = self.final_up(d1)

        # Ensure output size matches input size
        if out.size()[2:] != x.size()[2:]:
            out = F.interpolate(
                out, size=x.size()[2:], mode="bilinear", align_corners=False
            )

        out = self.final_conv(out)

        return out
