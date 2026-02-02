import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library import config


class LayerNorm2d(nn.Module):
    """
    LayerNorm for (N, C, H, W) tensors.
    ConvNeXt uses channel-wise LN.
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block with 7x7 Depthwise Conv.
    Maintains the 'Linearity Prior' and large receptive field.
    """

    def __init__(self, in_channels, out_channels, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels
        )
        self.norm = LayerNorm2d(in_channels)
        self.pwconv1 = nn.Linear(
            in_channels, expansion * in_channels
        )  # Implemented as 1x1 conv usually, but here manual
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * in_channels, out_channels)

        # Handling dimension mismatch for residual
        self.skip_conv = nn.Identity()
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        shortcut = self.skip_conv(x)

        # 7x7 Depthwise
        x = self.dwconv(x)
        x = self.norm(x)

        # Pointwise 1x1 (implemented with permute + linear for canonical ConvNeXt style)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        return x + shortcut


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    """

    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        self.convs = nn.ModuleList()
        # 1x1 Conv
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        # 3x3 Convs with rates
        for rate in rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        # Global Pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Global pooling upsampled to size
        g = self.global_pool(x)
        g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        res.append(g)

        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlock(nn.Module):
    """
    Decoder block with Multi-Scale Input Injection.
    Upsamples -> Concatenates (Skip + Upsampled + Injected Input) -> ConvNeXtBlock -> SCSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels, input_channels=6):
        super().__init__()
        # Calculate total channels after concatenation
        # Input comes from previous stage (in_channels)
        # Skip comes from encoder (skip_channels)
        # Injected Input comes from raw image (input_channels)
        total_in_channels = in_channels + skip_channels + input_channels

        # We use a 1x1 conv to reduce channels before the heavy ConvNeXt block
        self.reduce = nn.Conv2d(
            total_in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv = ConvNeXtBlock(out_channels, out_channels)
        self.attn = SCSEModule(out_channels)

    def forward(self, x, skip=None, raw_input=None):
        """
        x: Features from previous decoder stage (lower resolution)
        skip: Features from encoder (same resolution as target)
        raw_input: Original 6-channel input image
        """
        # 1. Upsample x
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # 2. Prepare Skip Connection
        if skip is not None:
            # Handle potential padding issues if dimensions don't match perfectly
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
        else:
            # If no skip (e.g., last stages), create empty tensor or just handle logic
            # For simplicity, we assume skip is handled by list logic in UNet,
            # but if skip is None, we just concat x and raw_input
            pass

        # 3. Input Injection
        # Downsample raw input to match current resolution x
        target_size = x.shape[2:]
        injected = F.adaptive_avg_pool2d(raw_input, target_size)

        # 4. Concatenate
        if skip is not None:
            out = torch.cat([x, skip, injected], dim=1)
        else:
            out = torch.cat([x, injected], dim=1)

        # 5. Process
        out = self.reduce(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.attn(out)

        return out


class ConvNeXtUNet(nn.Module):
    def __init__(
        self,
        backbone_name="convnext_tiny",
        pretrained=True,
        in_channels=config.MODEL_INPUT_CHANNELS,
        num_classes=1,
    ):
        super().__init__()

        # 1. Input Adapter
        # Maps 6 channels to 3 channels for the pretrained backbone
        self.stem_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)

        # 2. Encoder (Backbone)
        # ConvNeXt Tiny feature channels: [96, 192, 384, 768] at strides [4, 8, 16, 32]
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        encoder_channels = self.encoder.feature_info.channels()  # [96, 192, 384, 768]

        # 3. Bridge
        self.aspp = ASPP(encoder_channels[3], 256)

        # 4. Decoder
        # Stage 4: Stride 32 -> 16. In: Bridge(256), Skip: Enc[2](384). Out: 256
        self.dec4 = DecoderBlock(
            in_channels=256,
            skip_channels=encoder_channels[2],
            out_channels=256,
            input_channels=in_channels,
        )

        # Stage 3: Stride 16 -> 8. In: Dec4(256), Skip: Enc[1](192). Out: 128
        self.dec3 = DecoderBlock(
            in_channels=256,
            skip_channels=encoder_channels[1],
            out_channels=128,
            input_channels=in_channels,
        )

        # Stage 2: Stride 8 -> 4. In: Dec3(128), Skip: Enc[0](96). Out: 64
        self.dec2 = DecoderBlock(
            in_channels=128,
            skip_channels=encoder_channels[0],
            out_channels=64,
            input_channels=in_channels,
        )

        # Stage 1: Stride 4 -> 2. In: Dec2(64), Skip: None. Out: 32
        # ConvNeXt stem is stride 4, so no skip at stride 2.
        self.dec1 = DecoderBlock(
            in_channels=64, skip_channels=0, out_channels=32, input_channels=in_channels
        )

        # Stage 0: Stride 2 -> 1. In: Dec1(32), Skip: None. Out: 16
        self.dec0 = DecoderBlock(
            in_channels=32, skip_channels=0, out_channels=16, input_channels=in_channels
        )

        # 5. Head
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        input_image = x  # Keep reference for injection

        # Adapter
        x_adapt = self.stem_adapter(x)

        # Encoder
        enc_feats = self.encoder(x_adapt)
        # enc_feats[0]: stride 4, 96
        # enc_feats[1]: stride 8, 192
        # enc_feats[2]: stride 16, 384
        # enc_feats[3]: stride 32, 768

        # Bridge
        x = self.aspp(enc_feats[3])  # Stride 32, 256 ch

        # Decoder
        x = self.dec4(x, enc_feats[2], input_image)  # -> Stride 16
        x = self.dec3(x, enc_feats[1], input_image)  # -> Stride 8
        x = self.dec2(x, enc_feats[0], input_image)  # -> Stride 4
        x = self.dec1(x, None, input_image)  # -> Stride 2
        x = self.dec0(x, None, input_image)  # -> Stride 1

        # Head
        logits = self.head(x)

        # Ensure output size matches input size exactly
        if logits.shape[2:] != input_image.shape[2:]:
            logits = F.interpolate(
                logits, size=input_image.shape[2:], mode="bilinear", align_corners=False
            )

        return logits
