import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.utils import compute_dice_score, get_logger

logger = get_logger("Model")


class ConvBnRelu(nn.Module):
    """
    Standard Convolution -> BatchNorm -> ReLU block.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """
    U-Net++ Decoder Block.
    Concatenates input from the lower layer (upsampled) with skip connections
    from the same layer, then applies convolutions.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # in_ch: channels from the layer below (after upsampling)
        # skip_ch: total channels from all skip connections
        self.conv1 = ConvBnRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)

    def forward(self, x, skips):
        """
        Args:
            x: Input tensor from the lower layer (to be upsampled).
            skips: List of tensors from the same layer (skip connections).
        """
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Ensure shapes match (handle odd dimensions if any)
        if skips:
            ref = skips[0]
            if x.shape[2:] != ref.shape[2:]:
                x = F.interpolate(
                    x, size=ref.shape[2:], mode="bilinear", align_corners=False
                )

            # Concatenate
            x = torch.cat([x] + skips, dim=1)

        return self.conv2(self.conv1(x))


class HubmapUnetPlusPlus(nn.Module):
    """
    U-Net++ with ConvNeXt Encoder.
    """

    def __init__(
        self,
        encoder_name="convnext_base",
        in_channels=3,
        classes=1,
        deep_supervision=True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # 1. Encoder (ConvNeXt Base)
        # Features: [stem(s4), stage1(s8), stage2(s16), stage3(s32)]
        # Channels: [128, 256, 512, 1024]
        self.encoder = timm.create_model(
            encoder_name, pretrained=True, features_only=True, in_chans=in_channels
        )

        # Get channel counts automatically
        dummy = torch.randn(1, in_channels, 256, 256)
        feats = self.encoder(dummy)
        enc_ch = [f.shape[1] for f in feats]  # e.g., [128, 256, 512, 1024]

        # Define Decoder Channels (can be tuned)
        # We keep them proportional to encoder to maintain capacity
        dec_ch = enc_ch  # [128, 256, 512, 1024]

        # 2. Decoder Blocks
        # Notation: node_L_j where L is layer (row), j is dense block index (col)
        # L0 corresponds to enc_ch[0] (stride 4)
        # L1 corresponds to enc_ch[1] (stride 8)
        # L2 corresponds to enc_ch[2] (stride 16)
        # L3 corresponds to enc_ch[3] (stride 32)

        # Column 1 (j=1)
        # x2_1: up(x3_0) + x2_0
        self.conv2_1 = DecoderBlock(
            in_ch=enc_ch[3], skip_ch=enc_ch[2], out_ch=dec_ch[2]
        )
        # x1_1: up(x2_0) + x1_0
        self.conv1_1 = DecoderBlock(
            in_ch=enc_ch[2], skip_ch=enc_ch[1], out_ch=dec_ch[1]
        )
        # x0_1: up(x1_0) + x0_0
        self.conv0_1 = DecoderBlock(
            in_ch=enc_ch[1], skip_ch=enc_ch[0], out_ch=dec_ch[0]
        )

        # Column 2 (j=2)
        # x1_2: up(x2_1) + x1_0 + x1_1
        self.conv1_2 = DecoderBlock(
            in_ch=dec_ch[2], skip_ch=enc_ch[1] + dec_ch[1], out_ch=dec_ch[1]
        )
        # x0_2: up(x1_1) + x0_0 + x0_1
        self.conv0_2 = DecoderBlock(
            in_ch=dec_ch[1], skip_ch=enc_ch[0] + dec_ch[0], out_ch=dec_ch[0]
        )

        # Column 3 (j=3)
        # x0_3: up(x1_2) + x0_0 + x0_1 + x0_2
        self.conv0_3 = DecoderBlock(
            in_ch=dec_ch[1], skip_ch=enc_ch[0] + dec_ch[0] * 2, out_ch=dec_ch[0]
        )

        # 3. Segmentation Heads
        # All heads attached to L0 nodes (stride 4)
        self.final_conv = nn.Conv2d(dec_ch[0], classes, kernel_size=1)

        if self.deep_supervision:
            self.seg0_1 = nn.Conv2d(dec_ch[0], classes, kernel_size=1)
            self.seg0_2 = nn.Conv2d(dec_ch[0], classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[2:]

        # Encoder
        features = self.encoder(x)
        x0_0, x1_0, x2_0, x3_0 = features[0], features[1], features[2], features[3]

        # Decoder Column 1
        x2_1 = self.conv2_1(x3_0, [x2_0])
        x1_1 = self.conv1_1(x2_0, [x1_0])
        x0_1 = self.conv0_1(x1_0, [x0_0])

        # Decoder Column 2
        x1_2 = self.conv1_2(x2_1, [x1_0, x1_1])
        x0_2 = self.conv0_2(x1_1, [x0_0, x0_1])

        # Decoder Column 3
        x0_3 = self.conv0_3(x1_2, [x0_0, x0_1, x0_2])

        # Heads
        logits_0_3 = self.final_conv(x0_3)

        # Upsample to original input size (Stride 4 -> Stride 1)
        out_0_3 = F.interpolate(
            logits_0_3, size=input_shape, mode="bilinear", align_corners=False
        )

        if self.deep_supervision and self.training:
            logits_0_2 = self.seg0_2(x0_2)
            logits_0_1 = self.seg0_1(x0_1)

            out_0_2 = F.interpolate(
                logits_0_2, size=input_shape, mode="bilinear", align_corners=False
            )
            out_0_1 = F.interpolate(
                logits_0_1, size=input_shape, mode="bilinear", align_corners=False
            )

            return [out_0_3, out_0_2, out_0_1]

        return out_0_3


class DeepSupervisionLoss(nn.Module):
    """
    Combined BCE + Dice Loss with support for Deep Supervision.
    """

    def __init__(self, weights=[1.0, 0.5, 0.25]):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # If preds is a list (Deep Supervision), compute weighted loss
        if isinstance(preds, list):
            loss = 0
            for i, pred in enumerate(preds):
                w = self.weights[i] if i < len(self.weights) else 0.0
                if w > 0:
                    loss += w * self._compute_single_loss(pred, targets)
            return loss
        else:
            return self._compute_single_loss(preds, targets)

    def _compute_single_loss(self, pred, target):
        bce = self.bce(pred, target.float().unsqueeze(1))

        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        dice = 1.0 - compute_dice_score(
            target.cpu().detach().numpy(),
            (pred_sigmoid > 0.5).cpu().detach().numpy().squeeze(),
        )
        # Differentiable Dice approximation for training
        smooth = 1e-5
        intersection = (pred_sigmoid * target.unsqueeze(1)).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.unsqueeze(1).sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)

        return 0.5 * bce + 0.5 * dice_loss.mean()


def build_model(
    encoder_name="convnext_base", in_channels=3, classes=1, pretrained=True
):
    """
    Factory function to build the model.
    """
    logger.info(f"Building U-Net++ with encoder: {encoder_name}")
    model = HubmapUnetPlusPlus(
        encoder_name=encoder_name,
        in_channels=in_channels,
        classes=classes,
        deep_supervision=True,
    )
    return model
