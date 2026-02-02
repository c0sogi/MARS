import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from library.config import Config
from library.utils import calculate_rmse, save_checkpoint, get_logger

# =========================================================================
# Model Components
# =========================================================================


class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 2D inputs (N, C, H, W).
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


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module.
    Splits spatial pooling into vertical and horizontal directions to preserve
    positional information.
    """

    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Factorize global pooling
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate and transform
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block with Coordinate Attention.
    Structure: 7x7 DWConv -> LN -> CA -> 1x1 Conv -> SiLU -> 1x1 Conv -> Residual
    """

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.ca = CoordinateAttention(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.SiLU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)

        # Apply Coordinate Attention
        x = self.ca(x)

        # Pointwise (Channel-last for Linear)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Captures multi-scale context at the bottleneck.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        modules = []
        # 1x1 Conv
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )
        # Dilated Convs
        rates = [6, 12, 18]
        for rate in rates:
            modules.append(
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
                    nn.ReLU(),
                )
            )
        # Global Pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            out = conv(x)
            if out.size(2) != x.size(2):
                out = F.interpolate(
                    out, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            res.append(out)
        res = torch.cat(res, dim=1)
        return self.project(res)


class CoConvNeXtUNet(nn.Module):
    """
    Coordinate ConvNeXt U-Net.
    Predicts the noise residual.
    """

    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super().__init__()

        # Encoder
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            LayerNorm2d(base_filters),
        )

        self.enc1 = ConvNeXtBlock(base_filters)
        self.down1 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=2, stride=2)

        self.enc2 = ConvNeXtBlock(base_filters * 2)
        self.down2 = nn.Conv2d(
            base_filters * 2, base_filters * 4, kernel_size=2, stride=2
        )

        self.enc3 = ConvNeXtBlock(base_filters * 4)
        self.down3 = nn.Conv2d(
            base_filters * 4, base_filters * 8, kernel_size=2, stride=2
        )

        # Bottleneck
        self.aspp = ASPP(base_filters * 8, base_filters * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            base_filters * 8, base_filters * 4, kernel_size=2, stride=2
        )
        self.reduce3 = nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=1)
        self.dec3 = ConvNeXtBlock(base_filters * 4)

        self.up2 = nn.ConvTranspose2d(
            base_filters * 4, base_filters * 2, kernel_size=2, stride=2
        )
        self.reduce2 = nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=1)
        self.dec2 = ConvNeXtBlock(base_filters * 2)

        self.up1 = nn.ConvTranspose2d(
            base_filters * 2, base_filters, kernel_size=2, stride=2
        )
        self.reduce1 = nn.Conv2d(base_filters * 2, base_filters, kernel_size=1)
        self.dec1 = ConvNeXtBlock(base_filters)

        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.stem(x)
        s1 = self.enc1(s1)

        s2 = self.down1(s1)
        s2 = self.enc2(s2)

        s3 = self.down2(s2)
        s3 = self.enc3(s3)

        b = self.down3(s3)
        b = self.aspp(b)

        # Decoder
        d3 = self.up3(b)
        if d3.size(2) != s3.size(2) or d3.size(3) != s3.size(3):
            d3 = F.interpolate(
                d3, size=s3.shape[2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.reduce3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.size(2) != s2.size(2) or d2.size(3) != s2.size(3):
            d2 = F.interpolate(
                d2, size=s2.shape[2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.reduce2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.size(2) != s1.size(2) or d1.size(3) != s1.size(3):
            d1 = F.interpolate(
                d1, size=s1.shape[2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.reduce1(d1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out


# =========================================================================
# Training & Inference Logic
# =========================================================================


def train_model(model, train_loader, val_loader, device):
    """
    Trains the model using AdamW, Cosine Annealing, and Early Stopping.
    """
    logger = get_logger("Train")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LEARNING_RATE
    )

    best_rmse = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        # Training Loop
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()

            # Predict noise residual
            noise_pred = model(noisy)

            # Target is noise (Noisy - Clean)
            noise_target = noisy - clean

            loss = criterion(noise_pred, noise_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation Loop
        model.eval()
        val_rmse = 0.0
        total_pixels = 0

        with torch.no_grad():
            for noisy, clean, _ in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)

                noise_pred = model(noisy)
                clean_pred = noisy - noise_pred
                clean_pred = torch.clamp(clean_pred, 0, 1)

                # RMSE Calculation
                mse = F.mse_loss(clean_pred, clean, reduction="sum")
                val_rmse += mse.item()
                total_pixels += clean.numel()

        val_rmse = np.sqrt(val_rmse / total_pixels)

        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val RMSE: {val_rmse:.6f}"
        )

        # Checkpoint
        is_best = val_rmse < best_rmse
        if is_best:
            best_rmse = val_rmse
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric": best_rmse,
                },
                is_best,
                Config.WORKING_DIR,
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered.")
            break


def predict_patch_tiled(model, image, patch_size, overlap_ratio, device):
    """
    Predicts image using tiled sliding window with overlap.
    image: (C, H, W) tensor
    """
    c, h, w = image.shape
    stride = int(patch_size * (1 - overlap_ratio))

    # Pad image to handle boundaries
    pad_h = (patch_size - h % stride) % stride
    pad_w = (patch_size - w % stride) % stride
    if h < patch_size:
        pad_h += patch_size - h
    if w < patch_size:
        pad_w += patch_size - w

    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    ph, pw = padded_image.shape[1], padded_image.shape[2]

    output_sum = torch.zeros_like(padded_image)
    output_count = torch.zeros_like(padded_image)

    patches = []
    coords = []

    # Collect patches
    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            patch = padded_image[:, y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            coords.append((y, x))

    # Batch process
    batch_size = Config.BATCH_SIZE
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i : i + batch_size]).to(device)
        with torch.no_grad():
            pred_noise = model(batch)
            # Clean = Noisy - Noise
            pred_clean = batch - pred_noise
            pred_clean = torch.clamp(pred_clean, 0, 1)

        for j, pred in enumerate(pred_clean):
            y, x = coords[i + j]
            output_sum[:, y : y + patch_size, x : x + patch_size] += pred.cpu()
            output_count[:, y : y + patch_size, x : x + patch_size] += 1.0

    output = output_sum / output_count
    return output[:, :h, :w]


def inference(model, test_loader, device):
    """
    Generates predictions for the test set using TTA and Tiled Inference.
    Saves results to submission.csv.
    """
    logger = get_logger("Inference")
    model.eval()

    results = []

    # TTA Transforms: Identity, HFlip, VFlip
    active_transforms = [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, [2]), lambda x: torch.flip(x, [2])),
        (lambda x: torch.flip(x, [1]), lambda x: torch.flip(x, [1])),
    ]

    with torch.no_grad():
        for noisy, img_id in tqdm(test_loader, disable=True):
            noisy = noisy[
                0
            ]  # Remove batch dim (Dataset returns (C,H,W), Loader adds batch (1,C,H,W))

            ensemble_pred = torch.zeros_like(noisy)

            for fwd, inv in active_transforms:
                aug_noisy = fwd(noisy)
                pred = predict_patch_tiled(
                    model, aug_noisy, Config.PATCH_SIZE, Config.OVERLAP_RATIO, device
                )
                pred = inv(pred)
                ensemble_pred += pred

            ensemble_pred /= len(active_transforms)

            # Formatting for submission
            img_id_str = img_id[0]
            vals = ensemble_pred[0].numpy().flatten()  # (H, W) -> Flat

            h, w = ensemble_pred.shape[1], ensemble_pred.shape[2]

            # Vectorized ID generation
            rows, cols = np.indices((h, w))
            rows = rows.flatten() + 1
            cols = cols.flatten() + 1

            ids = [f"{img_id_str}_{r}_{c}" for r, c in zip(rows, cols)]

            df_chunk = pd.DataFrame({"id": ids, "value": vals})
            results.append(df_chunk)

    # Concatenate and save
    final_df = pd.concat(results)
    final_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
