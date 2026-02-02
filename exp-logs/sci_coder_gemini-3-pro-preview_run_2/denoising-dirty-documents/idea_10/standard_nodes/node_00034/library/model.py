import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import glob

from library.wavelet_layers import DWT, IWT
from library.utils import (
    get_device,
    save_checkpoint,
    load_checkpoint,
    calculate_rmse,
    save_submission,
    seed_everything,
    load_image_with_cache,
)
from library.dataset import get_dataloaders


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for efficient mobile network design.
    """

    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class ResBlock(nn.Module):
    """
    Residual Block with Coordinate Attention.
    """

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ca = CoordinateAttention(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out)
        out += residual
        out = self.relu(out)
        return out


class WaveCACResUNet(nn.Module):
    """
    Wavelet-Integrated Context-Aware Coordinate ResUNet.
    Uses DWT for downsampling and IWT for upsampling.
    """

    def __init__(self, in_channels=1, base_filters=64):
        super(WaveCACResUNet, self).__init__()

        self.dwt = DWT()
        self.iwt = IWT()

        # Initial Convolution
        self.head = nn.Conv2d(in_channels, base_filters, 3, padding=1)

        # Encoder 1
        self.enc1_res = ResBlock(base_filters)
        # DWT (base -> 4*base)
        self.enc1_conv = nn.Conv2d(4 * base_filters, 2 * base_filters, 1)

        # Encoder 2
        self.enc2_res = ResBlock(2 * base_filters)
        # DWT (2*base -> 8*base)
        self.enc2_conv = nn.Conv2d(8 * base_filters, 4 * base_filters, 1)

        # Encoder 3
        self.enc3_res = ResBlock(4 * base_filters)
        # DWT (4*base -> 16*base)
        self.enc3_conv = nn.Conv2d(16 * base_filters, 8 * base_filters, 1)

        # Bottleneck
        self.bottleneck = ResBlock(8 * base_filters)

        # Decoder 3
        self.dec3_up_conv = nn.Conv2d(8 * base_filters, 16 * base_filters, 1)
        # IWT (16*base -> 4*base)
        # Concat with enc3 (4*base) -> 8*base
        self.dec3_conv = nn.Conv2d(8 * base_filters, 4 * base_filters, 1)
        self.dec3_res = ResBlock(4 * base_filters)

        # Decoder 2
        self.dec2_up_conv = nn.Conv2d(4 * base_filters, 8 * base_filters, 1)
        # IWT (8*base -> 2*base)
        # Concat with enc2 (2*base) -> 4*base
        self.dec2_conv = nn.Conv2d(4 * base_filters, 2 * base_filters, 1)
        self.dec2_res = ResBlock(2 * base_filters)

        # Decoder 1
        self.dec1_up_conv = nn.Conv2d(2 * base_filters, 4 * base_filters, 1)
        # IWT (4*base -> base)
        # Concat with enc1 (base) -> 2*base
        self.dec1_conv = nn.Conv2d(2 * base_filters, base_filters, 1)
        self.dec1_res = ResBlock(base_filters)

        # Tail
        self.tail = nn.Conv2d(base_filters, in_channels, 3, padding=1)

    def forward(self, x):
        # Head
        x = self.head(x)  # (B, 64, H, W)

        # Encoder 1
        skip1 = self.enc1_res(x)  # (B, 64, H, W)
        x = self.dwt(skip1)  # (B, 256, H/2, W/2)
        x = self.enc1_conv(x)  # (B, 128, H/2, W/2)

        # Encoder 2
        skip2 = self.enc2_res(x)  # (B, 128, H/2, W/2)
        x = self.dwt(skip2)  # (B, 512, H/4, W/4)
        x = self.enc2_conv(x)  # (B, 256, H/4, W/4)

        # Encoder 3
        skip3 = self.enc3_res(x)  # (B, 256, H/4, W/4)
        x = self.dwt(skip3)  # (B, 1024, H/8, W/8)
        x = self.enc3_conv(x)  # (B, 512, H/8, W/8)

        # Bottleneck
        x = self.bottleneck(x)  # (B, 512, H/8, W/8)

        # Decoder 3
        x = self.dec3_up_conv(x)  # (B, 1024, H/8, W/8)
        x = self.iwt(x)  # (B, 256, H/4, W/4)
        x = torch.cat([x, skip3], dim=1)  # (B, 512, H/4, W/4)
        x = self.dec3_conv(x)  # (B, 256, H/4, W/4)
        x = self.dec3_res(x)

        # Decoder 2
        x = self.dec2_up_conv(x)  # (B, 512, H/4, W/4)
        x = self.iwt(x)  # (B, 128, H/2, W/2)
        x = torch.cat([x, skip2], dim=1)  # (B, 256, H/2, W/2)
        x = self.dec2_conv(x)  # (B, 128, H/2, W/2)
        x = self.dec2_res(x)

        # Decoder 1
        x = self.dec1_up_conv(x)  # (B, 256, H/2, W/2)
        x = self.iwt(x)  # (B, 64, H, W)
        x = torch.cat([x, skip1], dim=1)  # (B, 128, H, W)
        x = self.dec1_conv(x)  # (B, 64, H, W)
        x = self.dec1_res(x)

        # Tail
        out = self.tail(x)  # (B, 1, H, W)
        return out


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Model predicts noise residual
        target_noise = noisy - clean

        optimizer.zero_grad()
        predicted_noise = model(noisy)
        loss = criterion(predicted_noise, target_noise)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, device):
    model.eval()
    total_rmse = 0.0
    count = 0

    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean_np = clean.numpy()

            predicted_noise = model(noisy)

            # Reconstruct clean image: Clean = Noisy - Predicted_Noise
            predicted_clean = noisy - predicted_noise
            predicted_clean_np = predicted_clean.cpu().numpy()

            # Clip to valid range
            predicted_clean_np = np.clip(predicted_clean_np, 0, 1)

            # Calculate RMSE for this batch
            batch_rmse = calculate_rmse(clean_np, predicted_clean_np)
            total_rmse += batch_rmse * noisy.size(0)
            count += noisy.size(0)

    return total_rmse / count


def predict_tiled(model, image, device, patch_size=128, overlap=0.5):
    """
    Performs tiled inference with overlap to reduce boundary artifacts.
    """
    model.eval()
    h, w = image.shape
    stride = int(patch_size * (1 - overlap))

    # Add batch and channel dims
    img_tensor = (
        torch.from_numpy(image.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
    )

    output_sum = torch.zeros((1, 1, h, w), device=device)
    output_count = torch.zeros((1, 1, h, w), device=device)

    # Pad image to be multiple of patch size/stride if necessary
    # Simple reflection padding for the whole image to handle edges
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    # We might need more padding to ensure the last tile covers the edge properly with stride
    # For simplicity in this constrained environment, we iterate and handle boundaries

    with torch.no_grad():
        for y in range(0, h - patch_size + 1 + stride, stride):
            for x in range(0, w - patch_size + 1 + stride, stride):
                # Handle boundary conditions
                y_start = min(y, h - patch_size)
                x_start = min(x, w - patch_size)
                y_end = y_start + patch_size
                x_end = x_start + patch_size

                patch = img_tensor[:, :, y_start:y_end, x_start:x_end]

                # Predict noise
                pred_noise = model(patch)

                # Accumulate
                output_sum[:, :, y_start:y_end, x_start:x_end] += pred_noise
                output_count[:, :, y_start:y_end, x_start:x_end] += 1

                if x_start == w - patch_size:
                    break
            if y_start == h - patch_size:
                break

    avg_noise = output_sum / output_count

    # Clean = Input - Noise
    clean_est = img_tensor - avg_noise
    return clean_est.squeeze().cpu().numpy()


def run_pipeline(
    data_dir="./input",
    work_dir="./working/idea_10",
    epochs=100,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-2,
):
    seed_everything(42)
    device = get_device()

    # Directories
    cache_dir = os.path.join(work_dir, "cache")
    checkpoint_path = os.path.join(work_dir, "best_model.pth")
    submission_path = os.path.join(work_dir, "submission.csv")

    # 1. Data Loaders
    # High density sampling: 100 samples per image per epoch for training
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=4,
        patch_size=128,
        train_samples_per_epoch=100,
        val_samples_per_epoch=1,
    )

    # 2. Model Setup
    model = WaveCACResUNet(in_channels=1, base_filters=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training Loop
    best_rmse = float("inf")
    patience = 10
    no_improve_epochs = 0

    print(f"Starting training on {device}...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_rmse = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse:.6f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_rmse": best_rmse,
                },
                checkpoint_path,
            )
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 4. Inference & Submission
    print("Generating submission...")

    # Load best model
    checkpoint = load_checkpoint(checkpoint_path, model)
    model.eval()

    # Get Test Images
    test_csv = os.path.join("./metadata", "test.csv")
    if os.path.exists(test_csv):
        df_test = pd.read_csv(test_csv)
        predictions = {}

        for _, row in df_test.iterrows():
            img_id = row["id"]
            feature_path = os.path.join(data_dir, row["feature_path"])
            cache_path = os.path.join(cache_dir, "test", f"{img_id}_noisy.npy")

            # Load full image
            img = load_image_with_cache(feature_path, cache_path)

            # Predict
            pred_clean = predict_tiled(model, img, device)
            predictions[img_id] = pred_clean

        save_submission(predictions, submission_path)
        print(f"Submission saved to {submission_path}")
    else:
        print("Test metadata not found. Skipping submission generation.")
