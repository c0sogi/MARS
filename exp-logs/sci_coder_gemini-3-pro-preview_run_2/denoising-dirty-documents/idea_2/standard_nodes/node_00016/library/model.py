import os
import glob
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from library.utils import set_seed, save_submission_file

# =============================================================================
# 1. Architecture: ResUNet
# =============================================================================


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    A Residual Block with two 3x3 convolutions, Batch Normalization, SiLU, and SE Block.
    Implements the skip connection: Output = F(x) + x.
    """

    def __init__(self, in_channels, out_channels, reduction=16):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)

        # Shortcut connection to match dimensions if necessary
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    """
    Residual U-Net architecture for Global Residual Learning.
    The model predicts the noise map.
    """

    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super(ResUNet, self).__init__()

        # Encoder
        self.input_conv = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(base_filters)
        self.relu = nn.SiLU(inplace=True)

        self.enc1 = ResidualBlock(base_filters, base_filters)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = ResidualBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = ResidualBlock(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bridge
        self.bridge = ResidualBlock(base_filters * 8, base_filters * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            base_filters * 16, base_filters * 8, kernel_size=2, stride=2
        )
        self.dec4 = ResidualBlock(
            base_filters * 16, base_filters * 8
        )  # Concat 8+8 -> 16 in channels

        self.up3 = nn.ConvTranspose2d(
            base_filters * 8, base_filters * 4, kernel_size=2, stride=2
        )
        self.dec3 = ResidualBlock(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(
            base_filters * 4, base_filters * 2, kernel_size=2, stride=2
        )
        self.dec2 = ResidualBlock(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose2d(
            base_filters * 2, base_filters, kernel_size=2, stride=2
        )
        self.dec1 = ResidualBlock(base_filters * 2, base_filters)

        # Output
        self.output_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Initial Conv
        x0 = self.relu(self.bn_input(self.input_conv(x)))

        # Encoder
        e1 = self.enc1(x0)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bridge
        b = self.bridge(p4)

        # Decoder
        d4 = self.up4(b)
        # Handle padding issues if dimensions are not perfect powers of 2 (though 128x128 patches are fine)
        if d4.size() != e4.size():
            d4 = F.interpolate(
                d4, size=e4.shape[2:], mode="bilinear", align_corners=True
            )
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        if d3.size() != e3.size():
            d3 = F.interpolate(
                d3, size=e3.shape[2:], mode="bilinear", align_corners=True
            )
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.size() != e2.size():
            d2 = F.interpolate(
                d2, size=e2.shape[2:], mode="bilinear", align_corners=True
            )
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.size() != e1.size():
            d1 = F.interpolate(
                d1, size=e1.shape[2:], mode="bilinear", align_corners=True
            )
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output Noise Map
        noise = self.output_conv(d1)

        return noise


# =============================================================================
# 2. Data Management
# =============================================================================


def load_and_cache_data(
    metadata_path, cache_dir, input_dir="./input", load_cached=True
):
    """
    Loads images defined in metadata. Caches them as .npy files.
    """
    os.makedirs(cache_dir, exist_ok=True)

    df = pd.read_csv(metadata_path)
    data = []

    # Identify split name from metadata filename for cache naming
    split_name = os.path.splitext(os.path.basename(metadata_path))[0]

    # Check if we can load a single monolithic file (optional optimization)
    # But here we will cache per-image to avoid massive RAM spikes during creation if dataset was huge.
    # Given requirements, we will cache individual files and load them.

    for idx, row in df.iterrows():
        img_id = row["id"]
        feature_rel_path = row["feature_path"]

        # Determine paths
        cache_path_noisy = os.path.join(cache_dir, f"{img_id}_noisy.npy")

        has_label = "label_path" in row
        if has_label:
            cache_path_clean = os.path.join(cache_dir, f"{img_id}_clean.npy")

        # Logic: Load from cache if exists and requested
        noisy_loaded = False
        clean_loaded = False

        img_noisy = None
        img_clean = None

        if load_cached:
            if os.path.exists(cache_path_noisy):
                try:
                    img_noisy = np.load(cache_path_noisy)
                    noisy_loaded = True
                except:
                    noisy_loaded = False

            if has_label and os.path.exists(cache_path_clean):
                try:
                    img_clean = np.load(cache_path_clean)
                    clean_loaded = True
                except:
                    clean_loaded = False

        # If not loaded, read from source and save
        if not noisy_loaded:
            full_path = os.path.join(input_dir, feature_rel_path)
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip missing files
            img_noisy = img.astype(np.float32) / 255.0
            np.save(cache_path_noisy, img_noisy)

        if has_label and not clean_loaded:
            full_path = os.path.join(input_dir, row["label_path"])
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_clean = img.astype(np.float32) / 255.0
            np.save(cache_path_clean, img_clean)

        # Append to list
        item = {"id": img_id, "noisy": img_noisy}
        if has_label:
            item["clean"] = img_clean
        data.append(item)

    return data


class DenoisingDataset(Dataset):
    def __init__(self, data, patch_size=128, augment=False, patches_per_image=4):
        self.data = data
        self.patch_size = patch_size
        self.augment = augment
        self.patches_per_image = patches_per_image

    def __len__(self):
        if self.augment:
            return len(self.data) * self.patches_per_image
        return len(self.data)

    def __getitem__(self, idx):
        # Map index to image index
        if self.augment:
            img_idx = idx // self.patches_per_image
        else:
            img_idx = idx

        sample = self.data[img_idx]
        noisy = sample["noisy"]
        clean = sample.get("clean", None)

        h, w = noisy.shape

        if self.augment and clean is not None:
            # Random Crop
            if h > self.patch_size and w > self.patch_size:
                y = random.randint(0, h - self.patch_size)
                x = random.randint(0, w - self.patch_size)
                noisy_patch = noisy[y : y + self.patch_size, x : x + self.patch_size]
                clean_patch = clean[y : y + self.patch_size, x : x + self.patch_size]
            else:
                # Resize if smaller (unlikely based on EDA) or just take center
                noisy_patch = cv2.resize(noisy, (self.patch_size, self.patch_size))
                clean_patch = cv2.resize(clean, (self.patch_size, self.patch_size))

            # Geometric Augmentations
            # 1. Horizontal Flip
            if random.random() > 0.5:
                noisy_patch = np.fliplr(noisy_patch)
                clean_patch = np.fliplr(clean_patch)

            # 2. Vertical Flip
            if random.random() > 0.5:
                noisy_patch = np.flipud(noisy_patch)
                clean_patch = np.flipud(clean_patch)

            # 3. Rotation 90
            k = random.randint(0, 3)
            if k > 0:
                noisy_patch = np.rot90(noisy_patch, k)
                clean_patch = np.rot90(clean_patch, k)

            # To Tensor
            noisy_t = torch.from_numpy(noisy_patch.copy()).unsqueeze(0).float()
            clean_t = torch.from_numpy(clean_patch.copy()).unsqueeze(0).float()

            return noisy_t, clean_t

        else:
            # Validation/Test mode: Return full image (or handled by tiled inference)
            # Here we return tensor of full image
            noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
            if clean is not None:
                clean_t = torch.from_numpy(clean).unsqueeze(0).float()
                return noisy_t, clean_t
            return noisy_t, sample["id"]


# =============================================================================
# 3. Inference Logic (Tiled + TTA)
# =============================================================================


def predict_tiled(model, image_tensor, patch_size=128, overlap=32, device="cuda"):
    """
    Performs tiled inference on a single image tensor (C, H, W).
    """
    model.eval()
    c, h, w = image_tensor.shape

    # Output container
    # We accumulate noise predictions
    noise_sum = torch.zeros((c, h, w), device=device)
    count_map = torch.zeros((c, h, w), device=device)

    stride = patch_size - overlap

    # Generate patches
    # Simple sliding window
    r_steps = (h - patch_size) // stride + 2
    c_steps = (w - patch_size) // stride + 2

    patches = []
    coords = []

    for r in range(0, h, stride):
        for c_idx in range(0, w, stride):
            # Handle boundary
            r_end = min(r + patch_size, h)
            c_end = min(c_idx + patch_size, w)
            r_start = r_end - patch_size
            c_start = c_end - patch_size

            # Ensure valid
            if r_start < 0:
                r_start = 0
            if c_start < 0:
                c_start = 0

            patch = image_tensor[:, r_start:r_end, c_start:c_end]
            patches.append(patch)
            coords.append((r_start, r_end, c_start, c_end))

            if c_end == w:
                break
        if r_end == h:
            break

    if not patches:  # Image smaller than patch
        patches = [image_tensor]
        coords = [(0, h, 0, w)]

    # Batch process
    batch_size = 16
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i : i + batch_size]
        batch_coords = coords[i : i + batch_size]

        batch_tensor = torch.stack(batch_patches).to(device)

        with torch.no_grad():
            # Predict NOISE
            pred_noise = model(batch_tensor)

        for noise_patch, (r1, r2, c1, c2) in zip(pred_noise, batch_coords):
            noise_sum[:, r1:r2, c1:c2] += noise_patch
            count_map[:, r1:r2, c1:c2] += 1.0

    # Average
    avg_noise = noise_sum / torch.clamp(count_map, min=1.0)

    # Global Residual: Clean = Input - Noise
    clean_pred = image_tensor.to(device) - avg_noise

    return torch.clamp(clean_pred, 0, 1)


def inference_tiled_tta(model, image_tensor, device="cuda"):
    """
    Wraps tiled inference with Test Time Augmentation (Flip H, Flip V).
    """
    preds = []

    # 1. Original
    p1 = predict_tiled(model, image_tensor, device=device)
    preds.append(p1)

    # 2. Horizontal Flip
    img_h = torch.flip(image_tensor, [2])
    p2 = predict_tiled(model, img_h, device=device)
    preds.append(torch.flip(p2, [2]))

    # 3. Vertical Flip
    img_v = torch.flip(image_tensor, [1])
    p3 = predict_tiled(model, img_v, device=device)
    preds.append(torch.flip(p3, [1]))

    # Average
    final_pred = torch.stack(preds).mean(dim=0)
    return final_pred


# =============================================================================
# 4. Training Loop
# =============================================================================


def train_model(
    train_loader,
    val_data,
    epochs=50,
    lr=1e-4,
    device="cuda",
    save_path="./working/model.pth",
):
    set_seed(42)
    model = ResUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    patience = 5
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()

            # Predict Noise
            pred_noise = model(noisy)

            # Target Noise = Input - Clean
            target_noise = noisy - clean

            loss = criterion(pred_noise, target_noise)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_mse = 0.0
        total_pixels = 0

        # Validate on full images using tiled inference to match test conditions
        # We iterate manually over the validation list
        for item in val_data:
            noisy_np = item["noisy"]
            clean_np = item["clean"]

            noisy_t = (
                torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).float()
            )  # 1, 1, H, W
            clean_t = (
                torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0).float().to(device)
            )

            with torch.no_grad():
                # Use tiled inference without TTA for speed during validation, or with TTA for accuracy
                pred_clean = predict_tiled(
                    model, noisy_t.squeeze(0), device=device
                ).unsqueeze(0)

            # MSE on valid pixels
            diff = (pred_clean - clean_t) ** 2
            val_mse += diff.sum().item()
            total_pixels += diff.numel()

        val_rmse = np.sqrt(val_mse / total_pixels)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val RMSE: {val_rmse:.10f}"
        )

        # Early Stopping & Saving
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best Val RMSE: {best_rmse:.10f}")
    return save_path


def generate_submission(model_path, test_data, output_path, device="cuda"):
    model = ResUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    ids = []

    print("Generating predictions...")
    for item in test_data:
        img_id = item["id"]
        noisy_np = item["noisy"]

        noisy_t = (
            torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).float()
        )  # 1, 1, H, W

        with torch.no_grad():
            pred = inference_tiled_tta(model, noisy_t.squeeze(0), device=device)

        predictions.append(pred.cpu().numpy())
        ids.append(img_id)

    save_submission_file(predictions, ids, output_path)
    print(f"Submission saved to {output_path}")
