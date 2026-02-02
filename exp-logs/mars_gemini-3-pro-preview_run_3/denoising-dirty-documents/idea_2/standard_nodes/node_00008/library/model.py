import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
import time
from library.config import Config, seed_everything
from library.utils import (
    calculate_rmse,
    save_checkpoint,
    load_checkpoint,
    generate_submission_file,
)

# ==========================================
# U-Net Architecture
# ==========================================


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, mid_channels=in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ==========================================
# Data Processing & Loading
# ==========================================


def load_and_process_data(
    metadata_path, cache_prefix, load_cached_data=True, is_test=False
):
    """
    Loads images based on metadata. Caches processed arrays to disk.
    Handles variable image sizes by padding to max dimensions for storage,
    and returns a list of individual numpy arrays (cropped back to original size).
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    imgs_cache_path = os.path.join(cache_dir, f"{cache_prefix}_imgs.npy")
    shapes_cache_path = os.path.join(cache_dir, f"{cache_prefix}_shapes.npy")
    ids_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # Try loading from cache
    if (
        load_cached_data
        and os.path.exists(imgs_cache_path)
        and os.path.exists(shapes_cache_path)
    ):
        # print(f"Loading {cache_prefix} data from cache...")
        padded_imgs = np.load(imgs_cache_path)
        shapes = np.load(shapes_cache_path)
        ids = np.load(ids_cache_path, allow_pickle=True)  # IDs are strings

        # Reconstruct list of arrays
        data_list = []
        for i in range(len(padded_imgs)):
            h, w = shapes[i]
            # Crop back to original size
            img = padded_imgs[i, :h, :w]
            data_list.append(img)

        return ids, data_list

    # Process from scratch
    # print(f"Processing {cache_prefix} data from scratch...")
    df = pd.read_csv(metadata_path)

    # Debugging subsample
    if Config.DEBUG_SAMPLE_SIZE is not None:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    ids = []
    img_list = []

    max_h, max_w = 0, 0

    for idx, row in df.iterrows():
        img_id = row["image_id"]

        # Load Input
        input_path = os.path.join(Config.INPUT_DIR, row["input_path"])
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # If not test, we might have targets (but here we process inputs and targets separately or as pairs)
        # This function handles one list of images (either inputs or targets)
        # The caller should call this twice for train (inputs, targets)

        h, w = img.shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)

        ids.append(img_id)
        img_list.append(img)

    # Pad for storage
    count = len(img_list)
    padded_storage = np.zeros((count, max_h, max_w), dtype=np.float32)
    shapes_storage = np.zeros((count, 2), dtype=np.int32)

    for i, img in enumerate(img_list):
        h, w = img.shape
        padded_storage[i, :h, :w] = img
        shapes_storage[i] = [h, w]

    # Save to cache
    np.save(imgs_cache_path, padded_storage)
    np.save(shapes_cache_path, shapes_storage)
    np.save(ids_cache_path, np.array(ids))

    return np.array(ids), img_list


class DenoisingDataset(Dataset):
    def __init__(
        self, inputs, targets=None, transform=None, patch_size=None, train_mode=True
    ):
        self.inputs = inputs
        self.targets = targets
        self.patch_size = patch_size
        self.train_mode = train_mode

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_img = self.inputs[idx]

        if self.train_mode:
            # Random Crop
            target_img = self.targets[idx]
            h, w = input_img.shape

            # Ensure image is larger than patch
            if h < self.patch_size or w < self.patch_size:
                # Pad if necessary (though EDA suggests images are large enough)
                pad_h = max(0, self.patch_size - h)
                pad_w = max(0, self.patch_size - w)
                input_img = np.pad(input_img, ((0, pad_h), (0, pad_w)), mode="reflect")
                target_img = np.pad(
                    target_img, ((0, pad_h), (0, pad_w)), mode="reflect"
                )
                h, w = input_img.shape

            # Random coordinates
            y = np.random.randint(0, h - self.patch_size + 1)
            x = np.random.randint(0, w - self.patch_size + 1)

            patch_in = input_img[y : y + self.patch_size, x : x + self.patch_size]
            patch_tar = target_img[y : y + self.patch_size, x : x + self.patch_size]

            # Random Flip Augmentation
            if np.random.rand() > 0.5:
                patch_in = np.flipud(patch_in)
                patch_tar = np.flipud(patch_tar)
            if np.random.rand() > 0.5:
                patch_in = np.fliplr(patch_in)
                patch_tar = np.fliplr(patch_tar)

            # Add channel dim
            img_tensor = torch.from_numpy(patch_in.copy()).unsqueeze(0).float()
            target_tensor = torch.from_numpy(patch_tar.copy()).unsqueeze(0).float()

            return img_tensor, target_tensor

        else:
            # Validation / Test: Return full image
            # Add channel dim
            img_tensor = torch.from_numpy(input_img.copy()).unsqueeze(0).float()

            if self.targets is not None:
                target_tensor = (
                    torch.from_numpy(self.targets[idx].copy()).unsqueeze(0).float()
                )
                return img_tensor, target_tensor
            else:
                return img_tensor


# ==========================================
# Training & Inference
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total_rmse = 0.0
    count = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            # Targets are on CPU for metric calculation usually, but here we can keep on GPU or move back
            # calculate_rmse expects numpy or tensor

            # Pad input to be multiple of 16 (2^4 pooling layers)
            # This is necessary for U-Net if input dims are not divisible by 16
            h, w = inputs.shape[2], inputs.shape[3]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            if pad_h > 0 or pad_w > 0:
                inputs_padded = F.pad(inputs, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                inputs_padded = inputs

            outputs_padded = model(inputs_padded)

            # Crop back
            outputs = outputs_padded[:, :, :h, :w]

            # Clamp to [0, 1]
            outputs = torch.clamp(outputs, 0, 1)

            batch_rmse = calculate_rmse(targets, outputs)
            total_rmse += batch_rmse
            count += 1

    return total_rmse / count


def run_training(load_cached_data=True):
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # --- 1. Load Data ---
    # Train Inputs
    _, train_inputs = load_and_process_data(
        Config.TRAIN_METADATA_PATH, "train_in", load_cached_data
    )
    # Train Targets (Clean)
    # Note: We use the target path from metadata.
    # The load function reads 'input_path' column by default. We need to trick it or modify it.
    # Let's modify the load function slightly?
    # Better: Just pass the dataframe to a helper that extracts the right column.
    # Re-implementing specific target loader for clarity inside this block is cleaner given the constraints.

    # Helper to load targets specifically
    def load_targets(metadata_path, cache_prefix, load_cached):
        # Similar to load_and_process_data but uses 'target_path'
        cache_dir = Config.WORKING_DIR
        imgs_cache_path = os.path.join(cache_dir, f"{cache_prefix}_imgs.npy")
        shapes_cache_path = os.path.join(cache_dir, f"{cache_prefix}_shapes.npy")

        if load_cached and os.path.exists(imgs_cache_path):
            padded_imgs = np.load(imgs_cache_path)
            shapes = np.load(shapes_cache_path)
            data_list = []
            for i in range(len(padded_imgs)):
                h, w = shapes[i]
                data_list.append(padded_imgs[i, :h, :w])
            return data_list

        df = pd.read_csv(metadata_path)
        if Config.DEBUG_SAMPLE_SIZE:
            df = df.head(Config.DEBUG_SAMPLE_SIZE)
        img_list = []
        max_h, max_w = 0, 0
        for _, row in df.iterrows():
            path = os.path.join(Config.INPUT_DIR, row["target_path"])
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            h, w = img.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            img_list.append(img)

        # Cache logic (simplified for brevity, same as above)
        count = len(img_list)
        padded = np.zeros((count, max_h, max_w), dtype=np.float32)
        shapes = np.zeros((count, 2), dtype=np.int32)
        for i, img in enumerate(img_list):
            padded[i, : img.shape[0], : img.shape[1]] = img
            shapes[i] = img.shape
        np.save(imgs_cache_path, padded)
        np.save(shapes_cache_path, shapes)
        return img_list

    train_targets = load_targets(
        Config.TRAIN_METADATA_PATH, "train_target", load_cached_data
    )

    # Val Data
    _, val_inputs = load_and_process_data(
        Config.VAL_METADATA_PATH, "val_in", load_cached_data
    )
    val_targets = load_targets(Config.VAL_METADATA_PATH, "val_target", load_cached_data)

    # --- 2. Datasets & Loaders ---
    train_dataset = DenoisingDataset(
        train_inputs, train_targets, patch_size=Config.PATCH_SIZE, train_mode=True
    )
    val_dataset = DenoisingDataset(val_inputs, val_targets, train_mode=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    # Batch size 1 for validation to handle varying sizes
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    # --- 3. Model Setup ---
    model = UNet(n_channels=1, n_classes=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- 4. Training Loop ---
    best_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_rmse = validate(model, val_loader, device)

        scheduler.step(val_rmse)

        print(
            f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val RMSE = {val_rmse:.6f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_rmse, Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Best Validation RMSE: {best_rmse:.6f}")

    # --- 5. Inference on Test Set ---
    print("Generating submission...")

    # Load best model
    checkpoint = load_checkpoint(Config.MODEL_SAVE_PATH, model, device=device)
    model.eval()

    # Load Test Data
    test_ids, test_inputs = load_and_process_data(
        Config.TEST_METADATA_PATH, "test_in", load_cached_data
    )
    test_dataset = DenoisingDataset(test_inputs, train_mode=False)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    predictions = {}

    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device)
            img_id = test_ids[i]

            # Pad for U-Net
            h, w = inputs.shape[2], inputs.shape[3]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            if pad_h > 0 or pad_w > 0:
                inputs_padded = F.pad(inputs, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                inputs_padded = inputs

            outputs_padded = model(inputs_padded)

            # Crop back
            outputs = outputs_padded[:, :, :h, :w]
            outputs = torch.clamp(outputs, 0, 1)

            # Convert to numpy (H, W)
            pred_img = outputs.squeeze().cpu().numpy()
            predictions[img_id] = pred_img

    generate_submission_file(predictions, Config.SUBMISSION_PATH)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
