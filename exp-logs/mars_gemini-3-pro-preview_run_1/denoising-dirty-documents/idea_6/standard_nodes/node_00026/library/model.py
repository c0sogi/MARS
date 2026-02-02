import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    SUBMISSION_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    PATCH_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    IN_CHANNELS,
    OUT_CHANNELS,
    BASE_FILTERS,
    NUM_FOLDS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    T_MAX,
    ETA_MIN,
    DEVICE,
    TTA_VIEWS,
    SEED,
)
from library.utils import seed_everything, worker_init_fn, rmse_score

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


class DoubleConv(nn.Module):
    """
    (Conv3x3 -> BN -> ReLU) * 2
    """

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


class Up(nn.Module):
    """
    Upscaling then DoubleConv.
    Uses Bilinear Upsampling followed by a Convolution to reduce channels,
    then Concatenation and DoubleConv.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Bilinear upsampling
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # 1x1 Conv to halve the channels before concatenation
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        # DoubleConv to process the concatenated features
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv(x1)

        # Handle padding if dimensions don't match exactly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Standard U-Net Architecture (4 Levels).
    Cite Lesson 00024: Deeper architecture allows aggregating global context.
    Encoder: 32 -> 64 -> 128 -> 256 -> 512
    Decoder: 512 -> 256 -> 128 -> 64 -> 32
    """

    def __init__(
        self, n_channels=IN_CHANNELS, n_classes=OUT_CHANNELS, base_filters=BASE_FILTERS
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_filters, base_filters * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_filters * 2, base_filters * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_filters * 4, base_filters * 8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_filters * 8, base_filters * 16)
        )

        # Decoder
        self.up1 = Up(base_filters * 16, base_filters * 8)
        self.up2 = Up(base_filters * 8, base_filters * 4)
        self.up3 = Up(base_filters * 4, base_filters * 2)
        self.up4 = Up(base_filters * 2, base_filters)

        # Output Head
        self.outc = nn.Conv2d(base_filters, n_classes, kernel_size=1)

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

        return torch.sigmoid(self.outc(x))


# =============================================================================
# DATASET & CACHING
# =============================================================================


def load_data_entries(metadata_paths):
    entries = []
    for path in metadata_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                entry = {
                    "id": str(row["id"]),
                    "noisy_path": os.path.join(INPUT_DIR, row["noisy_image_path"]),
                }
                if "clean_image_path" in row:
                    entry["clean_path"] = os.path.join(
                        INPUT_DIR, row["clean_image_path"]
                    )
                entries.append(entry)
    return entries


def load_images_to_memory(entries):
    ids = []
    noisy_imgs = []
    clean_imgs = []

    for entry in entries:
        # Load as grayscale
        n_img = cv2.imread(entry["noisy_path"], cv2.IMREAD_GRAYSCALE)
        if n_img is None:
            continue

        ids.append(entry["id"])
        noisy_imgs.append(n_img)

        if "clean_path" in entry:
            c_img = cv2.imread(entry["clean_path"], cv2.IMREAD_GRAYSCALE)
            clean_imgs.append(c_img)

    return (
        np.array(ids),
        np.array(noisy_imgs, dtype=object),
        np.array(clean_imgs, dtype=object),
    )


def get_data(load_cached_data=True):
    """
    Loads dataset, using caching mechanism to speed up subsequent runs.
    Returns: (train_ids, train_noisy, train_clean), (val_ids, val_noisy, val_clean), (test_ids, test_noisy)
    """
    cache_train_path = os.path.join(WORKING_DIR, "train_only_cache.npz")
    cache_val_path = os.path.join(WORKING_DIR, "val_cache.npz")
    cache_test_path = os.path.join(WORKING_DIR, "test_cache.npz")

    # Try loading cache
    if (
        load_cached_data
        and os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    ):
        try:
            train_data = np.load(cache_train_path, allow_pickle=True)
            val_data = np.load(cache_val_path, allow_pickle=True)
            test_data = np.load(cache_test_path, allow_pickle=True)
            return (
                (train_data["ids"], train_data["noisy"], train_data["clean"]),
                (val_data["ids"], val_data["noisy"], val_data["clean"]),
                (test_data["ids"], test_data["noisy"]),
            )
        except Exception:
            pass  # Fallback to reload

    # Load from source
    # Load Train and Val separately
    train_entries = load_data_entries([TRAIN_METADATA_PATH])
    val_entries = load_data_entries([VAL_METADATA_PATH])
    test_entries = load_data_entries([TEST_METADATA_PATH])

    train_ids, train_noisy, train_clean = load_images_to_memory(train_entries)
    val_ids, val_noisy, val_clean = load_images_to_memory(val_entries)
    test_ids, test_noisy, _ = load_images_to_memory(test_entries)

    # Save cache
    np.savez(cache_train_path, ids=train_ids, noisy=train_noisy, clean=train_clean)
    np.savez(cache_val_path, ids=val_ids, noisy=val_noisy, clean=val_clean)
    np.savez(cache_test_path, ids=test_ids, noisy=test_noisy)

    return (
        (train_ids, train_noisy, train_clean),
        (val_ids, val_noisy, val_clean),
        (test_ids, test_noisy),
    )


class DenoisingDataset(Dataset):
    def __init__(self, noisy_imgs, clean_imgs=None, transform=None):
        self.noisy_imgs = noisy_imgs
        self.clean_imgs = clean_imgs
        self.transform = transform

    def __len__(self):
        return len(self.noisy_imgs)

    def __getitem__(self, idx):
        # Images are (H, W) uint8
        img = self.noisy_imgs[idx]

        # Normalize to [0, 1] float32
        img = img.astype(np.float32) / 255.0

        if self.clean_imgs is not None:
            mask = self.clean_imgs[idx].astype(np.float32) / 255.0
        else:
            mask = np.zeros_like(img)  # Placeholder

        if self.transform:
            # Albumentations expects [H, W, C] or [H, W]
            # It returns dictionary
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

            # Ensure mask has channel dimension (Albumentations returns 2D for masks)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            # Convert to tensor (C, H, W)
            img = torch.from_numpy(img).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask


# =============================================================================
# TRAINING & INFERENCE
# =============================================================================


def train_fold(fold_idx, train_idx, val_idx, images, masks, epochs=EPOCHS):
    """
    Trains a single fold of the model.
    """
    seed_everything(SEED + fold_idx)

    # Subset data
    train_imgs = images[train_idx]
    train_masks = masks[train_idx]
    val_imgs = images[val_idx]
    val_masks = masks[val_idx]

    # Transforms
    train_transform = A.Compose(
        [
            A.RandomCrop(height=PATCH_SIZE, width=PATCH_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # DataLoaders
    train_ds = DenoisingDataset(train_imgs, train_masks, transform=train_transform)
    val_ds = DenoisingDataset(val_imgs, val_masks, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Model setup
    model = UNet().to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=ETA_MIN
    )
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    model_path = os.path.join(WORKING_DIR, f"model_fold_{fold_idx}.pth")

    print(f"Starting Fold {fold_idx}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_rmse = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                # Handle padding for validation if dimensions are not divisible by 4
                h, w = x.shape[2], x.shape[3]
                pad_h = (4 - h % 4) % 4
                pad_w = (4 - w % 4) % 4
                if pad_h > 0 or pad_w > 0:
                    x = F.pad(x, (0, pad_w, 0, pad_h))

                pred = model(x)

                # Crop back
                if pad_h > 0 or pad_w > 0:
                    pred = pred[:, :, :h, :w]

                val_rmse += rmse_score(y, pred)

        val_rmse /= len(val_loader)

        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            print(
                f"Fold {fold_idx} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val RMSE: {val_rmse:.10f}"
            )

    print(f"Fold {fold_idx} Finished. Best RMSE: {best_rmse:.10f}")
    return best_rmse


def predict_tta(model, x):
    """
    Test-Time Augmentation (8 views).
    """
    preds = []

    # Helper for rotation/flip
    # x is (1, 1, H, W)

    # 1. Original
    preds.append(model(x))

    # 2. Horizontal Flip
    x_flip = torch.flip(x, [3])
    preds.append(torch.flip(model(x_flip), [3]))

    # 3. Vertical Flip
    x_vflip = torch.flip(x, [2])
    preds.append(torch.flip(model(x_vflip), [2]))

    # 4. Rotate 90
    x_rot90 = torch.rot90(x, 1, [2, 3])
    preds.append(torch.rot90(model(x_rot90), -1, [2, 3]))

    # 5. Rotate 180
    x_rot180 = torch.rot90(x, 2, [2, 3])
    preds.append(torch.rot90(model(x_rot180), -2, [2, 3]))

    # 6. Rotate 270
    x_rot270 = torch.rot90(x, 3, [2, 3])
    preds.append(torch.rot90(model(x_rot270), -3, [2, 3]))

    # 7. Transpose (Flip + Rot90 approx) - actually let's stick to simple flips/rots
    # Just averaging these 6 is often sufficient, but let's add FlipH + Rot90
    x_fr90 = torch.rot90(x_flip, 1, [2, 3])
    preds.append(torch.flip(torch.rot90(model(x_fr90), -1, [2, 3]), [3]))

    # 8. FlipV + Rot90
    x_fvr90 = torch.rot90(x_vflip, 1, [2, 3])
    preds.append(torch.flip(torch.rot90(model(x_fvr90), -1, [2, 3]), [2]))

    # Average
    return torch.stack(preds).mean(dim=0)


def generate_submission(test_ids, test_images):
    """
    Generates submission file using ensemble of 5 folds + TTA.
    """
    print("Generating submission...")

    # Load all models
    models = []
    for i in range(NUM_FOLDS):
        path = os.path.join(WORKING_DIR, f"model_fold_{i}.pth")
        if os.path.exists(path):
            m = UNet().to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            models.append(m)

    if not models:
        print("No models found!")
        return

    submission_rows = []

    transform = A.Compose([ToTensorV2()])

    with torch.no_grad():
        for idx, img_id in enumerate(test_ids):
            img = test_images[idx].astype(np.float32) / 255.0

            # Prepare tensor
            augmented = transform(image=img)["image"]
            x = augmented.unsqueeze(0).to(DEVICE)  # (1, 1, H, W)

            # Padding
            h, w = x.shape[2], x.shape[3]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))

            # Ensemble Prediction with TTA
            fold_preds = []
            for model in models:
                pred = predict_tta(model, x)
                fold_preds.append(pred)

            # Average folds
            avg_pred = torch.stack(fold_preds).mean(dim=0)

            # Crop back
            if pad_h > 0 or pad_w > 0:
                avg_pred = avg_pred[:, :, :h, :w]

            # Convert to numpy
            pred_np = avg_pred.squeeze().cpu().numpy()  # (H, W)

            # Flatten and format
            # id format: image_row_col (1-based indexing)
            # numpy is 0-based row, col
            rows, cols = pred_np.shape
            for r in range(rows):
                for c in range(cols):
                    pixel_id = f"{img_id}_{r+1}_{c+1}"
                    val = pred_np[r, c]
                    submission_rows.append(f"{pixel_id},{val:.6f}")

    # Write to file
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    with open(sub_path, "w") as f:
        f.write("id,value\n")
        f.write("\n".join(submission_rows))

    print(f"Submission saved to {sub_path}")


def run_training_pipeline():
    """
    Main entry point to run the 5-Fold CV training.
    """
    seed_everything(SEED)

    # Load Data
    (train_ids, train_noisy, train_clean), _ = get_data(load_cached_data=True)

    # K-Fold
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_noisy)):
        rmse = train_fold(fold, train_idx, val_idx, train_noisy, train_clean)
        fold_results.append(rmse)

    print(f"CV Average RMSE: {np.mean(fold_results):.10f}")


def run_inference_pipeline():
    """
    Main entry point to run inference.
    """
    _, (test_ids, test_noisy) = get_data(load_cached_data=True)
    generate_submission(test_ids, test_noisy)
