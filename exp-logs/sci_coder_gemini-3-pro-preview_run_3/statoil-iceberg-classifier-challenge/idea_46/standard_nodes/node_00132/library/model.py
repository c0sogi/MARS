import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.utils import set_seed, get_device

# ==========================================
# Configuration & Constants
# ==========================================
CACHE_DIR = "./working/idea_46/"
SUBMISSION_DIR = "./submission/"
CHECKPOINT_DIR = "./checkpoints/"
INPUT_DIR = "./input/"
METADATA_DIR = "./metadata/"

IMG_HEIGHT = 75
IMG_WIDTH = 75
IMG_CHANNELS = 3

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 75
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_FOLDS = 5

# ==========================================
# Data Loading & Caching
# ==========================================


def load_processed_data(load_cached_data=True):
    """
    Loads processed data (images, angles, labels, ids) from cache or raw files.
    Returns:
        X_train (np.ndarray): (N_train, 3, 75, 75)
        angle_train (np.ndarray): (N_train,)
        y_train (np.ndarray): (N_train,)
        ids_train (np.ndarray): (N_train,)
        X_test (np.ndarray): (N_test, 3, 75, 75)
        angle_test (np.ndarray): (N_test,)
        ids_test (np.ndarray): (N_test,)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "angle_train": os.path.join(CACHE_DIR, "angle_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "ids_train": os.path.join(CACHE_DIR, "ids_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "angle_test": os.path.join(CACHE_DIR, "angle_test.npy"),
        "ids_test": os.path.join(CACHE_DIR, "ids_test.npy"),
    }

    # Check if all cache files exist
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading data from cache...")
        return (
            np.load(cache_files["X_train"]),
            np.load(cache_files["angle_train"]),
            np.load(cache_files["y_train"]),
            np.load(cache_files["ids_train"], allow_pickle=True),
            np.load(cache_files["X_test"]),
            np.load(cache_files["angle_test"]),
            np.load(cache_files["ids_test"], allow_pickle=True),
        )

    print("Processing data from scratch...")

    # Load Metadata to identify splits (merging train and val for CV)
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Combine train and val for full CV
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Load Raw JSON
    # We load everything into memory; dataset is small enough (~200MB raw)
    with open(os.path.join(INPUT_DIR, "train.json"), "r") as f:
        raw_train = json.load(f)
    with open(os.path.join(INPUT_DIR, "test.json"), "r") as f:
        raw_test = json.load(f)

    # Helper to process list of dicts into arrays
    def process_json_data(raw_data, is_test=False):
        ids = []
        bands_1 = []
        bands_2 = []
        angles = []
        labels = []

        for item in raw_data:
            ids.append(item["id"])
            bands_1.append(item["band_1"])
            bands_2.append(item["band_2"])
            angles.append(item["inc_angle"])
            if not is_test:
                labels.append(item["is_iceberg"])

        # Reshape images
        b1 = np.array(bands_1, dtype=np.float32).reshape(-1, 75, 75)
        b2 = np.array(bands_2, dtype=np.float32).reshape(-1, 75, 75)
        # Synthetic 3rd band
        b3 = (b1 + b2) / 2.0

        # Stack: (N, 3, 75, 75)
        X = np.stack([b1, b2, b3], axis=1)

        # Process angles
        # Replace 'na' with NaN, convert to float
        angles_clean = []
        for a in angles:
            if isinstance(a, str) and a.lower() == "na":
                angles_clean.append(np.nan)
            else:
                angles_clean.append(float(a))
        angles_arr = np.array(angles_clean, dtype=np.float32)

        ids_arr = np.array(ids)

        if is_test:
            return X, angles_arr, ids_arr
        else:
            y_arr = np.array(labels, dtype=np.float32)
            return X, angles_arr, ids_arr, y_arr

    # Process
    X_train_full, angle_train_full, ids_train_full, y_train_full = process_json_data(
        raw_train, is_test=False
    )
    X_test, angle_test, ids_test = process_json_data(raw_test, is_test=True)

    # Impute Angles
    # Calculate median from training data (ignoring NaNs)
    angle_median = np.nanmedian(angle_train_full)

    # Fill NaNs
    angle_train_full = np.nan_to_num(angle_train_full, nan=angle_median)
    angle_test = np.nan_to_num(angle_test, nan=angle_median)

    # Filter/Reorder based on metadata if necessary?
    # The raw json order usually matches, but let's be safe and just use the raw processing
    # since we are doing 5-fold CV on all available labeled data.
    # The metadata files were derived from train.json, so the content is identical.

    # Save to cache
    np.save(cache_files["X_train"], X_train_full)
    np.save(cache_files["angle_train"], angle_train_full)
    np.save(cache_files["y_train"], y_train_full)
    np.save(cache_files["ids_train"], ids_train_full)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angle_test"], angle_test)
    np.save(cache_files["ids_test"], ids_test)

    return (
        X_train_full,
        angle_train_full,
        y_train_full,
        ids_train_full,
        X_test,
        angle_test,
        ids_test,
    )


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, transform=None):
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        angle = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img)  # (3, 75, 75)
        angle_tensor = torch.tensor(angle, dtype=torch.float32).unsqueeze(0)  # (1,)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)  # (1,)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


# ==========================================
# Model Architecture
# ==========================================


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.se = SEModule(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class SplitPolarityReadout(nn.Module):
    def __init__(self, in_channels, branch_channels):
        super(SplitPolarityReadout, self).__init__()
        # 1x1 Convolutions for projection
        self.conv_peak = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.conv_shadow = nn.Conv2d(in_channels, branch_channels, kernel_size=1)

    def forward(self, x):
        # Peak Branch: Global Max Pooling
        p = self.conv_peak(x)
        peak_feat = F.adaptive_max_pool2d(p, 1).flatten(1)

        # Shadow Branch: Global Min Pooling (implemented as Max(-x))
        s = self.conv_shadow(x)
        shadow_feat = F.adaptive_max_pool2d(-s, 1).flatten(1)

        # Concatenate
        return torch.cat([peak_feat, shadow_feat], dim=1)


class SPPCNN(nn.Module):
    def __init__(self):
        super(SPPCNN, self).__init__()

        # Backbone: Plain CNN 4 Stages
        # Input: 3 channels. 75x75
        self.stage1 = ConvBlock(3, 64)  # -> 37x37
        self.stage2 = ConvBlock(64, 128)  # -> 18x18
        self.stage3 = ConvBlock(128, 128)  # -> 9x9
        self.stage4 = ConvBlock(128, 128)  # -> 4x4

        # Split-Polarity Readout
        # Input 128 channels. Output 64+64=128 features
        self.readout = SplitPolarityReadout(128, 64)

        # Classification Head
        # Input: 128 (image features) + 1 (angle) = 129
        self.head = nn.Sequential(
            nn.Linear(129, 128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, angle):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        features = self.readout(x)

        # Fuse with angle
        combined = torch.cat([features, angle], dim=1)

        out = self.head(combined)
        return out


# ==========================================
# Training & Evaluation Logic
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, angles, labels in loader:
        images, angles, labels = images.to(device), angles.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for images, angles, labels in loader:
            images, angles, labels = (
                images.to(device),
                angles.to(device),
                labels.to(device),
            )
            outputs = model(images, angles)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return running_loss / len(loader.dataset)


def run_training_pipeline():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Load Data
    X, angles, y, ids, X_test, angle_test, ids_test = load_processed_data(
        load_cached_data=True
    )

    # Augmentations (Horizontal and Vertical Flip)
    # We implement simple random flips manually or via torchvision
    import torchvision.transforms as T

    train_transform = T.Compose(
        [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)]
    )

    # 5-Fold CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    test_preds_accum = np.zeros((len(X_test), 1))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold} ===")

        # Split Data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        angle_train_fold, angle_val_fold = angles[train_idx], angles[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Datasets
        train_dataset = IcebergDataset(
            X_train_fold, angle_train_fold, y_train_fold, transform=train_transform
        )
        val_dataset = IcebergDataset(
            X_val_fold, angle_val_fold, y_val_fold, transform=None
        )

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Model setup
        model = SPPCNN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Training Loop with Early Stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold}.pth")

        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or patience_counter == 0:
                print(
                    f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
                )

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Best Val Loss Fold {fold}: {best_val_loss:.6f}")

        # Inference on Test Set for this fold
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        test_dataset = IcebergDataset(X_test, angle_test, y=None, transform=None)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        fold_preds = []
        with torch.no_grad():
            for images, angles in test_loader:
                images, angles = images.to(device), angles.to(device)
                outputs = model(images, angles)
                fold_preds.extend(torch.sigmoid(outputs).cpu().numpy())

        test_preds_accum += np.array(fold_preds)

    # Average Predictions
    avg_preds = test_preds_accum / N_FOLDS

    # Save Submission
    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds.flatten()})

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


# Wrapper function to be called if needed, though not strictly required by "module class/functions" instruction,
# it serves as the entry point for the logic requested.
def main():
    run_training_pipeline()
