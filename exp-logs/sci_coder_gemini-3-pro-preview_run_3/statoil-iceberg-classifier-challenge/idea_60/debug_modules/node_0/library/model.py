import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as T
from library.config import Config
from library.utils import set_seed, save_checkpoint, AverageMeter

# -----------------------------------------------------------------------------
# Data Loading & Processing
# -----------------------------------------------------------------------------


def load_and_process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays (N, 3, 75, 75), and caches the result.
    """
    cache_files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "angles_train": os.path.join(Config.CACHE_DIR, "angles_train.npy"),
        "ids_train": os.path.join(Config.CACHE_DIR, "ids_train.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "angles_test": os.path.join(Config.CACHE_DIR, "angles_test.npy"),
        "ids_test": os.path.join(Config.CACHE_DIR, "ids_test.npy"),
    }

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
        return (
            np.load(cache_files["X_train"]),
            np.load(cache_files["y_train"]),
            np.load(cache_files["angles_train"]),
            np.load(cache_files["ids_train"]),
            np.load(cache_files["X_test"]),
            np.load(cache_files["angles_test"]),
            np.load(cache_files["ids_test"]),
        )

    # Load Raw Data
    with open(os.path.join(Config.INPUT_DIR, "train.json"), "r") as f:
        train_data = json.load(f)
    with open(os.path.join(Config.INPUT_DIR, "test.json"), "r") as f:
        test_data = json.load(f)

    def process_json_list(data_list, is_train=True):
        ids = []
        bands_1 = []
        bands_2 = []
        angles = []
        labels = []

        for item in data_list:
            ids.append(item["id"])
            bands_1.append(item["band_1"])
            bands_2.append(item["band_2"])

            # Handle angle
            ang = item["inc_angle"]
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))

            if is_train:
                labels.append(item["is_iceberg"])

        # Reshape images to (N, 75, 75)
        b1 = np.array(bands_1).reshape(-1, 75, 75)
        b2 = np.array(bands_2).reshape(-1, 75, 75)

        # Create 3rd channel: Average
        b3 = (b1 + b2) / 2.0

        # Stack: (N, 3, 75, 75)
        X = np.stack([b1, b2, b3], axis=1).astype(np.float32)

        ids = np.array(ids)
        angles = np.array(angles, dtype=np.float32)

        if is_train:
            y = np.array(labels, dtype=np.float32)
            return X, y, angles, ids
        else:
            return X, angles, ids

    X_train, y_train, angles_train, ids_train = process_json_list(
        train_data, is_train=True
    )
    X_test, angles_test, ids_test = process_json_list(test_data, is_train=False)

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angles_train"], angles_train)
    np.save(cache_files["ids_train"], ids_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angles_test"], angles_test)
    np.save(cache_files["ids_test"], ids_test)

    return X_train, y_train, angles_train, ids_train, X_test, angles_test, ids_test


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class ShipIcebergDataset(Dataset):
    def __init__(self, X, y, angles, transform=None):
        self.X = X
        self.y = y
        self.angles = angles
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        angle = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()

        if self.transform:
            img_tensor = self.transform(img_tensor)

        angle_tensor = torch.tensor([angle], dtype=torch.float32)

        if self.y is not None:
            label_tensor = torch.tensor([self.y[idx]], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            return img_tensor, angle_tensor


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------


class HybridSE(nn.Module):
    """
    Squeeze-and-Excitation block with Global Average Pooling.
    """

    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
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


class ISCI_CNN(nn.Module):
    """
    Inter-Stage Calibrated Isomorphic CNN.
    Features: 4-Stage Plain CNN, Inter-Stage Calibration, Dual-Polarity Readout.
    """

    def __init__(self):
        super(ISCI_CNN, self).__init__()

        # --- Backbone ---
        # Stage 1: 3 -> 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(64),
            nn.MaxPool2d(2, 2),  # 75 -> 37
        )

        # Stage 2: 64 -> 128
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(128),
            nn.MaxPool2d(2, 2),  # 37 -> 18
        )

        # Stage 3: 128 -> 128 (Split body and pool for calibration access)
        self.stage3_body = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(128),
        )
        self.stage3_pool = nn.MaxPool2d(2, 2)  # 18 -> 9

        # Stage 4: 128 -> 128
        self.stage4_body = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(128),
        )
        self.stage4_pool = nn.MaxPool2d(2, 2)  # 9 -> 4

        # --- Inter-Stage Calibration ---
        # Maps Stage 4 context (128) to gating weights (128)
        self.calibration_gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.Sigmoid(),
        )

        # --- Isomorphic Dual-Polarity Readout ---
        # Projections map 128 -> 64
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1)

        # --- Classification Head ---
        # Features (256) + Angle (1) = 257
        self.head = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, inc_angle):
        # Forward pass through backbone
        x1 = self.stage1(x)
        x2 = self.stage2(x1)

        # Stage 3 Pre-Pool (18x18)
        x3_pre = self.stage3_body(x2)
        x3_pooled = self.stage3_pool(x3_pre)

        # Stage 4 (4x4)
        x4_pre = self.stage4_body(x3_pooled)
        x4 = self.stage4_pool(x4_pre)

        # --- Calibration ---
        # 1. Extract Context from Stage 4
        c4 = F.adaptive_avg_pool2d(x4, 1).view(x4.size(0), -1)
        # 2. Compute Gate
        g = self.calibration_gate(c4).view(x4.size(0), 128, 1, 1)
        # 3. Calibrate Stage 3 (Pre-Pool)
        x3_calibrated = x3_pre * g

        # --- Readout ---
        # Branch 3 (Calibrated)
        f3 = self.proj3(x3_calibrated)  # (B, 64, 18, 18)
        f3_max = F.adaptive_max_pool2d(f3, 1).view(f3.size(0), -1)
        f3_min = -F.adaptive_max_pool2d(-f3, 1).view(f3.size(0), -1)  # Global Min
        feat3 = torch.cat([f3_max, f3_min], dim=1)  # 128

        # Branch 4
        f4 = self.proj4(x4)  # (B, 64, 4, 4)
        f4_max = F.adaptive_max_pool2d(f4, 1).view(f4.size(0), -1)
        f4_min = -F.adaptive_max_pool2d(-f4, 1).view(f4.size(0), -1)
        feat4 = torch.cat([f4_max, f4_min], dim=1)  # 128

        # Combine
        features = torch.cat([feat3, feat4], dim=1)  # 256

        # --- Head ---
        combined = torch.cat([features, inc_angle], dim=1)  # 257
        out = self.head(combined)
        return out


# -----------------------------------------------------------------------------
# Training & Inference Logic
# -----------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for images, angles, targets in loader:
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))
    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for images, angles, targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))
    return losses.avg


def predict_test(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for images, angles in loader:
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy().flatten())
    return preds


def run_training_and_submission():
    """
    Executes the full 5-fold training pipeline and generates submission.
    """
    set_seed(Config.SEED)

    # Load Data
    X_train, y_train, angles_train, ids_train, X_test, angles_test, ids_test = (
        load_and_process_data()
    )

    # Transforms
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip()])

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    test_preds_accum = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # Split Data
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        ang_tr, ang_val = angles_train[train_idx], angles_train[val_idx]

        # Leak-Free Imputation: Calculate median ONLY on train split
        valid_angles = ang_tr[~np.isnan(ang_tr)]
        median_angle = np.median(valid_angles) if len(valid_angles) > 0 else 0.0

        # Apply to Train and Val
        ang_tr = np.where(np.isnan(ang_tr), median_angle, ang_tr)
        ang_val = np.where(np.isnan(ang_val), median_angle, ang_val)

        # Prepare Datasets
        train_ds = ShipIcebergDataset(X_tr, y_tr, ang_tr, transform=train_transform)
        val_ds = ShipIcebergDataset(X_val, y_val, ang_val, transform=None)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = ISCI_CNN().to(Config.DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, Config.DEVICE
            )
            val_loss = validate(model, val_loader, criterion, Config.DEVICE)

            print(
                f"Fold {fold}, Epoch {epoch}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": best_loss,
                    },
                    is_best=True,
                    fold=fold,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                break

        # Inference on Test Set using Best Model of this Fold
        best_model_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth"
        )
        checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

        # Impute Test angles with Train median (Leak-Free)
        ang_test_imputed = np.where(np.isnan(angles_test), median_angle, angles_test)

        test_ds = ShipIcebergDataset(X_test, None, ang_test_imputed, transform=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        fold_preds = predict_test(model, test_loader, Config.DEVICE)
        test_preds_accum += np.array(fold_preds)

    # Average Predictions
    avg_preds = test_preds_accum / Config.NUM_FOLDS

    # Save Submission
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
