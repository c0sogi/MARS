import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import (
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    NUM_FOLDS,
    INPUT_DIR,
    WORKING_DIR,
    CACHE_DIR,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    TRAIN_JSON,
    TEST_JSON,
    SAMPLE_SUBMISSION,
    TRAIN_META,
    VAL_META,
    TEST_META,
)
from library.utils import set_seed, setup_logger, AverageMeter

# =============================================================================
# Model Architecture
# =============================================================================


class MADSELayer(nn.Module):
    """
    Mean Absolute Deviation Squeeze-and-Excitation Module.
    Computes Global Mean and Global MAD to capture texture robustness.
    """

    def __init__(self, channels, reduction=16):
        super(MADSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 1)

        # Input to MLP is 2 * channels (Mean + MAD)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 1. Global Mean: mu_c
        mu = self.avg_pool(x).view(b, c)  # (B, C)

        # 2. Global MAD: delta_c = Mean(|x - mu|)
        # Expand mu for broadcasting: (B, C, 1, 1)
        mu_expanded = mu.view(b, c, 1, 1)
        # Compute absolute deviation and take mean over spatial dimensions
        mad = torch.abs(x - mu_expanded).mean(dim=[2, 3])  # (B, C)

        # 3. Concatenate Statistics
        stats = torch.cat([mu, mad], dim=1)  # (B, 2C)

        # 4. Excitation
        scale = self.fc(stats).view(b, c, 1, 1)

        return x * scale


class RTICNN(nn.Module):
    """
    Robust-Texture Isomorphic CNN.
    4-Stage Plain CNN with MAD-SE and Corrected Decoupled Isomorphic Readout.
    """

    def __init__(self):
        super(RTICNN, self).__init__()

        # --- Stage 1 ---
        # 75x75 -> 37x37
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            MADSELayer(64, reduction=16),
            nn.MaxPool2d(2),
        )

        # --- Stage 2 ---
        # 37x37 -> 18x18
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            MADSELayer(128, reduction=16),
            nn.MaxPool2d(2),
        )

        # --- Stage 3 ---
        # 18x18 -> 9x9
        # Split definition to allow feature extraction
        self.stage3_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            MADSELayer(128, reduction=16),
        )
        self.stage3_pool = nn.MaxPool2d(2)

        # --- Stage 4 ---
        # 9x9 -> 4x4
        self.stage4_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            MADSELayer(128, reduction=16),
        )
        self.stage4_pool = nn.MaxPool2d(2)

        # --- Readout Projections ---
        # Decoupled 1x1 convolutions for Stage 3 and 4
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1, bias=True)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1, bias=True)

        # --- Classification Head ---
        # Features: (64*2 from Stage 3) + (64*2 from Stage 4) = 256
        # + 1 Incidence Angle = 257
        self.classifier = nn.Sequential(
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
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # Stage 1
        x = self.stage1(x)

        # Stage 2
        x = self.stage2(x)

        # Stage 3
        x = self.stage3_conv(x)
        feat3 = x  # Save for readout
        x = self.stage3_pool(x)

        # Stage 4
        x = self.stage4_conv(x)
        feat4 = x  # Save for readout
        x = self.stage4_pool(x)

        # Readout Stage 3: Isomorphic Dual-Polarity
        p3 = self.proj3(feat3)  # (B, 64, H, W)
        max3 = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        min3 = -F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)

        # Readout Stage 4: Isomorphic Dual-Polarity
        p4 = self.proj4(feat4)
        max4 = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)
        min4 = -F.adaptive_max_pool2d(-p4, 1).view(p4.size(0), -1)

        # Feature Fusion
        img_feats = torch.cat([max3, min3, max4, min4], dim=1)  # 256

        # Append Raw Incidence Angle
        angle = angle.view(-1, 1)
        combined = torch.cat([img_feats, angle], dim=1)  # 257

        return self.classifier(combined)


# =============================================================================
# Data Processing
# =============================================================================


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays, and caches it.
    Returns:
        X_train (np.ndarray): (N_train, 3, 75, 75)
        y_train (np.ndarray): (N_train,)
        angle_train (np.ndarray): (N_train,) with NaNs
        X_test (np.ndarray): (N_test, 3, 75, 75)
        ids_test (np.ndarray): (N_test,)
        angle_test (np.ndarray): (N_test,) with NaNs
    """
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "angle_train": os.path.join(CACHE_DIR, "angle_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "ids_test": os.path.join(CACHE_DIR, "ids_test.npy"),
        "angle_test": os.path.join(CACHE_DIR, "angle_test.npy"),
    }

    # Check cache
    if load_cached_data:
        all_exist = all(os.path.exists(f) for f in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            return (
                np.load(cache_files["X_train"]),
                np.load(cache_files["y_train"]),
                np.load(cache_files["angle_train"]),
                np.load(cache_files["X_test"]),
                np.load(cache_files["ids_test"]),
                np.load(cache_files["angle_test"]),
            )

    print("Processing data from scratch...")

    # Load Metadata to align with IDs
    train_meta = pd.read_csv(TRAIN_META)
    val_meta = pd.read_csv(VAL_META)
    # Combine train and val for CV splitting later
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    test_meta = pd.read_csv(TEST_META)

    # Load Raw JSON
    print(f"Loading {TRAIN_JSON}...")
    with open(TRAIN_JSON, "r") as f:
        raw_train = json.load(f)
    train_map = {item["id"]: item for item in raw_train}

    print(f"Loading {TEST_JSON}...")
    with open(TEST_JSON, "r") as f:
        raw_test = json.load(f)
    test_map = {item["id"]: item for item in raw_test}

    # Helper to process a dataframe
    def process_subset(df, data_map, is_train=True):
        X = []
        angles = []
        ids = []
        targets = []

        for idx, row in df.iterrows():
            img_id = row["id"]
            item = data_map[img_id]

            # Bands
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            avg = (b1 + b2) / 2.0

            # Stack: (3, 75, 75)
            img = np.stack([b1, b2, avg], axis=0)
            X.append(img)

            # Angle
            ang = item["inc_angle"]
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))

            ids.append(img_id)

            if is_train:
                targets.append(row["is_iceberg"])

        return (
            np.array(X, dtype=np.float32),
            np.array(angles, dtype=np.float32),
            np.array(ids),
            np.array(targets, dtype=np.float32),
        )

    # Process Train (Full)
    X_train, angle_train, _, y_train = process_subset(
        full_train_meta, train_map, is_train=True
    )

    # Process Test
    X_test, angle_test, ids_test, _ = process_subset(
        test_meta, test_map, is_train=False
    )

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angle_train"], angle_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["ids_test"], ids_test)
    np.save(cache_files["angle_test"], angle_test)

    return X_train, y_train, angle_train, X_test, ids_test, angle_test


class IcebergDataset(Dataset):
    def __init__(self, X, y, angles, transform=None):
        self.X = X
        self.y = y
        self.angles = angles
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.X[idx])  # (3, 75, 75)
        angle = self.angles[idx]

        if self.transform:
            img = self.transform(img)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, torch.tensor(angle, dtype=torch.float32), label
        else:
            return img, torch.tensor(angle, dtype=torch.float32)


# =============================================================================
# Training & Inference
# =============================================================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for inputs, angles, targets in loader:
        inputs = inputs.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs, angles)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    preds = []
    true_labels = []

    with torch.no_grad():
        for inputs, angles, targets in loader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            true_labels.extend(targets.cpu().numpy())

    return losses.avg, np.array(preds), np.array(true_labels)


def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for inputs, angles in loader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            outputs = model(inputs, angles)
            preds.extend(torch.sigmoid(outputs).cpu().numpy())

    return np.array(preds)


def run_training():
    set_seed(SEED)
    logger = setup_logger(os.path.join(WORKING_DIR, "training.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    X_full, y_full, angle_full, X_test, ids_test, angle_test = process_data(
        load_cached_data=True
    )

    # 2. Cross Validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(X_full))
    test_preds_accum = np.zeros((len(X_test), 1))

    # Augmentation for training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"\n--- Fold {fold + 1}/{NUM_FOLDS} ---")

        # Split Data
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]
        angle_train, angle_val = angle_full[train_idx], angle_full[val_idx]

        # Impute Angles (Leak-free: compute median on train, apply to all)
        # Note: angle_train contains NaNs.
        train_angle_median = np.nanmedian(angle_train)

        # Fill NaNs
        angle_train_filled = np.nan_to_num(angle_train, nan=train_angle_median)
        angle_val_filled = np.nan_to_num(angle_val, nan=train_angle_median)
        # For test set inference in this fold, use this fold's median
        angle_test_filled = np.nan_to_num(angle_test, nan=train_angle_median)

        # Datasets
        train_ds = IcebergDataset(
            X_train, y_train, angle_train_filled, transform=train_transform
        )
        val_ds = IcebergDataset(X_val, y_val, angle_val_filled, transform=None)
        test_ds = IcebergDataset(X_test, None, angle_test_filled, transform=None)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Model Setup
        model = RTICNN().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold}.pth")

        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, _, _ = validate(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}"
                )

            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load Best Model
        model.load_state_dict(torch.load(best_model_path))

        # OOF Prediction
        _, val_preds, _ = validate(model, val_loader, criterion, device)
        oof_preds[val_idx] = val_preds.flatten()

        # Test Prediction (No TTA)
        fold_test_preds = predict(model, test_loader, device)
        test_preds_accum += fold_test_preds

        logger.info(f"Fold {fold+1} Best Val Loss: {best_val_loss:.6f}")

    # Calculate OOF Score
    oof_loss = log_loss(y_full, oof_preds)
    logger.info(f"\nOverall OOF Log Loss: {oof_loss:.6f}")

    # Average Test Predictions
    avg_test_preds = test_preds_accum / NUM_FOLDS

    # Create Submission
    submission_df = pd.DataFrame(
        {"id": ids_test, "is_iceberg": avg_test_preds.flatten()}
    )

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    run_training()
