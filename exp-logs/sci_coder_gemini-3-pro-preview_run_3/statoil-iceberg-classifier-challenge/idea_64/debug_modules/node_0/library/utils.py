import os
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name):
    """Configures and returns a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger("MSI-CNN")

# ------------------------------------------------------------------------------
# Data Processing & Caching
# ------------------------------------------------------------------------------


def load_and_process_data(load_cached_data=True, base_dir="./working/idea_64"):
    """
    Loads raw data, processes it into numpy arrays, and caches it.
    Follows strict caching logic: checks for existence, loads if available, else computes and saves.
    """
    os.makedirs(base_dir, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(base_dir, "X_train.npy"),
        "y_train": os.path.join(base_dir, "y_train.npy"),
        "angle_train": os.path.join(base_dir, "angle_train.npy"),
        "ids_train": os.path.join(base_dir, "ids_train.npy"),
        "X_test": os.path.join(base_dir, "X_test.npy"),
        "angle_test": os.path.join(base_dir, "angle_test.npy"),
        "ids_test": os.path.join(base_dir, "ids_test.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(f) for f in cache_files.values())

    if load_cached_data and all_cached:
        logger.info("Loading cached data...")
        try:
            data = {}
            for k, v in cache_files.items():
                data[k] = np.load(v, allow_pickle=True)
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    logger.info("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv("./metadata/train.csv")
    val_meta = pd.read_csv("./metadata/val.csv")
    test_meta = pd.read_csv("./metadata/test.csv")

    # Combine train and val metadata to form the full training set for CV
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Load Raw JSON
    logger.info("Loading raw JSON files...")
    with open("./input/train.json", "r") as f:
        raw_train = json.load(f)
    with open("./input/test.json", "r") as f:
        raw_test = json.load(f)

    def process_subset(meta_df, raw_data_list):
        ids = []
        bands_1 = []
        bands_2 = []
        angles = []
        targets = []

        has_target = "is_iceberg" in meta_df.columns

        # Optimize lookup by creating a map if list is large, but list is ~1600 items, linear scan is ok-ish
        # but direct indexing via 'original_index' is O(1) and safest given metadata generation logic.

        for _, row in meta_df.iterrows():
            idx = int(row["original_index"])
            item = raw_data_list[idx]

            # Safety check
            if item["id"] != row["id"]:
                # Fallback
                item = next(x for x in raw_data_list if x["id"] == row["id"])

            ids.append(item["id"])
            bands_1.append(item["band_1"])
            bands_2.append(item["band_2"])
            angles.append(item["inc_angle"])

            if has_target:
                targets.append(row["is_iceberg"])

        # Convert to numpy and reshape
        X_b1 = np.array(bands_1, dtype=np.float32).reshape(-1, 75, 75)
        X_b2 = np.array(bands_2, dtype=np.float32).reshape(-1, 75, 75)

        # Create 3rd channel: Avg
        X_avg = (X_b1 + X_b2) / 2.0

        # Stack: (N, 3, 75, 75)
        X = np.stack([X_b1, X_b2, X_avg], axis=1)

        # Process angles: 'na' to NaN
        angles_fixed = []
        for a in angles:
            if a == "na":
                angles_fixed.append(np.nan)
            else:
                angles_fixed.append(float(a))
        angles_np = np.array(angles_fixed, dtype=np.float32)

        ids_np = np.array(ids)

        if has_target:
            y_np = np.array(targets, dtype=np.float32)
            return X, y_np, angles_np, ids_np
        else:
            return X, None, angles_np, ids_np

    logger.info("Processing Training Set...")
    X_train, y_train, angle_train, ids_train = process_subset(
        full_train_meta, raw_train
    )

    logger.info("Processing Test Set...")
    X_test, _, angle_test, ids_test = process_subset(test_meta, raw_test)

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angle_train"], angle_train)
    np.save(cache_files["ids_train"], ids_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angle_test"], angle_test)
    np.save(cache_files["ids_test"], ids_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "angle_train": angle_train,
        "ids_train": ids_train,
        "X_test": X_test,
        "angle_test": angle_test,
        "ids_test": ids_test,
    }


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------


class IcebergDataset(Dataset):
    def __init__(self, X, y, angles, transform=None, angle_impute_val=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.angles = angles
        self.transform = transform

        # Handle angle imputation
        if angle_impute_val is not None:
            self.angle_impute_val = angle_impute_val
        else:
            valid_angles = self.angles[~np.isnan(self.angles)]
            self.angle_impute_val = (
                np.median(valid_angles) if len(valid_angles) > 0 else 0.0
            )

        # Fill NaNs
        self.angles_processed = np.array(self.angles)
        mask = np.isnan(self.angles_processed)
        self.angles_processed[mask] = self.angle_impute_val
        self.angles_processed = torch.FloatTensor(self.angles_processed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        angle = self.angles_processed[idx]

        if self.transform:
            # Random Horizontal Flip
            if random.random() > 0.5:
                img = torch.flip(img, [2])
            # Random Vertical Flip
            if random.random() > 0.5:
                img = torch.flip(img, [1])

        if self.y is not None:
            return img, angle, self.y[idx]
        else:
            return img, angle


# ------------------------------------------------------------------------------
# Model Architecture (MSI-CNN)
# ------------------------------------------------------------------------------


class HybridSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MSICNN(nn.Module):
    def __init__(self):
        super(MSICNN, self).__init__()

        # Backbone: Plain CNN
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.se1 = HybridSE(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        self.se2 = HybridSE(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU(0.1, inplace=True)
        self.se3 = HybridSE(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.LeakyReLU(0.1, inplace=True)
        self.se4 = HybridSE(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Readout Projections
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1)

        # Head
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.fc = nn.Linear(257, 1)  # 256 features + 1 angle

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
        x = self.pool1(self.se1(self.act1(self.bn1(self.conv1(x)))))
        x = self.pool2(self.se2(self.act2(self.bn2(self.conv2(x)))))

        x3 = self.se3(self.act3(self.bn3(self.conv3(x))))
        x = self.pool3(x3)

        x4 = self.se4(self.act4(self.bn4(self.conv4(x))))
        x = self.pool4(x4)

        # Isomorphic Readout
        f3 = self.proj3(x3)
        max3 = F.adaptive_max_pool2d(f3, 1).view(f3.size(0), -1)
        min3 = -F.adaptive_max_pool2d(-f3, 1).view(f3.size(0), -1)

        f4 = self.proj4(x4)
        max4 = F.adaptive_max_pool2d(f4, 1).view(f4.size(0), -1)
        min4 = -F.adaptive_max_pool2d(-f4, 1).view(f4.size(0), -1)

        features = torch.cat([max3, min3, max4, min4, angle.view(-1, 1)], dim=1)

        if self.training:
            out = []
            for drop in self.dropouts:
                out.append(self.fc(drop(features)))
            return torch.stack(out, dim=1)
        else:
            return self.fc(features)


# ------------------------------------------------------------------------------
# Training & Evaluation
# ------------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, angles, targets in loader:
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images, angles)  # (B, 5, 1)

        loss = 0
        for i in range(5):
            loss += criterion(outputs[:, i, :], targets)
        loss /= 5.0

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, angles, targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).view(-1, 1)

            output = model(images, angles)  # (B, 1)
            loss = criterion(output, targets)
            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def run_training(epochs=75, batch_size=32, patience=12):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data = load_and_process_data(load_cached_data=True)
    X, y, angles = data["X_train"], data["y_train"], data["angle_train"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds_accum = np.zeros(len(data["X_test"]))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Starting Fold {fold+1}/5")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        ang_tr, ang_val = angles[train_idx], angles[val_idx]

        # Leak-free angle imputation
        valid_ang_tr = ang_tr[~np.isnan(ang_tr)]
        fold_median = np.median(valid_ang_tr) if len(valid_ang_tr) > 0 else 0.0

        train_ds = IcebergDataset(
            X_tr, y_tr, ang_tr, transform=True, angle_impute_val=fold_median
        )
        val_ds = IcebergDataset(
            X_val, y_val, ang_val, transform=False, angle_impute_val=fold_median
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        model = MSICNN().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            # Print full precision as requested
            print(
                f"Fold {fold+1} Epoch {epoch+1} Train Loss: {train_loss} Val Loss: {val_loss}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Predict on Test with Best Model
        model.load_state_dict(best_state)
        model.eval()

        test_ds = IcebergDataset(
            data["X_test"],
            None,
            data["angle_test"],
            transform=False,
            angle_impute_val=fold_median,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        fold_preds = []
        with torch.no_grad():
            for images, angs in test_loader:
                images = images.to(device)
                angs = angs.to(device)
                out = model(images, angs)
                fold_preds.append(torch.sigmoid(out).cpu().numpy())

        test_preds_accum += np.concatenate(fold_preds).flatten() / 5.0

    # Submission
    submission = pd.DataFrame({"id": data["ids_test"], "is_iceberg": test_preds_accum})
    os.makedirs("./submission", exist_ok=True)
    submission.to_csv("./submission/submission.csv", index=False)
    logger.info("Submission saved successfully.")


def main():
    run_training()
