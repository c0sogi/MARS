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
from sklearn.metrics import log_loss


class Config:
    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_73"
    SUBMISSION_DIR = "./submission"

    # Files
    TRAIN_JSON = "train.json"
    TEST_JSON = "test.json"

    # Data Specs
    IMG_SIZE = 75
    CHANNELS = 3

    # Model Specs
    BASE_WIDTH = 64
    DROPOUT_RATE = 0.5

    # Training Specs
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 75
    PATIENCE = 12
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Ensure directories exist and set random seeds."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        torch.manual_seed(cls.SEED)
        np.random.seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)


# --- Model Architecture ---


class HybridSE(nn.Module):
    """Hybrid Squeeze-and-Excitation block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
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


class ADSICNN(nn.Module):
    """
    Asymmetric Dual-Statistic Isomorphic CNN.
    Features 4-stage Plain CNN backbone with asymmetric readouts for
    Shadow (Min-Pool) and Texture (MAD-Pool) extraction.
    """

    def __init__(self):
        super().__init__()

        def make_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True),
                HybridSE(out_c),
                nn.MaxPool2d(2),
            )

        # Stage 1: 75 -> 37
        self.stage1 = make_block(Config.CHANNELS, Config.BASE_WIDTH)
        # Stage 2: 37 -> 18
        self.stage2 = make_block(Config.BASE_WIDTH, 128)

        # Stage 3: 18 -> 9 (High resolution for Shadows)
        self.stage3_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(128),
        )
        self.stage3_pool = nn.MaxPool2d(2)

        # Stage 4: 9 -> 4 (Global receptive field for Texture)
        self.stage4_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(128),
        )
        self.stage4_pool = nn.MaxPool2d(2)

        # Decoupled Projections
        self.proj_s3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj_s4 = nn.Conv2d(128, 64, kernel_size=1)

        # Classification Head
        # Features: S3_Max(64) + S3_Min(64) + S4_Max(64) + S4_MAD(64) + Angle(1) = 257
        self.head = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

    def forward(self, x, angle):
        # Backbone
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3_conv(x)
        s3_out = self.stage3_pool(x)  # 9x9

        x = self.stage4_conv(s3_out)
        s4_out = self.stage4_pool(x)  # 4x4

        # Readout Stage 3 (Shape & Shadow)
        p3 = self.proj_s3(s3_out)  # (B, 64, 9, 9)
        s3_max = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        s3_min = -F.adaptive_max_pool2d(-p3, 1).view(
            p3.size(0), -1
        )  # Min pooling via negative max

        # Readout Stage 4 (Existence & Texture)
        p4 = self.proj_s4(s4_out)  # (B, 64, 4, 4)
        s4_max = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)

        # MAD Pooling (Mean Absolute Deviation)
        p4_mean = F.adaptive_avg_pool2d(p4, 1)
        mad = torch.abs(p4 - p4_mean)
        s4_mad = F.adaptive_avg_pool2d(mad, 1).view(p4.size(0), -1)

        # Feature Fusion
        angle = angle.view(-1, 1)
        features = torch.cat([s3_max, s3_min, s4_max, s4_mad, angle], dim=1)

        return self.head(features)


# --- Dataset ---


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, transform=False):
        self.X = torch.FloatTensor(X)
        self.angles = torch.FloatTensor(angles)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        angle = self.angles[idx]

        if self.transform:
            # Random Horizontal Flip
            if torch.rand(1) < 0.5:
                img = torch.flip(img, [1])
            # Random Vertical Flip
            if torch.rand(1) < 0.5:
                img = torch.flip(img, [2])

        if self.y is not None:
            return img, angle, self.y[idx]
        return img, angle


# --- Data Processing ---


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes into numpy arrays (3 channels),
    imputes missing angles, and caches results.
    """
    Config.setup()

    # Cache paths
    cache_files = {
        "X_train": os.path.join(Config.WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.WORKING_DIR, "y_train.npy"),
        "angle_train": os.path.join(Config.WORKING_DIR, "angle_train.npy"),
        "X_test": os.path.join(Config.WORKING_DIR, "X_test.npy"),
        "angle_test": os.path.join(Config.WORKING_DIR, "angle_test.npy"),
        "ids_test": os.path.join(Config.WORKING_DIR, "ids_test.npy"),
    }

    # Try loading cache
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        return (
            np.load(cache_files["X_train"]),
            np.load(cache_files["y_train"]),
            np.load(cache_files["angle_train"]),
            np.load(cache_files["X_test"]),
            np.load(cache_files["angle_test"]),
            np.load(cache_files["ids_test"], allow_pickle=True),
        )

    # Process from scratch
    with open(os.path.join(Config.INPUT_DIR, Config.TRAIN_JSON), "r") as f:
        train_data = json.load(f)
    with open(os.path.join(Config.INPUT_DIR, Config.TEST_JSON), "r") as f:
        test_data = json.load(f)

    def parse_json(data, has_label=True):
        X, angles, ids, y = [], [], [], []
        for item in data:
            # Reshape flattened bands
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            b3 = (b1 + b2) / 2.0  # Synthetic average
            X.append(np.stack([b1, b2, b3], axis=0))

            ids.append(item["id"])

            ang = item["inc_angle"]
            angles.append(np.nan if ang == "na" else float(ang))

            if has_label:
                y.append(item["is_iceberg"])

        return (
            np.array(X),
            np.array(angles),
            np.array(ids),
            np.array(y) if has_label else None,
        )

    X_train, angle_train, _, y_train = parse_json(train_data, True)
    X_test, angle_test, ids_test, _ = parse_json(test_data, False)

    # Impute missing angles (Median of Train)
    train_median = np.nanmedian(angle_train)
    angle_train = np.where(np.isnan(angle_train), train_median, angle_train)
    angle_test = np.where(np.isnan(angle_test), train_median, angle_test)

    # Save cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angle_train"], angle_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angle_test"], angle_test)
    np.save(cache_files["ids_test"], ids_test)

    return X_train, y_train, angle_train, X_test, angle_test, ids_test


# --- Training & Inference ---


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for imgs, angles, labels in loader:
        imgs, angles, labels = (
            imgs.to(Config.DEVICE),
            angles.to(Config.DEVICE),
            labels.to(Config.DEVICE).unsqueeze(1),
        )

        optimizer.zero_grad()
        outputs = model(imgs, angles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for imgs, angles, labels in loader:
            imgs, angles, labels = (
                imgs.to(Config.DEVICE),
                angles.to(Config.DEVICE),
                labels.to(Config.DEVICE).unsqueeze(1),
            )
            outputs = model(imgs, angles)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)

            preds_list.extend(torch.sigmoid(outputs).cpu().numpy())
            targets_list.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    # Calculate precise log loss
    y_true = np.array(targets_list).flatten()
    y_pred = np.array(preds_list).flatten()
    metric_score = log_loss(y_true, y_pred)

    return epoch_loss, metric_score


def run_training(load_cached_data=True):
    """
    Main execution function: Loads data, runs 5-Fold CV, and generates submission.
    """
    X, y, angles, X_test, angles_test, ids_test = process_data(load_cached_data)

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    test_preds_accum = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{Config.N_FOLDS} ---")

        # Prepare Fold Data
        train_ds = IcebergDataset(
            X[train_idx], angles[train_idx], y[train_idx], transform=True
        )
        val_ds = IcebergDataset(
            X[val_idx], angles[val_idx], y[val_idx], transform=False
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Initialize Model
        model = ADSICNN().to(Config.DEVICE)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_metric = validate(model, val_loader, criterion)

            print(
                f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val LogLoss: {val_metric:.15f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Inference on Test Set with Best Model
        model.load_state_dict(best_state)
        model.eval()

        test_ds = IcebergDataset(X_test, angles_test, transform=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        fold_preds = []
        with torch.no_grad():
            for imgs, angs in test_loader:
                imgs, angs = imgs.to(Config.DEVICE), angs.to(Config.DEVICE)
                out = model(imgs, angs)
                fold_preds.extend(torch.sigmoid(out).cpu().numpy())

        test_preds_accum += np.array(fold_preds).flatten() / Config.N_FOLDS

    # Generate Submission
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": test_preds_accum})
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission generated at: {sub_path}")
