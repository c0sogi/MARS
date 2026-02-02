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
import random
import copy

# ==========================================
# CONFIGURATION
# ==========================================


class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_33"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Data
    IMG_SIZE = 75
    NUM_CHANNELS = 3  # Band 1, Band 2, Mean

    # Model
    FILTERS = 128
    DROPOUT = 0.5

    # Training
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 8
    NUM_WORKERS = 2

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# DATA PROCESSING
# ==========================================


def load_and_process_data(load_cached_data=True, limit_data=None):
    """
    Loads raw JSON data, performs global scaling, creates 3-channel images,
    and returns numpy arrays. Caches results to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.
        limit_data (int, optional): If set, limits the dataset size for debugging.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X_train = data["X_train"]
        y_train = data["y_train"]
        inc_angle_train = data["inc_angle_train"]
        X_test = data["X_test"]
        inc_angle_test = data["inc_angle_test"]
        test_ids = data["test_ids"]

        if limit_data:
            return (
                X_train[:limit_data],
                y_train[:limit_data],
                inc_angle_train[:limit_data],
                X_test[:limit_data],
                inc_angle_test[:limit_data],
                test_ids[:limit_data],
            )
        return X_train, y_train, inc_angle_train, X_test, inc_angle_test, test_ids

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Combine train and val for full training set processing
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Load JSONs
    print("Loading train.json...")
    with open(os.path.join(Config.INPUT_DIR, "train.json"), "r") as f:
        train_json = json.load(f)
    train_df_raw = pd.DataFrame(train_json)

    print("Loading test.json...")
    with open(os.path.join(Config.INPUT_DIR, "test.json"), "r") as f:
        test_json = json.load(f)
    test_df_raw = pd.DataFrame(test_json)

    # Align with metadata
    train_df_raw = train_df_raw.set_index("id")
    full_train_meta = full_train_meta.set_index("id")
    df_train = full_train_meta.join(
        train_df_raw[["band_1", "band_2"]], how="left"
    ).reset_index()

    test_df_raw = test_df_raw.set_index("id")
    test_meta = test_meta.set_index("id")
    df_test = test_meta.join(
        test_df_raw[["band_1", "band_2"]], how="left"
    ).reset_index()

    # Extract Bands
    def get_bands(df):
        b1 = np.stack([np.array(b) for b in df["band_1"]]).reshape(-1, 75, 75)
        b2 = np.stack([np.array(b) for b in df["band_2"]]).reshape(-1, 75, 75)
        return b1, b2

    train_b1, train_b2 = get_bands(df_train)
    test_b1, test_b2 = get_bands(df_test)

    # Global Scaling Statistics (from Training Set ONLY)
    b1_min = train_b1.min()
    b1_max = train_b1.max()
    b2_min = train_b2.min()
    b2_max = train_b2.max()

    print(
        f"Global Stats - B1: [{b1_min:.4f}, {b1_max:.4f}], B2: [{b2_min:.4f}, {b2_max:.4f}]"
    )

    # Scale function
    def scale(bands, vmin, vmax):
        return (bands - vmin) / (vmax - vmin)

    # Process Train
    train_b1_s = scale(train_b1, b1_min, b1_max)
    train_b2_s = scale(train_b2, b2_min, b2_max)
    train_avg_s = (train_b1_s + train_b2_s) / 2.0

    X_train = np.stack([train_b1_s, train_b2_s, train_avg_s], axis=1)  # (N, 3, 75, 75)

    # Process Test
    test_b1_s = scale(test_b1, b1_min, b1_max)
    test_b2_s = scale(test_b2, b2_min, b2_max)
    test_avg_s = (test_b1_s + test_b2_s) / 2.0

    X_test = np.stack([test_b1_s, test_b2_s, test_avg_s], axis=1)

    # Targets and Meta
    y_train = df_train["is_iceberg"].values.astype(np.float32)

    inc_angle_train = df_train["inc_angle"].values
    inc_angle_test = df_test["inc_angle"].values

    mean_angle = np.nanmean(inc_angle_train)
    inc_angle_train = np.nan_to_num(inc_angle_train, nan=mean_angle).astype(np.float32)
    inc_angle_test = np.nan_to_num(inc_angle_test, nan=mean_angle).astype(np.float32)

    test_ids = df_test["id"].values

    # Save
    np.savez(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        inc_angle_train=inc_angle_train,
        X_test=X_test,
        inc_angle_test=inc_angle_test,
        test_ids=test_ids,
    )

    print("Data processed and cached.")

    if limit_data:
        return (
            X_train[:limit_data],
            y_train[:limit_data],
            inc_angle_train[:limit_data],
            X_test[:limit_data],
            inc_angle_test[:limit_data],
            test_ids[:limit_data],
        )

    return X_train, y_train, inc_angle_train, X_test, inc_angle_test, test_ids


class IcebergDataset(Dataset):
    def __init__(self, X, inc_angles, y=None, transform=False):
        self.X = X
        self.inc_angles = inc_angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # (3, 75, 75)
        angle = self.inc_angles[idx]

        if self.transform:
            # Rotation
            k = random.choice([0, 1, 2, 3])
            img = np.rot90(img, k, axes=(1, 2))

            # Horizontal Flip
            if random.random() > 0.5:
                img = np.flip(img, axis=2)

        # Convert to tensor
        img_tensor = torch.from_numpy(img.copy()).float()
        angle_tensor = torch.tensor([angle], dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor([self.y[idx]], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


# ==========================================
# MODEL ARCHITECTURE
# ==========================================


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channels = channels

        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        # Mixed Pooling (Max + Avg)
        b, c, h, w = x.size()

        # Global Avg Pool
        avg_pool = F.avg_pool2d(x, (h, w)).view(b, c)
        # Global Max Pool
        max_pool = F.max_pool2d(x, (h, w)).view(b, c)

        channel_att = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(
            b, c, 1, 1
        )
        x = x * channel_att

        # Spatial Attention
        # Channel Pool (Max + Avg)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)

        spatial_att = self.sigmoid(self.conv_spatial(spatial_in))
        x = x * spatial_att

        return x


class DualPooling(nn.Module):
    def __init__(self):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)
        # Min Pooling implemented as -MaxPool(-x)

    def forward(self, x):
        max_p = self.max_pool(x)
        min_p = -self.max_pool(-x)
        return torch.cat([max_p, min_p], dim=1)


class MDS_WBN(nn.Module):
    def __init__(self):
        super(MDS_WBN, self).__init__()

        # Block 1
        # Input 3 -> 128
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.cbam1 = CBAM(128)
        self.pool1 = DualPooling()  # Output 256

        # Block 2
        # Input 256 -> 128 (Delayed Integration)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAM(128)
        self.pool2 = DualPooling()  # Output 256

        # Block 3
        # Input 256 -> 128
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)
        self.pool3 = DualPooling()  # Output 256

        # Block 4
        # Input 256 -> 128
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.cbam4 = CBAM(128)
        self.pool4 = DualPooling()  # Output 256

        # Readout Path A
        # Block 4 Out (256) -> Conv (48)
        self.readout_conv = nn.Conv2d(256, 48, kernel_size=3, padding=1)

        # Metadata Branch
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )

        # Fusion Head
        # Visual Dim:
        # Path A: 4*4*48 = 768
        # Path B: 256 (B2) + 256 (B3) + 256 (B4) = 768
        # Total Visual: 1536
        # Meta: 32
        # Total: 1568
        self.head = nn.Sequential(
            nn.Linear(1568, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(128, 1),
        )

    def forward(self, x, inc_angle):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = self.pool1(x)  # -> 37x37, 256ch

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x2_out = self.pool2(x)  # -> 18x18, 256ch

        # Block 3
        x = F.relu(self.bn3(self.conv3(x2_out)))
        x = self.cbam3(x)
        x3_out = self.pool3(x)  # -> 9x9, 256ch

        # Block 4
        x = F.relu(self.bn4(self.conv4(x3_out)))
        x = self.cbam4(x)
        x4_out = self.pool4(x)  # -> 4x4, 256ch

        # Readout Path A
        path_a = self.readout_conv(x4_out)  # 4x4x48
        path_a = path_a.view(path_a.size(0), -1)  # Flatten -> 768

        # Readout Path B (GAP)
        gap2 = F.adaptive_avg_pool2d(x2_out, (1, 1)).view(x2_out.size(0), -1)  # 256
        gap3 = F.adaptive_avg_pool2d(x3_out, (1, 1)).view(x3_out.size(0), -1)  # 256
        gap4 = F.adaptive_avg_pool2d(x4_out, (1, 1)).view(x4_out.size(0), -1)  # 256
        path_b = torch.cat([gap2, gap3, gap4], dim=1)  # 768

        # Visual Fusion
        visual_feat = torch.cat([path_a, path_b], dim=1)  # 1536

        # Meta Branch
        meta_feat = self.meta_mlp(inc_angle)

        # Final Fusion
        combined = torch.cat([visual_feat, meta_feat], dim=1)
        out = self.head(combined)

        return torch.sigmoid(out)


# ==========================================
# TRAINING
# ==========================================


def train_one_epoch(model, loader, optimizer, criterion, device):
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

    return running_loss / len(loader.dataset)


def run_training(epochs=Config.EPOCHS, debug=False):
    """
    Runs the full 5-fold cross-validation training and generates submission.

    Args:
        epochs (int): Number of epochs per fold.
        debug (bool): If True, limits data size for quick testing.
    """
    Config.set_seed(Config.SEED)

    limit = 100 if debug else None

    # Load Data
    X, y, inc, X_test, inc_test, test_ids = load_and_process_data(limit_data=limit)

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    test_preds = np.zeros((len(X_test), 1))

    # Create submission directory
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation on {Config.DEVICE}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{Config.N_FOLDS}")

        # Split
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        inc_train_fold, inc_val_fold = inc[train_idx], inc[val_idx]

        # Datasets
        train_ds = IcebergDataset(
            X_train_fold, inc_train_fold, y_train_fold, transform=True
        )
        val_ds = IcebergDataset(X_val_fold, inc_val_fold, y_val_fold, transform=False)

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

        # Model
        model = MDS_WBN().to(Config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        criterion = nn.BCELoss()

        # Training Loop
        best_loss = float("inf")
        patience_counter = 0
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, Config.DEVICE
            )
            val_loss = validate(model, val_loader, criterion, Config.DEVICE)

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered")
                break

        # Load best weights
        model.load_state_dict(best_model_wts)

        # Predict on Test
        test_ds = IcebergDataset(X_test, inc_test, transform=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        model.eval()
        fold_preds = []
        with torch.no_grad():
            for images, angles in test_loader:
                images, angles = images.to(Config.DEVICE), angles.to(Config.DEVICE)
                outputs = model(images, angles)
                fold_preds.extend(outputs.cpu().numpy())

        test_preds += np.array(fold_preds)

    # Average predictions
    test_preds /= Config.N_FOLDS

    # Save submission
    sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds.flatten()})
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
