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
import copy
import random


# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Paths
    INPUT_DIR = "./input"
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    WORK_DIR = "./working/idea_43"
    CACHE_FILE = os.path.join(WORK_DIR, "processed_data.npz")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Hyperparameters
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 50  # High epoch count, controlled by Early Stopping
    LEARNING_RATE = 2e-4
    PATIENCE = 10
    NUM_WORKERS = 2

    # Model Params
    IMG_SIZE = 75
    IN_CHANNELS = 3  # Band1, Band2, Mean


# Ensure directories exist
os.makedirs(Config.WORK_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# ==========================================
# UTILITIES
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(Config.SEED)


# ==========================================
# DATA PROCESSING
# ==========================================
def process_data(load_cached_data=True):
    """
    Loads and processes data with caching mechanism.
    Returns: X_train, y_train, inc_train, X_test, inc_test, test_ids
    """
    if load_cached_data and os.path.exists(Config.CACHE_FILE):
        print(f"Loading cached data from {Config.CACHE_FILE}...")
        data = np.load(Config.CACHE_FILE)
        return (
            data["X_train"],
            data["y_train"],
            data["inc_train"],
            data["X_test"],
            data["inc_test"],
            data["test_ids"],
        )

    print("Processing data from scratch...")

    # Load JSONs
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    # Helper to reshape bands
    def get_images(df):
        imgs = []
        for i, row in df.iterrows():
            # Band 1 and Band 2
            b1 = np.array(row["band_1"]).reshape(75, 75)
            b2 = np.array(row["band_2"]).reshape(75, 75)
            # Band 3: Mean
            b3 = (b1 + b2) / 2.0

            # Stack: (75, 75, 3)
            img = np.stack([b1, b2, b3], axis=-1)
            imgs.append(img)
        return np.array(imgs)

    print("Constructing images...")
    X_train = get_images(df_train)
    X_test = get_images(df_test)

    y_train = df_train["is_iceberg"].values.astype(np.float32)
    test_ids = df_test["id"].values

    # Process Incidence Angle
    # Replace 'na' with 0.0 (or mean of valid).
    # Since the model has a metadata branch, we want to preserve the info.
    # A common strategy is to fill with mean, but let's check valid ones first.

    def process_inc_angle(series):
        # Convert to numeric, errors='coerce' turns 'na' to NaN
        vals = pd.to_numeric(series, errors="coerce").values
        # Fill NaN with mean of valid data
        mask = ~np.isnan(vals)
        mean_val = np.mean(vals[mask])
        vals[~mask] = mean_val
        return vals.astype(np.float32)

    inc_train = process_inc_angle(df_train["inc_angle"])
    inc_test = process_inc_angle(df_test["inc_angle"])

    # Global Normalization
    # Compute stats on TRAIN only
    print("Applying global normalization...")
    # Flatten to (N*H*W, 3) to compute stats per channel
    train_flat = X_train.reshape(-1, 3)

    min_vals = train_flat.min(axis=0)
    max_vals = train_flat.max(axis=0)

    # Apply MinMax Scaling: (X - min) / (max - min)
    # Allow values outside [0, 1] for test/val if they exceed train bounds
    X_train = (X_train - min_vals) / (max_vals - min_vals)
    X_test = (X_test - min_vals) / (max_vals - min_vals)

    # Transpose to PyTorch format: (N, C, H, W)
    X_train = X_train.transpose(0, 3, 1, 2).astype(np.float32)
    X_test = X_test.transpose(0, 3, 1, 2).astype(np.float32)

    print(f"Saving processed data to {Config.CACHE_FILE}...")
    np.savez(
        Config.CACHE_FILE,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        X_test=X_test,
        inc_test=inc_test,
        test_ids=test_ids,
    )

    return X_train, y_train, inc_train, X_test, inc_test, test_ids


# ==========================================
# DATASET
# ==========================================
class IcebergDataset(Dataset):
    def __init__(self, X, inc_angles, y=None, transform=False):
        self.X = X
        self.inc_angles = inc_angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # (C, H, W)
        inc = self.inc_angles[idx]

        if self.transform:
            # Random Rotation (0, 90, 180, 270)
            k = random.randint(0, 3)
            img = np.rot90(img, k, axes=(1, 2))  # rotate spatial dims

            # Random Horizontal Flip
            if random.random() > 0.5:
                img = np.flip(img, axis=2)  # flip width

        # Convert to tensor
        img_tensor = torch.from_numpy(img.copy()).float()
        inc_tensor = torch.tensor([inc], dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor([self.y[idx]], dtype=torch.float32)
            return img_tensor, inc_tensor, label
        else:
            return img_tensor, inc_tensor


# ==========================================
# MODEL: DMWB-Net
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Mixed Pooling for Attention Logic (Max + Avg)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class WideBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WideBlock, self).__init__()
        # Delayed Integration: Wide Conv -> BN -> ReLU
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pre-Pooling Attention
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.cbam(x)

        # Dual-Stream Pooling (Max + Min)
        # Min Pooling = -MaxPool(-x)
        max_p = F.max_pool2d(x, 2)
        min_p = -F.max_pool2d(-x, 2)

        # Concatenate -> Output channels doubles for next stage input
        out = torch.cat([max_p, min_p], dim=1)
        return out


class DMWBNet(nn.Module):
    def __init__(self):
        super(DMWBNet, self).__init__()

        # --- Visual Branch ---
        # Stage 1: Input 3 -> 128 (Wide)
        self.stage1_conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.stage1_bn = nn.BatchNorm2d(128)
        self.stage1_relu = nn.ReLU(inplace=True)
        self.stage1_cbam = CBAM(128)
        # Pooling 1: 128 -> 256

        # Stage 2: Input 256 -> 128
        self.stage2 = WideBlock(256, 128)

        # Stage 3: Input 256 -> 128
        self.stage3 = WideBlock(256, 128)

        # Stage 4: Input 256 -> 128
        self.stage4 = WideBlock(256, 128)

        # Output of Stage 4 is 256 channels (due to dual pooling)
        # Size: 75 -> 37 -> 18 -> 9 -> 4

        # --- Readout ---
        # Path A: Spatial Context
        self.path_a_conv = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        # Flattened size: 64 * 4 * 4 = 1024

        # Path B: Robust Intensity (GAP)
        self.path_b_pool = nn.AdaptiveAvgPool2d(1)
        # Size: 256

        # --- Metadata Branch ---
        self.meta_fc1 = nn.Linear(1, 32)
        self.meta_fc2 = nn.Linear(32, 32)
        self.meta_bn = nn.BatchNorm1d(32)

        # --- Fusion Head ---
        # Input: 1024 (Path A) + 256 (Path B) + 32 (Meta) = 1312
        self.fusion_fc1 = nn.Linear(1312, 512)
        self.fusion_bn = nn.BatchNorm1d(512)
        self.fusion_drop = nn.Dropout(0.5)
        self.fusion_out = nn.Linear(512, 1)

    def forward(self, x_img, x_meta):
        # Visual Branch
        # Stage 1
        x = self.stage1_relu(self.stage1_bn(self.stage1_conv(x_img)))
        x = self.stage1_cbam(x)
        # Dual Pool
        x = torch.cat([F.max_pool2d(x, 2), -F.max_pool2d(-x, 2)], dim=1)

        # Stages 2-4
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)  # Output (B, 256, 4, 4)

        # Readout Path A
        xa = self.path_a_conv(x)  # (B, 64, 4, 4)
        xa = xa.view(xa.size(0), -1)  # (B, 1024)

        # Readout Path B
        xb = self.path_b_pool(x).view(x.size(0), -1)  # (B, 256)

        # Metadata Branch
        xm = F.relu(self.meta_fc1(x_meta))
        xm = self.meta_fc2(xm)
        xm = F.relu(self.meta_bn(xm))  # (B, 32)

        # Fusion
        fused = torch.cat([xa, xb, xm], dim=1)
        x = F.relu(self.fusion_bn(self.fusion_fc1(fused)))
        x = self.fusion_drop(x)
        x = self.fusion_out(x)

        return torch.sigmoid(x)


# ==========================================
# TRAINING
# ==========================================
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    X, y, inc, X_test, inc_test, test_ids = process_data()

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold + 1}/{Config.N_FOLDS} ===")

        # Split
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        inc_tr, inc_val = inc[train_idx], inc[val_idx]

        # Datasets
        train_ds = IcebergDataset(X_tr, inc_tr, y_tr, transform=True)
        val_ds = IcebergDataset(X_val, inc_val, y_val, transform=False)

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
        model = DMWBNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        # Training Loop
        best_loss = float("inf")
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            model.train()
            running_loss = 0.0

            for inputs, angles, labels in train_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs, angles)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_ds)

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds_fold = []

            with torch.no_grad():
                for inputs, angles, labels in val_loader:
                    inputs = inputs.to(device)
                    angles = angles.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs, angles)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_preds_fold.extend(outputs.cpu().numpy().flatten())

            val_loss = val_loss / len(val_ds)
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Save best model
        model_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
        torch.save(best_model_wts, model_path)
        print(f"Saved best model for fold {fold} to {model_path}")

        # Load best weights for inference
        model.load_state_dict(best_model_wts)
        model.eval()

        # Predict on Test
        test_ds = IcebergDataset(X_test, inc_test, transform=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        fold_test_preds = []
        with torch.no_grad():
            for inputs, angles in test_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)
                outputs = model(inputs, angles)
                fold_test_preds.extend(outputs.cpu().numpy().flatten())

        test_preds += np.array(fold_test_preds) / Config.N_FOLDS

    # Save Submission
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds})
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    run_training()
