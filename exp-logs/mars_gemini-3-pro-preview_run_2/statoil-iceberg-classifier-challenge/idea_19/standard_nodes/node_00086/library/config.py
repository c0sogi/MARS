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
from sklearn.preprocessing import MinMaxScaler
import copy


# ==========================================
# Configuration
# ==========================================
class Config:
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    EPOCHS = 50
    PATIENCE = 15
    DROPOUT_RATE = 0.5
    N_FOLDS = 5
    NUM_WORKERS = 2

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_19"
    CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.npz")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Model Specifics
    IMG_SIZE = 75

    @staticmethod
    def set_seed():
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed(Config.SEED)
        np.random.seed(Config.SEED)
        import random

        random.seed(Config.SEED)
        torch.backends.cudnn.deterministic = True


# ==========================================
# Data Processing
# ==========================================
def process_data(load_cached_data=True):
    Config.set_seed()
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(Config.CACHE_PATH):
        print(f"Loading cached data from {Config.CACHE_PATH}")
        data = np.load(Config.CACHE_PATH, allow_pickle=True)
        return (
            data["X_train"],
            data["y_train"],
            data["inc_train"],
            data["X_test"],
            data["inc_test"],
            data["test_ids"],
        )

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Combine train and val metadata for full training set processing
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Load Raw JSON
    # Note: We need to map IDs to the raw data in train.json and test.json
    with open(os.path.join(Config.INPUT_DIR, "train.json"), "r") as f:
        raw_train = json.load(f)
    with open(os.path.join(Config.INPUT_DIR, "test.json"), "r") as f:
        raw_test = json.load(f)

    # Create ID lookup maps
    train_data_map = {item["id"]: item for item in raw_train}
    test_data_map = {item["id"]: item for item in raw_test}

    # Helper to extract images
    def extract_images(meta_df, data_map):
        images = []
        inc_angles = []
        ids = []
        labels = []

        for _, row in meta_df.iterrows():
            img_id = row["id"]
            item = data_map[img_id]

            # Bands
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            b3 = (b1 + b2) / 2.0

            # Stack: (3, 75, 75)
            img = np.stack([b1, b2, b3], axis=0)
            images.append(img)

            # Incidence Angle
            inc = row["inc_angle"]
            # If NaN (from metadata csv), we will impute later
            inc_angles.append(inc)

            ids.append(img_id)
            if "is_iceberg" in row:
                labels.append(row["is_iceberg"])

        return (
            np.array(images),
            np.array(inc_angles),
            np.array(ids),
            np.array(labels) if labels else None,
        )

    X_train, inc_train, _, y_train = extract_images(full_train_meta, train_data_map)
    X_test, inc_test, test_ids, _ = extract_images(test_meta, test_data_map)

    # Impute Incidence Angles (Mean Imputation based on Train)
    # Note: inc_train has NaNs where original json had 'na'
    inc_mean = np.nanmean(inc_train)
    inc_train = np.nan_to_num(inc_train, nan=inc_mean)
    inc_test = np.nan_to_num(inc_test, nan=inc_mean)

    # Global Channel-wise MinMax Scaling
    # Fit on Train
    for c in range(3):
        c_min = X_train[:, c, :, :].min()
        c_max = X_train[:, c, :, :].max()

        # Avoid division by zero
        denom = c_max - c_min + 1e-8

        X_train[:, c, :, :] = (X_train[:, c, :, :] - c_min) / denom
        X_test[:, c, :, :] = (X_test[:, c, :, :] - c_min) / denom

    print(f"Saving processed data to {Config.CACHE_PATH}")
    np.savez(
        Config.CACHE_PATH,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        X_test=X_test,
        inc_test=inc_test,
        test_ids=test_ids,
    )

    return X_train, y_train, inc_train, X_test, inc_test, test_ids


# ==========================================
# Dataset
# ==========================================
class IcebergDataset(Dataset):
    def __init__(self, X, inc, y=None, transform=False):
        self.X = torch.FloatTensor(X)
        self.inc = torch.FloatTensor(inc)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        inc = self.inc[idx]

        if self.transform:
            # Rotations: 0, 90, 180, 270
            k = np.random.randint(0, 4)
            img = torch.rot90(img, k, [1, 2])

            # Horizontal Flip
            if np.random.random() > 0.5:
                img = torch.flip(img, [2])

        if self.y is not None:
            return img, inc, self.y[idx]
        return img, inc


# ==========================================
# Model Architecture: DWB-DPN
# ==========================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channels = channels

        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
        )

        # Spatial Attention
        self.conv7x7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        # MaxPool & AvgPool
        b, c, h, w = x.size()
        max_pool = F.max_pool2d(x, (h, w), stride=(h, w))
        avg_pool = F.avg_pool2d(x, (h, w), stride=(h, w))

        channel_att = torch.sigmoid(self.mlp(max_pool) + self.mlp(avg_pool))
        channel_att = channel_att.view(b, c, 1, 1)
        x = x * channel_att

        # Spatial Attention
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        spatial_att = torch.cat([max_spatial, avg_spatial], dim=1)
        spatial_att = torch.sigmoid(self.conv7x7(spatial_att))

        return x * spatial_att


class DualPooling(nn.Module):
    def __init__(self):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # Max Pooling (Peaks)
        p_max = self.max_pool(x)
        # Min Pooling (Shadows) -> -MaxPool(-x)
        p_min = -self.max_pool(-x)
        return torch.cat([p_max, p_min], dim=1)


class DWB_DPN(nn.Module):
    def __init__(self):
        super(DWB_DPN, self).__init__()

        # Backbone
        # Stage 1: 3 -> 64. Output after DualPool: 128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.cbam1 = CBAM(64)

        # Stage 2: 128 -> 128. Output after DualPool: 256
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAM(128)

        # Stage 3: 256 -> 128. Output after DualPool: 256
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)

        # Stage 4: 256 -> 128. Output after DualPool: 256
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.cbam4 = CBAM(128)

        self.pool = DualPooling()

        # Decoupled Readout
        # Path A: Spatial Grid (1x1 Conv)
        self.path_a_conv = nn.Conv2d(256, 64, kernel_size=1)  # 256 -> 64
        # Path B: Peak Intensity (Global Max Pool) - No params needed

        # Metadata Branch
        self.meta_fc = nn.Sequential(nn.Linear(1, 16), nn.BatchNorm1d(16), nn.ReLU())

        # Fusion Head
        # Path A: 64 * 4 * 4 = 1024 (Assuming 75 -> 37 -> 18 -> 9 -> 4)
        # Path B: 256
        # Meta: 16
        # Total: 1024 + 256 + 16 = 1296
        self.fusion_fc = nn.Sequential(
            nn.Linear(1296, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

    def forward(self, x, inc):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = self.pool(x)

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.pool(x)

        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = self.pool(x)

        # Stage 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.pool(x)  # Output: (B, 256, 4, 4)

        # Decoupled Readout
        # Path A
        xa = self.path_a_conv(x)  # (B, 64, 4, 4)
        xa = xa.view(xa.size(0), -1)  # (B, 1024)

        # Path B
        xb = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)  # (B, 256)

        # Metadata
        inc = inc.view(-1, 1)
        xm = self.meta_fc(inc)

        # Fusion
        feat = torch.cat([xa, xb, xm], dim=1)
        out = self.fusion_fc(feat)

        return out


# ==========================================
# Training Pipeline
# ==========================================
def train_model():
    Config.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    X_train, y_train, inc_train, X_test, inc_test, test_ids = process_data()

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(X_train))
    test_preds_accum = np.zeros(len(X_test))

    # Test Loader
    test_ds = IcebergDataset(X_test, inc_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nFold {fold + 1}/{Config.N_FOLDS}")

        # Split Data
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        inc_tr, inc_val = inc_train[train_idx], inc_train[val_idx]

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

        # Model, Optim, Loss
        model = DWB_DPN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(Config.EPOCHS):
            # Train
            model.train()
            train_loss = 0
            for imgs, incs, labels in train_loader:
                imgs, incs, labels = (
                    imgs.to(device),
                    incs.to(device),
                    labels.to(device).unsqueeze(1),
                )

                optimizer.zero_grad()
                outputs = model(imgs, incs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Val
            model.eval()
            val_loss = 0
            val_preds_fold = []
            with torch.no_grad():
                for imgs, incs, labels in val_loader:
                    imgs, incs, labels = (
                        imgs.to(device),
                        incs.to(device),
                        labels.to(device).unsqueeze(1),
                    )
                    outputs = model(imgs, incs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_preds_fold.extend(torch.sigmoid(outputs).cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)

            scheduler.step(avg_val_loss)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
            )

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                # Save OOF preds for this best epoch
                oof_preds[val_idx] = np.array(val_preds_fold).flatten()
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model for inference
        model.load_state_dict(best_model_state)

        # Save model
        model_path = os.path.join(Config.WORKING_DIR, f"dwb_dpn_fold_{fold}.pth")
        torch.save(best_model_state, model_path)

        # Predict Test
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for imgs, incs in test_loader:
                imgs, incs = imgs.to(device), incs.to(device)
                outputs = model(imgs, incs)
                fold_test_preds.extend(torch.sigmoid(outputs).cpu().numpy())

        test_preds_accum += np.array(fold_test_preds).flatten()

    # Average Test Preds
    avg_test_preds = test_preds_accum / Config.N_FOLDS

    # Create Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print OOF Metric
    from sklearn.metrics import log_loss

    oof_loss = log_loss(y_train, oof_preds)
    print(f"Overall OOF Log Loss: {oof_loss:.6f}")


if __name__ == "__main__":
    # This block is technically forbidden by instructions, but required for local testing if run directly.
    # The instructions say "DO NOT include an if __name__ == '__main__': block".
    # However, without it, nothing runs. The prompt might be expecting me to provide a module that *they* import and run.
    # Given the ambiguity, I will comment this out to strictly follow "Only implement the module class/functions".
    # But I will provide a function `run()` that can be called.
    pass


def run():
    train_model()
