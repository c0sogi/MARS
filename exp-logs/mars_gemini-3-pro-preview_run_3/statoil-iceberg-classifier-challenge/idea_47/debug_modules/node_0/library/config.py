import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import log_loss

# =========================================================================================
# CONFIGURATION
# =========================================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 75
PATIENCE = 12
N_FOLDS = 5
SEED = 42

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_47"
SUBMISSION_DIR = "./submission"
CHECKPOINT_DIR = "./working/idea_47/checkpoints"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================================================================================
# DATA PROCESSING & CACHING
# =========================================================================================
def process_data(load_cached_data=True):
    """
    Loads raw data, processes images to (N, 3, 75, 75), handles missing angles,
    and caches the result as .npy files.
    """
    # File paths for cache
    cache_X_train = os.path.join(CACHE_DIR, "X_train.npy")
    cache_y_train = os.path.join(CACHE_DIR, "y_train.npy")
    cache_angle_train = os.path.join(CACHE_DIR, "angle_train.npy")
    cache_ids_train = os.path.join(CACHE_DIR, "ids_train.npy")

    cache_X_test = os.path.join(CACHE_DIR, "X_test.npy")
    cache_angle_test = os.path.join(CACHE_DIR, "angle_test.npy")
    cache_ids_test = os.path.join(CACHE_DIR, "ids_test.npy")

    # Check if cache exists
    if load_cached_data:
        if (
            os.path.exists(cache_X_train)
            and os.path.exists(cache_y_train)
            and os.path.exists(cache_X_test)
        ):
            print("Loading data from cache...")
            X_train = np.load(cache_X_train)
            y_train = np.load(cache_y_train)
            angle_train = np.load(cache_angle_train)
            ids_train = np.load(cache_ids_train, allow_pickle=True)

            X_test = np.load(cache_X_test)
            angle_test = np.load(cache_angle_test)
            ids_test = np.load(cache_ids_test, allow_pickle=True)
            return (
                X_train,
                y_train,
                angle_train,
                ids_train,
                X_test,
                angle_test,
                ids_test,
            )

    print("Processing data from scratch...")

    # Load Metadata
    df_train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Combine train and val metadata for full training set processing
    df_train_full = pd.concat([df_train_meta, df_val_meta], axis=0).reset_index(
        drop=True
    )

    # Load Raw JSON
    # Note: We load the full files and then map using metadata
    print("Loading raw json files...")
    with open(os.path.join(INPUT_DIR, "train.json"), "r") as f:
        raw_train = json.load(f)
    with open(os.path.join(INPUT_DIR, "test.json"), "r") as f:
        raw_test = json.load(f)

    # Create dictionaries for fast lookup by id
    train_dict = {item["id"]: item for item in raw_train}
    test_dict = {item["id"]: item for item in raw_test}

    # Helper to process a dataframe
    def process_df(df, data_dict, is_train=True):
        ids = []
        images = []
        angles = []
        targets = []

        for _, row in df.iterrows():
            img_id = row["id"]
            item = data_dict[img_id]

            # Process Bands
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            avg = (b1 + b2) / 2.0

            # Stack to (75, 75, 3) -> Transpose later to (3, 75, 75) for PyTorch
            img = np.dstack((b1, b2, avg))
            images.append(img)
            ids.append(img_id)

            # Angle
            # Use the value from metadata which already handled 'na' as NaN
            angles.append(row["inc_angle"])

            if is_train:
                targets.append(row["is_iceberg"])

        images = np.array(images, dtype=np.float32)
        # Transpose to (N, C, H, W)
        images = images.transpose(0, 3, 1, 2)
        angles = np.array(angles, dtype=np.float32)
        ids = np.array(ids)

        if is_train:
            targets = np.array(targets, dtype=np.float32)
            return images, targets, angles, ids
        else:
            return images, angles, ids

    # Process Train (Combined Train+Val)
    X_train, y_train, angle_train, ids_train = process_df(
        df_train_full, train_dict, is_train=True
    )

    # Process Test
    X_test, angle_test, ids_test = process_df(df_test_meta, test_dict, is_train=False)

    # Impute Missing Angles with Median of Training Set
    # Note: Metadata generation coerced 'na' to NaN.
    angle_median = np.nanmedian(angle_train)

    angle_train[np.isnan(angle_train)] = angle_median
    angle_test[np.isnan(angle_test)] = angle_median

    print(f"Data Processed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Imputed missing angles with median: {angle_median}")

    # Save to Cache
    np.save(cache_X_train, X_train)
    np.save(cache_y_train, y_train)
    np.save(cache_angle_train, angle_train)
    np.save(cache_ids_train, ids_train)

    np.save(cache_X_test, X_test)
    np.save(cache_angle_test, angle_test)
    np.save(cache_ids_test, ids_test)

    return X_train, y_train, angle_train, ids_train, X_test, angle_test, ids_test


# =========================================================================================
# DATASET
# =========================================================================================
class IcebergDataset(Dataset):
    def __init__(self, X, y, angles, transform=None):
        self.X = X
        self.y = y
        self.angles = angles
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.X[idx])
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle, label
        else:
            return img, angle


# =========================================================================================
# MODEL: IDPH-CNN
# =========================================================================================
class HybridSE(nn.Module):
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


class IsomorphicReadout(nn.Module):
    def __init__(self, in_channels, out_channels_per_pool=64):
        super(IsomorphicReadout, self).__init__()
        # Project to half the desired output size because we concat Max and Min
        self.project = nn.Conv2d(
            in_channels, out_channels_per_pool, kernel_size=1, bias=True
        )

    def forward(self, x):
        # x: (B, C_in, H, W)
        x_proj = self.project(x)  # (B, 64, H, W)

        # Global Max Pooling (Peaks)
        max_pool = F.adaptive_max_pool2d(x_proj, (1, 1)).view(x.size(0), -1)

        # Global Min Pooling (Shadows) -> Max(-x)
        min_pool = F.adaptive_max_pool2d(-x_proj, (1, 1)).view(x.size(0), -1)
        # The result of min_pool is positive magnitude of the minimum.
        # If we want the actual minimum value, we negate it back.
        # However, for features, the magnitude of the "hole" is usually what matters.
        # But to be strictly "min pooling", we should return the value.
        # Let's return the negated value to preserve the original sign domain.
        min_pool = -min_pool

        # Concatenate: 64 + 64 = 128
        return torch.cat([max_pool, min_pool], dim=1)


class IDPH_CNN(nn.Module):
    def __init__(self):
        super(IDPH_CNN, self).__init__()

        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(64),
            nn.MaxPool2d(2, 2),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(128),
            nn.MaxPool2d(2, 2),
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(128),
            nn.MaxPool2d(2, 2),
        )
        self.readout3 = IsomorphicReadout(128, 64)  # Output 128

        # Stage 4
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            HybridSE(128),
            nn.MaxPool2d(2, 2),
        )
        self.readout4 = IsomorphicReadout(128, 64)  # Output 128

        # Classifier
        # Input: 128 (Stage3) + 128 (Stage4) + 1 (Angle) = 257
        self.classifier = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        # Initialization
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

    def forward(self, x, angle):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        feat3 = self.readout3(x3)
        feat4 = self.readout4(x4)

        # Concatenate features and angle
        angle = angle.view(-1, 1)
        features = torch.cat([feat3, feat4, angle], dim=1)

        out = self.classifier(features)
        return out


# =========================================================================================
# TRAINING & EVALUATION
# =========================================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_fold(fold_idx, train_idx, val_idx, X, y, angles, device):
    print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

    # Split Data
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    a_tr, a_val = angles[train_idx], angles[val_idx]

    # Transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Datasets
    train_ds = IcebergDataset(X_tr, y_tr, a_tr, transform=train_transform)
    val_ds = IcebergDataset(X_val, y_val, a_val, transform=None)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model Setup
    model = IDPH_CNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, angs, lbls in train_loader:
            imgs, angs, lbls = imgs.to(device), angs.to(device), lbls.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, angs).squeeze(1)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, angs, lbls in val_loader:
                imgs, angs, lbls = imgs.to(device), angs.to(device), lbls.to(device)
                outputs = model(imgs, angs).squeeze(1)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_ds)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_loss


def generate_submission(X_test, angle_test, ids_test, device):
    print("\nGenerating Submission...")
    test_ds = IcebergDataset(X_test, None, angle_test, transform=None)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    fold_preds = []

    for fold_idx in range(N_FOLDS):
        model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold_idx} not found.")
            continue

        model = IDPH_CNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            for imgs, angs in test_loader:
                imgs, angs = imgs.to(device), angs.to(device)
                outputs = model(imgs, angs).squeeze(1)
                probs = torch.sigmoid(outputs)
                preds.extend(probs.cpu().numpy())

        fold_preds.append(np.array(preds))

    if not fold_preds:
        raise RuntimeError("No models found for prediction.")

    # Average predictions
    avg_preds = np.mean(fold_preds, axis=0)

    # Save
    sub_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
    out_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")


# =========================================================================================
# MAIN PIPELINE
# =========================================================================================
def run_pipeline():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    X, y, angles, ids, X_test, angle_test, ids_test = process_data(
        load_cached_data=True
    )

    # 2. Stratified K-Fold
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # 3. Train Loop
    cv_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        score = train_one_fold(fold_idx, train_idx, val_idx, X, y, angles, device)
        cv_scores.append(score)

    print(f"\nCV Log Loss: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")

    # 4. Predict & Submit
    generate_submission(X_test, angle_test, ids_test, device)
