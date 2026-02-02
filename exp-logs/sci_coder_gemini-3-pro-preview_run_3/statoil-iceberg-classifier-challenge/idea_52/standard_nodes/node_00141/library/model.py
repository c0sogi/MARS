import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import AverageMeter, seed_everything, save_checkpoint

# -----------------------------------------------------------------------------
# Model Components
# -----------------------------------------------------------------------------


class HybridSEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block using Global Average Pooling.
    """

    def __init__(self, channels, reduction=16):
        super(HybridSEBlock, self).__init__()
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


class AngleEmbedder(nn.Module):
    """
    Projects scalar incidence angle into a latent embedding.
    """

    def __init__(self, output_dim=16):
        super(AngleEmbedder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, output_dim), nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        # x shape: (batch_size, 1)
        return self.net(x)


class MS_IDPH_CNN(nn.Module):
    """
    Multi-Spectral Isomorphic CNN with Angle Embedding.
    """

    def __init__(self):
        super(MS_IDPH_CNN, self).__init__()

        # --- Backbone (4-Stage Plain CNN) ---
        # We explicitly retain bias terms.
        # Structure: Conv -> BN -> LeakyReLU -> SE -> MaxPool

        # Stage 1: 4 input channels -> 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            HybridSEBlock(64),
            nn.MaxPool2d(2, 2),
        )

        # Stage 2: 64 -> 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            HybridSEBlock(128),
            nn.MaxPool2d(2, 2),
        )

        # Stage 3: 128 -> 128
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            HybridSEBlock(128),
            nn.MaxPool2d(2, 2),
        )

        # Stage 4: 128 -> 128
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            HybridSEBlock(128),
            nn.MaxPool2d(2, 2),
        )

        # --- Readout (Isomorphic Dual-Polarity) ---
        # Projections to compress channels before pooling
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1, bias=True)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1, bias=True)

        # --- Angle Embedding ---
        self.angle_embed = AngleEmbedder(output_dim=16)

        # --- Classifier ---
        # Input: (64_max + 64_min) * 2 stages + 16 angle = 256 + 16 = 272
        self.classifier = nn.Sequential(
            nn.Linear(272, 256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # x: (B, 2, 75, 75) - HH, HV
        # angle: (B, 1)

        # --- Data Representation (4 Channels) ---
        # 1. HH
        hh = x[:, 0:1, :, :]
        # 2. HV
        hv = x[:, 1:2, :, :]
        # 3. Synthetic Average
        avg = (hh + hv) / 2.0
        # 4. Depolarization Ratio (HH - HV)
        ratio = hh - hv

        x_in = torch.cat([hh, hv, avg, ratio], dim=1)  # (B, 4, 75, 75)

        # --- Backbone ---
        x1 = self.layer1(x_in)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)  # Stage 3 features
        x4 = self.layer4(x3)  # Stage 4 features

        # --- Readout ---
        # Stage 3
        p3 = self.proj3(x3)
        p3_max = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        p3_min = F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)  # Min pooling
        feat3 = torch.cat([p3_max, p3_min], dim=1)  # 128 dim

        # Stage 4
        p4 = self.proj4(x4)
        p4_max = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)
        p4_min = F.adaptive_max_pool2d(-p4, 1).view(p4.size(0), -1)
        feat4 = torch.cat([p4_max, p4_min], dim=1)  # 128 dim

        img_feat = torch.cat([feat3, feat4], dim=1)  # 256 dim

        # --- Angle Fusion ---
        ang_emb = self.angle_embed(angle)  # 16 dim

        combined = torch.cat([img_feat, ang_emb], dim=1)  # 272 dim

        # --- Classification ---
        out = self.classifier(combined)
        return out


# -----------------------------------------------------------------------------
# Data Processing & Caching
# -----------------------------------------------------------------------------


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays, and caches it.
    Implements median imputation for incidence angles.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "angle_train": os.path.join(Config.CACHE_DIR, "angle_train.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "angle_test": os.path.join(Config.CACHE_DIR, "angle_test.npy"),
        "ids_test": os.path.join(Config.CACHE_DIR, "ids_test.npy"),
    }

    # Check if cache exists
    if load_cached_data and all(os.path.exists(f) for f in files.values()):
        print("Loading cached data...")
        data = {k: np.load(v, allow_pickle=True) for k, v in files.items()}
        return data

    print("Processing raw data from scratch...")

    # --- Process Train ---
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)

    df_train = pd.DataFrame(train_data)

    # Images
    # Reshape flattened 5625 -> 75x75
    b1_train = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_1"]
        ]
    )
    b2_train = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_2"]
        ]
    )
    X_train = np.stack([b1_train, b2_train], axis=1)  # (N, 2, 75, 75)

    # Targets
    y_train = df_train["is_iceberg"].values.astype(np.float32)

    # Angles
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    angle_train = df_train["inc_angle"].values.astype(np.float32)

    # --- Process Test ---
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    df_test = pd.DataFrame(test_data)

    b1_test = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_test["band_1"]
        ]
    )
    b2_test = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_test["band_2"]
        ]
    )
    X_test = np.stack([b1_test, b2_test], axis=1)

    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")
    angle_test = df_test["inc_angle"].values.astype(np.float32)
    ids_test = df_test["id"].values

    # --- Imputation ---
    # Combine train and test angles to compute global median for robustness, or just train.
    # We'll use train median to avoid leakage, though test distribution should be similar.
    # Actually, let's use all available valid angles for the most robust median.
    all_angles = np.concatenate([angle_train, angle_test])
    median_angle = np.nanmedian(all_angles)

    angle_train[np.isnan(angle_train)] = median_angle
    angle_test[np.isnan(angle_test)] = median_angle

    # --- Save to Cache ---
    np.save(files["X_train"], X_train)
    np.save(files["y_train"], y_train)
    np.save(files["angle_train"], angle_train)
    np.save(files["X_test"], X_test)
    np.save(files["angle_test"], angle_test)
    np.save(files["ids_test"], ids_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "angle_train": angle_train,
        "X_test": X_test,
        "angle_test": angle_test,
        "ids_test": ids_test,
    }


class IcebergDataset(Dataset):
    def __init__(self, X, y, angle, transform=None):
        self.X = X
        self.y = y
        self.angle = angle
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        ang = self.angle[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        ang_tensor = torch.tensor([ang], dtype=torch.float32)

        if self.transform:
            # Apply random flips.
            # torchvision transforms usually expect PIL or (C,H,W) tensor.
            # We implement simple flip manually or use torchvision functional
            if np.random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])  # Horizontal
            if np.random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [1])  # Vertical

        if self.y is not None:
            label = torch.tensor([self.y[idx]], dtype=torch.float32)
            return img_tensor, ang_tensor, label
        else:
            return img_tensor, ang_tensor


# -----------------------------------------------------------------------------
# Training Pipeline
# -----------------------------------------------------------------------------


def train_one_fold(fold_idx, train_idx, val_idx, data, device):
    print(f"\n--- Starting Fold {fold_idx} ---")

    X, y, angle = data["X_train"], data["y_train"], data["angle_train"]

    # Subset data
    X_tr, y_tr, ang_tr = X[train_idx], y[train_idx], angle[train_idx]
    X_val, y_val, ang_val = X[val_idx], y[val_idx], angle[val_idx]

    # Debug mode
    if Config.DEBUG:
        X_tr, y_tr, ang_tr = (
            X_tr[: Config.DEBUG_SUBSET_SIZE],
            y_tr[: Config.DEBUG_SUBSET_SIZE],
            ang_tr[: Config.DEBUG_SUBSET_SIZE],
        )
        X_val, y_val, ang_val = (
            X_val[: Config.DEBUG_SUBSET_SIZE],
            y_val[: Config.DEBUG_SUBSET_SIZE],
            ang_val[: Config.DEBUG_SUBSET_SIZE],
        )

    # Datasets
    train_dataset = IcebergDataset(X_tr, y_tr, ang_tr, transform=True)
    val_dataset = IcebergDataset(X_val, y_val, ang_val, transform=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model, Optimizer, Criterion
    model = MS_IDPH_CNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        model.train()
        train_loss = AverageMeter()

        for imgs, angs, labels in train_loader:
            imgs, angs, labels = imgs.to(device), angs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, angs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), imgs.size(0))

        # Validate
        model.eval()
        val_loss = AverageMeter()

        with torch.no_grad():
            for imgs, angs, labels in val_loader:
                imgs, angs, labels = imgs.to(device), angs.to(device), labels.to(device)
                outputs = model(imgs, angs)
                loss = criterion(outputs, labels)
                val_loss.update(loss.item(), imgs.size(0))

        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss.avg:.6f} | Val Loss: {val_loss.avg:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_loss.avg < best_loss:
            best_loss = val_loss.avg
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss.avg,
                },
                is_best=True,
                checkpoint_dir=Config.CHECKPOINT_DIR,
                filename=f"checkpoint_fold_{fold_idx}.pth",
            )
            # Rename best model specifically for easy retrieval
            best_path = os.path.join(
                Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
            )
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return best_loss


def predict_test(data, device):
    """
    Generates predictions using the ensemble of 5 trained models.
    """
    print("\n--- Generating Submission ---")

    X_test, angle_test = data["X_test"], data["angle_test"]
    ids_test = data["ids_test"]

    test_dataset = IcebergDataset(X_test, None, angle_test, transform=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Array to store sum of probabilities from all folds
    ensemble_probs = np.zeros((len(X_test), 1))

    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found. Skipping.")
            continue

        print(f"Loading model for Fold {fold}...")
        model = MS_IDPH_CNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for imgs, angs in test_loader:
                imgs, angs = imgs.to(device), angs.to(device)
                outputs = model(imgs, angs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_probs.append(probs)

        ensemble_probs += np.vstack(fold_probs)

    # Average probabilities
    avg_probs = ensemble_probs / Config.NUM_FOLDS

    # Create submission dataframe
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_probs.flatten()})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline():
    """
    Main driver function.
    """
    seed_everything(Config.SEED)
    Config.setup()

    # 1. Load Data
    data = process_and_cache_data(load_cached_data=True)

    # 2. 5-Fold Cross Validation
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    X = data["X_train"]
    y = data["y_train"]

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_one_fold(fold_idx, train_idx, val_idx, data, Config.DEVICE)

    # 3. Predict and Submit
    predict_test(data, Config.DEVICE)
