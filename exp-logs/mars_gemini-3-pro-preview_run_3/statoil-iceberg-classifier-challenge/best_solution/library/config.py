import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

# ==========================================
# Configuration Constants
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 75
NUM_FOLDS = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT_RATE = 0.5
LEAKY_RELU_SLOPE = 0.1
PATIENCE = 12
WEIGHT_DECAY = 1e-4
CACHE_DIR = "./working/idea_48/"


# ==========================================
# Utilities
# ==========================================
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# Data Processing & Caching
# ==========================================
def load_and_process_data(load_cached_data=True):
    """
    Loads data from json, processes into numpy arrays, and caches them.
    Returns: X_train, angles_train, y_train, X_test, angles_test, test_ids
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    paths = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "angles_train": os.path.join(CACHE_DIR, "angles_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "angles_test": os.path.join(CACHE_DIR, "angles_test.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
    }

    # Try loading from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in paths.values())
        if all_exist:
            print("Loading data from cache...")
            return (
                np.load(paths["X_train"]),
                np.load(paths["angles_train"]),
                np.load(paths["y_train"]),
                np.load(paths["X_test"]),
                np.load(paths["angles_test"]),
                np.load(paths["test_ids"], allow_pickle=True),
            )
        else:
            print("Cache missing. Processing from scratch...")

    # Process Training Data
    print("Loading train.json...")
    with open("./input/train.json", "r") as f:
        train_data = json.load(f)

    df_train = pd.DataFrame(train_data)

    # Process Images (Band 1, Band 2, Avg)
    # Reshape from 5625 to 75x75
    b1 = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_1"]
        ]
    )
    b2 = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_2"]
        ]
    )
    b3 = (b1 + b2) / 2.0

    # Stack to (N, 3, 75, 75)
    X_train = np.stack([b1, b2, b3], axis=1)

    # Process Angles
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    angle_median = df_train["inc_angle"].median()
    df_train["inc_angle"] = df_train["inc_angle"].fillna(angle_median)
    angles_train = df_train["inc_angle"].values.astype(np.float32)

    # Process Targets
    y_train = df_train["is_iceberg"].values.astype(np.float32)

    # Process Test Data
    print("Loading test.json...")
    with open("./input/test.json", "r") as f:
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
    b3_test = (b1_test + b2_test) / 2.0

    X_test = np.stack([b1_test, b2_test, b3_test], axis=1)

    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")
    df_test["inc_angle"] = df_test["inc_angle"].fillna(angle_median)  # Use train median
    angles_test = df_test["inc_angle"].values.astype(np.float32)
    test_ids = df_test["id"].values

    # Save to cache
    np.save(paths["X_train"], X_train)
    np.save(paths["angles_train"], angles_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_test"], X_test)
    np.save(paths["angles_test"], angles_test)
    np.save(paths["test_ids"], test_ids)

    return X_train, angles_train, y_train, X_test, angles_test, test_ids


# ==========================================
# Dataset
# ==========================================
class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, transform=None):
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # (3, 75, 75)
        angle = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.y is not None:
            label = self.y[idx]
            return img_tensor, torch.tensor(angle), torch.tensor(label)
        else:
            return img_tensor, torch.tensor(angle)


# ==========================================
# Model Architecture (IDPH-CNN)
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
        self.act = nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True)
        self.se = SEModule(out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class IDPH_CNN(nn.Module):
    def __init__(self):
        super(IDPH_CNN, self).__init__()

        # Backbone: Plain CNN 4 blocks
        # Input: 75x75
        self.block1 = ConvBlock(3, 64)  # -> 37x37
        self.block2 = ConvBlock(64, 128)  # -> 18x18
        self.block3 = ConvBlock(128, 128)  # -> 9x9 (Stage 3 Output)
        self.block4 = ConvBlock(128, 128)  # -> 4x4 (Stage 4 Output)

        # Isomorphic Projections
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1)

        # Classifier Head
        # 256 (Image features) + 1 (Angle)
        self.head = nn.Sequential(
            nn.Linear(256 + 1, 256),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 1),
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

    def forward(self, x, angle):
        # Backbone
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)  # Stage 3
        x4 = self.block4(x3)  # Stage 4

        # Isomorphic Readout Stage 3
        p3 = self.proj3(x3)  # (B, 64, H, W)
        p3_max = torch.amax(p3, dim=(2, 3))  # Global Max Pool
        p3_min = -torch.amax(-p3, dim=(2, 3))  # Global Min Pool

        # Isomorphic Readout Stage 4
        p4 = self.proj4(x4)  # (B, 64, H, W)
        p4_max = torch.amax(p4, dim=(2, 3))
        p4_min = -torch.amax(-p4, dim=(2, 3))

        # Aggregate Image Features
        img_feats = torch.cat([p3_max, p3_min, p4_max, p4_min], dim=1)  # 64*4 = 256

        # Fusion
        angle = angle.view(-1, 1)
        combined = torch.cat([img_feats, angle], dim=1)  # 257

        # Classification
        out = self.head(combined)
        return out


# ==========================================
# Training Logic
# ==========================================
def train_model(load_cached_data=True):
    set_seed(SEED)

    # Load Data
    X, angles, y, X_test, angles_test, test_ids = load_and_process_data(
        load_cached_data
    )

    # Augmentations
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # 5-Fold CV
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(y))
    test_preds_accum = np.zeros(len(test_ids))

    os.makedirs("./checkpoints", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{NUM_FOLDS}")

        # Prepare Datasets
        train_dataset = IcebergDataset(
            X[train_idx], angles[train_idx], y[train_idx], transform=train_transform
        )
        val_dataset = IcebergDataset(
            X[val_idx], angles[val_idx], y[val_idx], transform=None
        )

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Model & Optimizer
        model = IDPH_CNN().to(DEVICE)
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop with Early Stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = f"./checkpoints/model_fold_{fold}.pth"

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss_sum = 0

            for images, angs, labels in train_loader:
                images, angs, labels = (
                    images.to(DEVICE),
                    angs.to(DEVICE),
                    labels.to(DEVICE).unsqueeze(1),
                )

                optimizer.zero_grad()
                outputs = model(images, angs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * images.size(0)

            avg_train_loss = train_loss_sum / len(train_dataset)

            # Validation
            model.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for images, angs, labels in val_loader:
                    images, angs, labels = (
                        images.to(DEVICE),
                        angs.to(DEVICE),
                        labels.to(DEVICE).unsqueeze(1),
                    )
                    outputs = model(images, angs)
                    loss = criterion(outputs, labels)
                    val_loss_sum += loss.item() * images.size(0)

            avg_val_loss = val_loss_sum / len(val_dataset)

            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}"
            )

            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Load best model for OOF and Test Inference
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # OOF Predictions
        val_preds = []
        with torch.no_grad():
            for images, angs, _ in val_loader:
                images, angs = images.to(DEVICE), angs.to(DEVICE)
                out = torch.sigmoid(model(images, angs))
                val_preds.extend(out.cpu().numpy().flatten())
        oof_preds[val_idx] = val_preds

        # Test Inference (No TTA)
        test_dataset = IcebergDataset(X_test, angles_test, transform=None)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        fold_test_preds = []
        with torch.no_grad():
            for images, angs in test_loader:
                images, angs = images.to(DEVICE), angs.to(DEVICE)
                out = torch.sigmoid(model(images, angs))
                fold_test_preds.extend(out.cpu().numpy().flatten())

        test_preds_accum += np.array(fold_test_preds)

    # Final Average
    avg_test_preds = test_preds_accum / NUM_FOLDS

    # Save Submission
    os.makedirs("submission", exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})
    submission.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")


def generate_submission():
    # Wrapper to trigger training and submission generation
    train_model()
