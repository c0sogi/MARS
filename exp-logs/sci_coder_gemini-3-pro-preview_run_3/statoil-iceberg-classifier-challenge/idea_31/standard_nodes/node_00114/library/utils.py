import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss

# ------------------------------------------------------------------------------
# 1. Foundational Utilities
# ------------------------------------------------------------------------------


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Returns the available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# 2. Data Processing & Caching
# ------------------------------------------------------------------------------


def process_data(load_cached_data=True, base_dir="./working/idea_31"):
    """
    Loads raw data, processes it (images and angles), and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.
        base_dir (str): Directory to store cached .npy files.

    Returns:
        tuple: (X_train, y_train, angle_train, X_test, ids_test, angle_test)
    """
    os.makedirs(base_dir, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(base_dir, "X_train.npy"),
        "y_train": os.path.join(base_dir, "y_train.npy"),
        "angle_train": os.path.join(base_dir, "angle_train.npy"),
        "X_test": os.path.join(base_dir, "X_test.npy"),
        "ids_test": os.path.join(base_dir, "ids_test.npy"),
        "angle_test": os.path.join(base_dir, "angle_test.npy"),
    }

    # Try loading from cache
    if load_cached_data:
        all_exist = all(os.path.exists(f) for f in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            return (
                np.load(cache_files["X_train"]),
                np.load(cache_files["y_train"]),
                np.load(cache_files["angle_train"]),
                np.load(cache_files["X_test"]),
                np.load(cache_files["ids_test"], allow_pickle=True),
                np.load(cache_files["angle_test"]),
            )
        else:
            print("Cache missing or incomplete. Processing from scratch...")
    else:
        print("Forcing data processing from scratch...")

    # Load raw JSON data
    # We use metadata to locate files but here we just load the raw json directly
    # as per standard practice when we need the bands.
    print("Loading train.json...")
    with open("./input/train.json", "r") as f:
        train_data = json.load(f)

    print("Loading test.json...")
    with open("./input/test.json", "r") as f:
        test_data = json.load(f)

    # Helper to process images
    def process_images(data_list):
        imgs = []
        for item in data_list:
            band1 = np.array(item["band_1"]).reshape(75, 75)
            band2 = np.array(item["band_2"]).reshape(75, 75)
            avg = (band1 + band2) / 2.0
            # Stack to (75, 75, 3)
            img = np.dstack((band1, band2, avg))
            imgs.append(img)
        return np.array(imgs, dtype=np.float32)

    # Helper to process angles
    def process_angles(data_list, is_train=True, median_val=None):
        angles = []
        for item in data_list:
            a = item["inc_angle"]
            if a == "na":
                angles.append(np.nan)
            else:
                angles.append(float(a))
        angles = np.array(angles, dtype=np.float32)

        if is_train:
            # Calculate median from valid values
            valid_angles = angles[~np.isnan(angles)]
            median_val = np.median(valid_angles)
            # Impute
            angles[np.isnan(angles)] = median_val
            return angles, median_val
        else:
            # Impute with provided median
            angles[np.isnan(angles)] = median_val
            return angles

    print("Processing training data...")
    X_train = process_images(train_data)
    y_train = np.array([item["is_iceberg"] for item in train_data], dtype=np.float32)
    angle_train, median_angle = process_angles(train_data, is_train=True)

    print("Processing test data...")
    X_test = process_images(test_data)
    ids_test = np.array([item["id"] for item in test_data])
    angle_test = process_angles(test_data, is_train=False, median_val=median_angle)

    # Reshape images to (N, C, H, W) for PyTorch: (N, 3, 75, 75)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    print("Saving to cache...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angle_train"], angle_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["ids_test"], ids_test)
    np.save(cache_files["angle_test"], angle_test)

    return X_train, y_train, angle_train, X_test, ids_test, angle_test


# ------------------------------------------------------------------------------
# 3. Dataset Class
# ------------------------------------------------------------------------------


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, transform=None):
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is (C, H, W)
        img = self.X[idx]
        angle = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply augmentation if provided (Random flips)
        if self.transform:
            # torchvision transforms expect (C, H, W)
            img_tensor = self.transform(img_tensor)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


# ------------------------------------------------------------------------------
# 4. Model Architecture (BHA-ResNet)
# ------------------------------------------------------------------------------


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=True
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.act(out)
        return out


class BHA_ResNet(nn.Module):
    def __init__(self):
        super(BHA_ResNet, self).__init__()

        # Input: 75x75x3
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        # Stages
        # Stage 1: 64 -> 64, stride 2 (75->38)
        self.stage1 = ResBlock(64, 64, stride=2)
        # Stage 2: 64 -> 128, stride 2 (38->19)
        self.stage2 = ResBlock(64, 128, stride=2)
        # Stage 3: 128 -> 128, stride 2 (19->10)
        self.stage3 = ResBlock(128, 128, stride=2)
        # Stage 4: 128 -> 128, stride 2 (10->5)
        self.stage4 = ResBlock(128, 128, stride=2)

        # Readout: Global Max Pooling
        self.global_max = nn.AdaptiveMaxPool2d(1)

        # Head
        # Concat Stage 3 (128) + Stage 4 (128) + Angle (1) = 257
        self.fc1 = nn.Linear(128 + 128 + 1, 256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

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
        # Stem
        x = self.act(self.bn1(self.conv1(x)))

        # Stages
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        # Selective Hierarchical Max Pooling
        # Stage 3
        p3 = self.global_max(s3).view(s3.size(0), -1)  # (B, 128)
        # Stage 4
        p4 = self.global_max(s4).view(s4.size(0), -1)  # (B, 128)

        # Feature Fusion
        angle = angle.view(-1, 1)  # (B, 1)
        features = torch.cat((p3, p4, angle), dim=1)  # (B, 257)

        # Classification Head
        out = self.fc1(features)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out


# ------------------------------------------------------------------------------
# 5. Training Logic
# ------------------------------------------------------------------------------


def train_one_fold(
    fold_idx, model, train_loader, val_loader, device, epochs=75, patience=12, lr=1e-3
):
    """
    Trains the model for one fold with Early Stopping.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"--- Fold {fold_idx} Start ---")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        for images, angles, labels in train_loader:
            images, angles, labels = (
                images.to(device),
                angles.to(device),
                labels.to(device),
            )
            labels = labels.unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()
            outputs = model(images, angles)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)

        avg_train_loss = train_loss_sum / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images, angles, labels = (
                    images.to(device),
                    angles.to(device),
                    labels.to(device),
                )
                labels = labels.unsqueeze(1)

                outputs = model(images, angles)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs).cpu().numpy()
                targets = labels.cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(targets)

        avg_val_loss = val_loss_sum / len(val_loader.dataset)

        # Check Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Print metrics (Full precision)
        print(
            f"Fold {fold_idx} Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.10f} - Val Loss: {avg_val_loss:.10f}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold_idx} Best Val Loss: {best_loss:.10f}")
    return best_model_state, best_loss


# ------------------------------------------------------------------------------
# 6. Inference
# ------------------------------------------------------------------------------


def predict_test(models, test_loader, device):
    """
    Generates predictions using an ensemble of models.
    Returns average probabilities.
    """
    all_preds = []

    # Iterate over each model in the ensemble
    for model in models:
        model.eval()
        model_preds = []
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(device)
                angles = batch[1].to(device)
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy()
                model_preds.extend(probs)
        all_preds.append(np.array(model_preds))

    # Average predictions across ensemble
    avg_preds = np.mean(all_preds, axis=0)  # (N, 1)
    return avg_preds.flatten()
