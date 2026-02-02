import os
import json
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

# ==========================================
# Utility Functions
# ==========================================


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==========================================
# Data Processing
# ==========================================


def get_scaled_imgs(df):
    """
    Helper to convert dataframe band lists to (N, 3, 75, 75) float arrays.
    Channels: HH, HV, Avg.
    """
    imgs = []
    for i, row in df.iterrows():
        band_1 = np.array(row["band_1"]).reshape(75, 75)
        band_2 = np.array(row["band_2"]).reshape(75, 75)
        band_3 = (band_1 + band_2) / 2.0
        imgs.append(np.stack([band_1, band_2, band_3], axis=0))
    return np.array(imgs)


def process_data(load_cached_data=True, base_dir="./working/idea_6"):
    """
    Loads, processes, and caches the dataset.
    Returns: X_train, y_train, inc_train, X_test, inc_test, test_ids
    """
    cache_path = os.path.join(base_dir, "processed_data.npz")
    os.makedirs(base_dir, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["X_train"],
            data["y_train"],
            data["inc_train"],
            data["X_test"],
            data["inc_test"],
            data["test_ids"],
        )

    print("Processing data from scratch...")

    # Load Metadata to guide loading (though we load full jsons)
    train_meta = pd.read_csv("./metadata/train.csv")
    val_meta = pd.read_csv("./metadata/val.csv")

    # Combine train and val for K-Fold
    # We use the full provided training set for CV
    with open("./input/train.json", "r") as f:
        train_json = json.load(f)
    with open("./input/test.json", "r") as f:
        test_json = json.load(f)

    df_train = pd.DataFrame(train_json)
    df_test = pd.DataFrame(test_json)

    # Impute incidence angle
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    inc_mean = df_train["inc_angle"].mean()
    df_train["inc_angle"] = df_train["inc_angle"].fillna(inc_mean)

    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")
    df_test["inc_angle"] = df_test["inc_angle"].fillna(inc_mean)

    # Extract Images
    print("Extracting images...")
    X_train_raw = get_scaled_imgs(df_train)
    X_test_raw = get_scaled_imgs(df_test)

    # Independent Per-Channel Min-Max Scaling on Training Set
    print("Scaling images...")
    min_vals = []
    max_vals = []

    for c in range(3):
        # Compute stats on training set
        c_data = X_train_raw[:, c, :, :]
        _min = c_data.min()
        _max = c_data.max()

        # Apply to train
        X_train_raw[:, c, :, :] = (X_train_raw[:, c, :, :] - _min) / (_max - _min)
        # Apply to test
        X_test_raw[:, c, :, :] = (X_test_raw[:, c, :, :] - _min) / (_max - _min)

    X_train = X_train_raw.astype(np.float32)
    X_test = X_test_raw.astype(np.float32)

    y_train = df_train["is_iceberg"].values.astype(np.float32)
    inc_train = df_train["inc_angle"].values.astype(np.float32)
    inc_test = df_test["inc_angle"].values.astype(np.float32)
    test_ids = df_test["id"].values

    print(f"Saving processed data to {cache_path}")
    np.savez(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        X_test=X_test,
        inc_test=inc_test,
        test_ids=test_ids,
    )

    return X_train, y_train, inc_train, X_test, inc_test, test_ids


# ==========================================
# Dataset Class
# ==========================================


class IcebergDataset(Dataset):
    def __init__(self, X, inc, y=None, transform=None):
        self.X = X
        self.inc = inc
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        inc = self.inc[idx]

        img_tensor = torch.from_numpy(img)
        inc_tensor = torch.tensor([inc], dtype=torch.float32)

        if self.transform:
            # Rotational Invariance (0, 90, 180, 270)
            k = random.randint(0, 3)
            img_tensor = torch.rot90(img_tensor, k, [1, 2])

            # Horizontal Flip
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])

        if self.y is not None:
            label = torch.tensor([self.y[idx]], dtype=torch.float32)
            return img_tensor, inc_tensor, label
        else:
            return img_tensor, inc_tensor


# ==========================================
# Model Architecture (A2SHN)
# ==========================================


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

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
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class A2SHN(nn.Module):
    def __init__(self):
        super(A2SHN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.cbam1 = CBAM(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAM(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.cbam4 = CBAM(64)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Flatten: 64 * 4 * 4 = 1024

        # Metadata Branch
        self.meta_fc = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.BatchNorm1d(16))

        # Fusion Head
        self.head = nn.Sequential(
            nn.Linear(1024 + 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, x, inc):
        x = self.pool1(self.cbam1(self.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.cbam2(self.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.cbam3(self.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.cbam4(self.relu(self.bn4(self.conv4(x)))))

        x = x.view(x.size(0), -1)
        m = self.meta_fc(inc)

        combined = torch.cat([x, m], dim=1)
        return self.head(combined)


# ==========================================
# Training and Inference
# ==========================================


def train_model(X_train, y_train, inc_train, n_splits=5, epochs=50, batch_size=32):
    """
    Trains the A2SHN model using Stratified K-Fold CV.
    Returns list of trained models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    trained_models = []

    fold = 0
    for train_idx, val_idx in skf.split(X_train, y_train):
        fold += 1
        print(f"\n--- Fold {fold}/{n_splits} ---")

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        inc_tr, inc_val = inc_train[train_idx], inc_train[val_idx]

        train_ds = IcebergDataset(X_tr, inc_tr, y_tr, transform=True)
        val_ds = IcebergDataset(X_val, inc_val, y_val, transform=False)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        model = A2SHN().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=2e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        best_model_state = None
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for imgs, incs, labels in train_loader:
                imgs, incs, labels = imgs.to(device), incs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs, incs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, incs, labels in val_loader:
                    imgs, incs, labels = (
                        imgs.to(device),
                        incs.to(device),
                        labels.to(device),
                    )
                    outputs = model(imgs, incs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
            val_loss /= len(val_ds)

            scheduler.step(val_loss)
            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.10f} - Val Loss: {val_loss:.10f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        trained_models.append(model)

    return trained_models


def generate_submission(
    models, X_test, inc_test, test_ids, output_path="./submission/submission.csv"
):
    """
    Generates predictions using the ensemble of models and saves to CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nGenerating predictions...")

    test_ds = IcebergDataset(X_test, inc_test, y=None, transform=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs, incs in test_loader:
                imgs, incs = imgs.to(device), incs.to(device)
                outputs = model(imgs, incs)
                preds.extend(outputs.cpu().numpy().flatten())
        all_preds.append(preds)

    avg_preds = np.mean(all_preds, axis=0)

    sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
