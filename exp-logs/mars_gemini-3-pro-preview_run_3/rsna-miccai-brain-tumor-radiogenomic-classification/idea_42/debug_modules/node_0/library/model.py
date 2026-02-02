import os
import sys
import glob
import re
import random
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm
from library.utils import seed_everything, get_device

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    "seed": 42,
    "img_size": 224,
    "num_slices": 32,  # Total slices to extract per patient
    "batch_size": 16,
    "epochs": 15,
    "lr": 1e-4,
    "backbone": "efficientnet_b0",
    "drop_path_rate": 0.2,
    "patience": 4,  # Early stopping
    "num_workers": 4,
    "cache_dir": "./working/idea_snr_sf",
    "metadata_dir": "./metadata",
    "input_dir": "./input",
}

# ==========================================
# Data Processing & Caching
# ==========================================


def load_dicom_slice(path, img_size=224):
    """Reads a DICOM file, normalizes, and resizes it."""
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(float)

        # Resize if necessary
        if img.shape != (img_size, img_size):
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        return img
    except Exception as e:
        # Return zero placeholder if read fails
        return np.zeros((img_size, img_size), dtype=float)


def get_sorted_file_paths(paths):
    """Sorts file paths based on the integer index in the filename."""

    def extract_id(path):
        # Extract number from 'Image-123.dcm'
        match = re.search(r"Image-(\d+)\.dcm", path)
        return int(match.group(1)) if match else -1

    # Filter out paths that don't match the pattern
    valid_paths = [p for p in paths if extract_id(p) != -1]
    return sorted(valid_paths, key=extract_id)


def process_patient_volume(row, input_dir, img_size=224, num_slices=32):
    """
    Extracts high-density volume, normalizes, and splits into Even/Odd streams.
    Returns: (C, H, W) for even and odd streams. C = 64 (16 slices * 4 modalities).
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    volume_slices = {mod: [] for mod in modalities}

    # 1. Load and Select Slices
    for mod in modalities:
        paths = row[f"{mod}_paths"]
        full_paths = [os.path.join(input_dir, p) for p in paths]
        sorted_paths = get_sorted_file_paths(full_paths)

        num_files = len(sorted_paths)
        if num_files == 0:
            # Handle missing modality by creating zero volume
            selected_imgs = [np.zeros((img_size, img_size)) for _ in range(num_slices)]
        else:
            # High-Density Uniform Sampling (10% - 90%)
            start_idx = int(num_files * 0.1)
            end_idx = int(num_files * 0.9)

            # Ensure valid range
            if end_idx <= start_idx:
                start_idx, end_idx = 0, num_files

            # Generate indices
            if end_idx - start_idx < num_slices:
                # If not enough slices, take all available and repeat/pad
                indices = np.linspace(0, num_files - 1, num_slices, dtype=int)
                # Use original range 0 to N if the cropped range is too small
                sample_paths = [sorted_paths[i] for i in indices]
            else:
                indices = np.linspace(start_idx, end_idx - 1, num_slices, dtype=int)
                sample_paths = [sorted_paths[i] for i in indices]

            # Load images
            imgs = [load_dicom_slice(p, img_size) for p in sample_paths]

            # Subset-Adaptive Normalization
            # Normalize based on the min/max of the selected subset to preserve local contrast
            stack = np.array(imgs)
            min_val, max_val = stack.min(), stack.max()
            if max_val - min_val > 0:
                stack = (stack - min_val) / (max_val - min_val)
            else:
                stack = np.zeros_like(stack)

            # Store processed slices
            # stack shape: (32, 224, 224)
            selected_imgs = [stack[i] for i in range(num_slices)]

        volume_slices[mod] = selected_imgs

    # 2. Stack and Split into Even/Odd Streams
    # Structure: Modality-Grouped Stacking
    # Stream A (Even): [FLAIR_even, T1w_even, ...]
    # Stream B (Odd):  [FLAIR_odd, T1w_odd, ...]

    even_channels = []
    odd_channels = []

    for mod in modalities:
        slices = volume_slices[mod]  # List of 32 arrays

        # Deterministic Strided Splitting
        # Even indices: 0, 2, ..., 30
        even_subset = slices[0::2]
        # Odd indices: 1, 3, ..., 31
        odd_subset = slices[1::2]

        even_channels.extend(even_subset)
        odd_channels.extend(odd_subset)

    # Stack to tensor shape (C, H, W)
    # Total channels per stream = 4 modalities * 16 slices = 64
    x_even = np.array(even_channels, dtype=np.float32)
    x_odd = np.array(odd_channels, dtype=np.float32)

    return x_even, x_odd


def get_dataset_arrays(metadata_path, cache_name, load_cached_data=True):
    """
    Loads metadata, checks cache, and returns X_even, X_odd, y, and ids.
    """
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)

    path_x_even = os.path.join(CONFIG["cache_dir"], f"X_{cache_name}_even.npy")
    path_x_odd = os.path.join(CONFIG["cache_dir"], f"X_{cache_name}_odd.npy")
    path_y = os.path.join(CONFIG["cache_dir"], f"y_{cache_name}.npy")
    path_ids = os.path.join(CONFIG["cache_dir"], f"ids_{cache_name}.npy")

    # Check cache
    if load_cached_data and os.path.exists(path_x_even) and os.path.exists(path_x_odd):
        print(f"Loading cached data for {cache_name}...")
        X_even = np.load(path_x_even)
        X_odd = np.load(path_x_odd)
        ids = np.load(path_ids, allow_pickle=True)
        if os.path.exists(path_y):
            y = np.load(path_y)
        else:
            y = None
        return X_even, X_odd, y, ids

    # Process from scratch
    print(f"Processing data for {cache_name}...")
    df = pd.read_parquet(metadata_path)

    X_even_list = []
    X_odd_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        xe, xo = process_patient_volume(
            row, CONFIG["input_dir"], CONFIG["img_size"], CONFIG["num_slices"]
        )
        X_even_list.append(xe)
        X_odd_list.append(xo)
        ids_list.append(row["BraTS21ID"])
        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X_even = np.array(X_even_list)
    X_odd = np.array(X_odd_list)
    ids = np.array(ids_list)

    np.save(path_x_even, X_even)
    np.save(path_x_odd, X_odd)
    np.save(path_ids, ids)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
        np.save(path_y, y)
    else:
        y = None

    return X_even, X_odd, y, ids


class SiameseBraTSDataset(Dataset):
    def __init__(self, X_even, X_odd, y=None):
        self.X_even = X_even
        self.X_odd = X_odd
        self.y = y

    def __len__(self):
        return len(self.X_even)

    def __getitem__(self, idx):
        xe = torch.tensor(self.X_even[idx], dtype=torch.float32)
        xo = torch.tensor(self.X_odd[idx], dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
            return xe, xo, label
        else:
            return xe, xo


# ==========================================
# Model Architecture
# ==========================================


class SiameseRSFNet(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", pretrained=True):
        super().__init__()

        # Shared Backbone
        # in_chans=64 because we have 16 slices * 4 modalities per stream
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=64,
            num_classes=0,
            global_pool="",
            drop_path_rate=CONFIG["drop_path_rate"],
        )

        # Determine feature dimension
        if hasattr(self.backbone, "num_features"):
            num_features = self.backbone.num_features
        else:
            # Fallback for EfficientNet-B0 if attribute missing
            num_features = 1280

        # Fusion Head
        # Concatenates features from both streams (2 * num_features)
        # Reduces back to num_features with 1x1 Conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, 1)

    def forward(self, x_even, x_odd):
        # Pass both streams through shared backbone
        # Output shape: (B, C, H', W') -> e.g., (B, 1280, 7, 7)
        f_even = self.backbone(x_even)
        f_odd = self.backbone(x_odd)

        # Spatial Feature Fusion
        # Concatenate along channel dimension
        f_cat = torch.cat([f_even, f_odd], dim=1)  # (B, 2C, 7, 7)

        # Fuse
        f_fused = self.fusion_conv(f_cat)  # (B, C, 7, 7)

        # Classification
        pool = self.global_pool(f_fused).flatten(1)  # (B, C)
        logits = self.fc(pool)

        return logits


# ==========================================
# Training & Evaluation
# ==========================================


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for x_even, x_odd, y in loader:
        x_even, x_odd, y = x_even.to(device), x_odd.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x_even, x_odd)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_even, x_odd, y in loader:
            x_even, x_odd, y = x_even.to(device), x_odd.to(device), y.to(device)

            logits = model(x_even, x_odd)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


def predict(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x_even, x_odd in loader:
            x_even, x_odd = x_even.to(device), x_odd.to(device)
            logits = model(x_even, x_odd)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)

    return np.concatenate(all_preds)


# ==========================================
# Main Execution
# ==========================================


def main():
    seed_everything(CONFIG["seed"])
    device = get_device()

    # 1. Load Data
    train_meta = os.path.join(CONFIG["metadata_dir"], "train.parquet")
    val_meta = os.path.join(CONFIG["metadata_dir"], "val.parquet")
    test_meta = os.path.join(CONFIG["metadata_dir"], "test.parquet")

    X_train_e, X_train_o, y_train, _ = get_dataset_arrays(train_meta, "train")
    X_val_e, X_val_o, y_val, _ = get_dataset_arrays(val_meta, "val")
    X_test_e, X_test_o, _, test_ids = get_dataset_arrays(test_meta, "test")

    # 2. Datasets & Loaders
    train_dataset = SiameseBraTSDataset(X_train_e, X_train_o, y_train)
    val_dataset = SiameseBraTSDataset(X_val_e, X_val_o, y_val)
    test_dataset = SiameseBraTSDataset(X_test_e, X_test_o, None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    # 3. Model Setup
    model = SiameseRSFNet(backbone_name=CONFIG["backbone"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # 4. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(CONFIG["cache_dir"], "best_model.pth")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    predictions = predict(model, test_loader, device)

    # 6. Submission
    submission_df = pd.DataFrame(
        {"BraTS21ID": test_ids, "MGMT_value": predictions.flatten()}
    )

    # Ensure directory exists
    os.makedirs("submission", exist_ok=True)
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
