import os
import sys
import glob
import random
import re
import time
import numpy as np
import pandas as pd
import cv2
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

# ==========================================
# Utility Functions
# ==========================================


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Returns the CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Data Processing & Caching
# ==========================================


def extract_slice_number(filename):
    """Extracts the integer slice number from a DICOM filename."""
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0


def load_dicom_image(path, size=(224, 224)):
    """Reads a DICOM file and resizes it. Returns a zero array on failure."""
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array
        img = img.astype(np.float32)
        if img.shape != size:
            img = cv2.resize(img, size)
        return img
    except Exception:
        return np.zeros(size, dtype=np.float32)


def process_patient(row, input_dir="./input", target_size=(224, 224)):
    """
    Processes a single patient's data according to the SNR-SF strategy:
    - Sorts and samples 32 slices (10%-90% range).
    - Normalizes based on the subset statistics.
    - Splits into Even and Odd streams.
    - Stacks modalities into 64-channel tensors.
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    modality_slices = {}

    for mod in modalities:
        paths = row[f"{mod}_paths"]
        if paths is None or len(paths) == 0:
            modality_slices[mod] = [
                np.zeros(target_size, dtype=np.float32) for _ in range(32)
            ]
            continue

        # Sort by integer slice number
        paths = sorted(paths, key=lambda x: extract_slice_number(os.path.basename(x)))

        # High-Density Uniform Sampling (10%-90% depth range)
        total_slices = len(paths)
        start_idx = int(total_slices * 0.1)
        end_idx = int(total_slices * 0.9)

        if end_idx <= start_idx:
            roi_paths = paths
        else:
            roi_paths = paths[start_idx:end_idx]

        if len(roi_paths) == 0:
            roi_paths = paths

        # Sample exactly 32 slices
        if len(roi_paths) > 0:
            indices = np.linspace(0, len(roi_paths) - 1, 32, dtype=int)
            selected_paths = [roi_paths[i] for i in indices]
        else:
            modality_slices[mod] = [
                np.zeros(target_size, dtype=np.float32) for _ in range(32)
            ]
            continue

        # Load images
        images = []
        for p in selected_paths:
            full_path = os.path.join(input_dir, p)
            img = load_dicom_image(full_path, target_size)
            images.append(img)

        images = np.array(images)  # Shape: (32, 224, 224)

        # Subset-Adaptive Per-Modality Normalization
        min_val = np.min(images)
        max_val = np.max(images)

        if max_val - min_val > 1e-6:
            images = (images - min_val) / (max_val - min_val)
        else:
            images = np.zeros_like(images)

        modality_slices[mod] = list(images)

    # Split into Even/Odd streams and Stack
    # Stream A (Even): Indices 0, 2, ..., 30 (16 slices)
    # Stream B (Odd): Indices 1, 3, ..., 31 (16 slices)
    even_stream = []
    odd_stream = []

    for mod in modalities:
        slices = modality_slices[mod]
        even_slices = [slices[i] for i in range(0, 32, 2)]
        odd_slices = [slices[i] for i in range(1, 32, 2)]
        even_stream.extend(even_slices)
        odd_stream.extend(odd_slices)

    # Final shapes: (64, 224, 224)
    X_even = np.array(even_stream, dtype=np.float32)
    X_odd = np.array(odd_stream, dtype=np.float32)

    return X_even, X_odd


def load_data_and_cache(
    metadata_path,
    cache_dir="./working/idea_40/",
    load_cached_data=True,
    input_dir="./input",
    dataset_name="train",
):
    """
    Loads data from metadata, processing it if cache is missing or load_cached_data is False.
    Caches results as .npy files in cache_dir.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_X_even = os.path.join(cache_dir, f"X_{dataset_name}_even.npy")
    cache_X_odd = os.path.join(cache_dir, f"X_{dataset_name}_odd.npy")
    cache_ids = os.path.join(cache_dir, f"ids_{dataset_name}.npy")
    cache_y = os.path.join(cache_dir, f"y_{dataset_name}.npy")

    # Try loading from cache
    if (
        load_cached_data
        and os.path.exists(cache_X_even)
        and os.path.exists(cache_X_odd)
        and os.path.exists(cache_ids)
    ):
        # print(f"Loading cached data for {dataset_name}...")
        X_even = np.load(cache_X_even)
        X_odd = np.load(cache_X_odd)
        ids = np.load(cache_ids, allow_pickle=True)
        y = np.load(cache_y) if os.path.exists(cache_y) else None
        return X_even, X_odd, y, ids

    # Process from scratch
    # print(f"Processing data for {dataset_name} from scratch...")
    df = pd.read_parquet(metadata_path)

    X_even_list = []
    X_odd_list = []
    y_list = []
    ids_list = []

    for _, row in df.iterrows():
        xe, xo = process_patient(row, input_dir=input_dir)
        X_even_list.append(xe)
        X_odd_list.append(xo)
        ids_list.append(row["BraTS21ID"])
        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X_even = np.array(X_even_list, dtype=np.float32)
    X_odd = np.array(X_odd_list, dtype=np.float32)
    ids = np.array(ids_list)

    np.save(cache_X_even, X_even)
    np.save(cache_X_odd, X_odd)
    np.save(cache_ids, ids)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
        np.save(cache_y, y)
    else:
        y = None

    return X_even, X_odd, y, ids


# ==========================================
# Dataset & Model
# ==========================================


class SiameseDataset(Dataset):
    def __init__(self, X_even, X_odd, y=None):
        self.X_even = X_even
        self.X_odd = X_odd
        self.y = y

    def __len__(self):
        return len(self.X_even)

    def __getitem__(self, idx):
        xe = torch.from_numpy(self.X_even[idx])
        xo = torch.from_numpy(self.X_odd[idx])

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return xe, xo, label
        return xe, xo


class SiameseEfficientNet(nn.Module):
    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super(SiameseEfficientNet, self).__init__()

        # Shared backbone with in_chans=64
        # timm handles weight recycling automatically
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=64,
            features_only=True,
            out_indices=(4,),  # Last feature map
        )

        feature_dim = self.backbone.feature_info[-1]["num_chs"]

        # Spatial Fusion Head
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, 1)

    def forward_one(self, x):
        feats = self.backbone(x)
        return feats[-1]

    def forward(self, x_even, x_odd):
        # Shared weights
        f_even = self.forward_one(x_even)
        f_odd = self.forward_one(x_odd)

        # Spatial Fusion
        concat = torch.cat([f_even, f_odd], dim=1)
        fused = self.fusion_conv(concat)

        # Classification
        pooled = self.global_pool(fused).flatten(1)
        logits = self.fc(pooled)
        return logits


# ==========================================
# Training & Inference
# ==========================================


def train_model(
    train_meta_path="./metadata/train.parquet",
    val_meta_path="./metadata/val.parquet",
    test_meta_path="./metadata/test.parquet",
    submission_path="./submission/submission.csv",
    epochs=15,
    batch_size=16,
    lr=1e-4,
    load_cached_data=True,
):
    device = get_device()

    # Load Data
    X_train_e, X_train_o, y_train, _ = load_data_and_cache(
        train_meta_path, dataset_name="train", load_cached_data=load_cached_data
    )
    X_val_e, X_val_o, y_val, _ = load_data_and_cache(
        val_meta_path, dataset_name="val", load_cached_data=load_cached_data
    )

    train_dataset = SiameseDataset(X_train_e, X_train_o, y_train)
    val_dataset = SiameseDataset(X_val_e, X_val_o, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = SiameseEfficientNet("efficientnet_b0", pretrained=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = "./working/idea_40/best_model.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for xe, xo, labels in train_loader:
            xe, xo, labels = xe.to(device), xo.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(xe, xo).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xe.size(0)
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_loss /= len(train_dataset)
        train_auc = roc_auc_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for xe, xo, labels in val_loader:
                xe, xo, labels = xe.to(device), xo.to(device), labels.to(device)
                outputs = model(xe, xo).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * xe.size(0)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.10f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Inference
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    X_test_e, X_test_o, _, test_ids = load_data_and_cache(
        test_meta_path, dataset_name="test", load_cached_data=load_cached_data
    )
    test_dataset = SiameseDataset(X_test_e, X_test_o, None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    predictions = []
    with torch.no_grad():
        for xe, xo in test_loader:
            xe, xo = xe.to(device), xo.to(device)
            outputs = model(xe, xo).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(probs)

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})
    submission_df.to_csv(submission_path, index=False)
