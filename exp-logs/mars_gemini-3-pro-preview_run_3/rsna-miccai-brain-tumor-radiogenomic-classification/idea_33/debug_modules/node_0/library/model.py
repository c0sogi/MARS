import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import pydicom
import cv2
from sklearn.metrics import roc_auc_score
from library.utils import seed_everything, get_device

# Configuration
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_33"
SUBMISSION_DIR = "./submission"
IMG_SIZE = 256
NUM_SLICES = 16
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
TOTAL_CHANNELS = NUM_SLICES * len(MODALITIES)  # 64


class VAMSNet(nn.Module):
    def __init__(self, drop_path_rate=0.2):
        super().__init__()
        # EfficientNet-B0 backbone
        # in_chans=64 to accept 4 modalities * 16 slices
        # num_classes=1 for binary classification (outputs logits)
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=TOTAL_CHANNELS,
            drop_path_rate=drop_path_rate,
            num_classes=1,
        )

    def forward(self, x):
        # Input shape: (B, 64, 256, 256)
        return self.backbone(x)


def get_image_plane(path):
    """Parses the integer slice number from the filename."""
    basename = os.path.basename(path)
    nums = re.findall(r"\d+", basename)
    if nums:
        return int(nums[-1])
    return -1


def load_dicom_volume(paths, num_slices=16, img_size=256):
    """
    Loads, sorts, subsamples, resizes, and normalizes a volume for a single modality.
    Returns: numpy array of shape (num_slices, img_size, img_size)
    """
    if not paths or len(paths) == 0:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    # 1. External Integer Sorting
    sorted_paths = []
    for p in paths:
        idx = get_image_plane(p)
        sorted_paths.append((idx, p))
    sorted_paths.sort(key=lambda x: x[0])

    full_paths = [os.path.join(INPUT_DIR, x[1]) for x in sorted_paths]
    total_files = len(full_paths)

    # 2. Uniform Sampling (10%-90%)
    if total_files < num_slices:
        # If fewer slices than desired, sample with replacement or linspace over available
        indices = np.linspace(0, total_files - 1, num_slices).astype(int)
    else:
        # Exclude top and bottom 10%
        start_idx = int(total_files * 0.1)
        end_idx = int(total_files * 0.9)
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = total_files
        indices = np.linspace(start_idx, end_idx - 1, num_slices).astype(int)

    selected_paths = [full_paths[i] for i in indices]

    volume = []
    for p in selected_paths:
        try:
            dcm = pydicom.dcmread(p)
            img = dcm.pixel_array.astype(np.float32)

            # Resize
            if img.shape != (img_size, img_size):
                img = cv2.resize(
                    img, (img_size, img_size), interpolation=cv2.INTER_AREA
                )

            volume.append(img)
        except Exception:
            # Robust fallback for corrupt files
            volume.append(np.zeros((img_size, img_size), dtype=np.float32))

    volume = np.array(volume)  # Shape: (16, 256, 256)

    # 3. View-Adaptive Per-Modality Normalization
    # Normalize based on the min/max of the SELECTED slices only
    min_val = np.min(volume)
    max_val = np.max(volume)

    if max_val - min_val > 0:
        volume = (volume - min_val) / (max_val - min_val)
    else:
        volume = np.zeros_like(volume)

    return volume


def process_dataset(df, split_name, load_cached_data=True, debug_limit=None):
    """
    Processes the dataframe into X (images) and y (labels).
    Uses caching to avoid re-processing.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_X = os.path.join(CACHE_DIR, f"cached_{split_name}_X.npy")
    cache_y = os.path.join(CACHE_DIR, f"cached_{split_name}_y.npy")
    cache_ids = os.path.join(CACHE_DIR, f"cached_{split_name}_ids.npy")

    if load_cached_data and os.path.exists(cache_X) and os.path.exists(cache_ids):
        print(f"Loading cached {split_name} data...")
        X = np.load(cache_X)
        ids = np.load(cache_ids, allow_pickle=True)
        y = np.load(cache_y) if os.path.exists(cache_y) else None

        # Apply debug limit after loading if requested
        if debug_limit and len(X) > debug_limit:
            return (
                X[:debug_limit],
                y[:debug_limit] if y is not None else None,
                ids[:debug_limit],
            )
        return X, y, ids

    print(f"Processing {split_name} data from scratch...")
    X_list = []
    y_list = []
    ids_list = []

    # Map modality names to dataframe columns
    mod_cols = {
        "FLAIR": "flair_paths",
        "T1w": "t1w_paths",
        "T1wCE": "t1wce_paths",
        "T2w": "t2w_paths",
    }

    count = 0
    for idx, row in df.iterrows():
        if debug_limit and count >= debug_limit:
            break

        patient_id = row["BraTS21ID"]

        # 4. Modality-Grouped Stacking
        # Order: FLAIR, T1w, T1wCE, T2w
        patient_volumes = []
        for mod in MODALITIES:
            paths = row.get(mod_cols[mod], [])
            if paths is None:
                paths = []
            paths = list(paths)

            vol = load_dicom_volume(paths, num_slices=NUM_SLICES, img_size=IMG_SIZE)
            patient_volumes.append(vol)

        # Concatenate along channel dimension (axis 0)
        # Result: (64, 256, 256)
        full_volume = np.concatenate(patient_volumes, axis=0)

        X_list.append(full_volume)
        ids_list.append(patient_id)

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

        count += 1

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None

    # Save to cache
    np.save(cache_X, X)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, y, ids


class BrainTumorDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        tensor_img = torch.tensor(img, dtype=torch.float32)

        if self.y is not None:
            label = self.y[idx]
            return tensor_img, torch.tensor(label, dtype=torch.float32)
        return tensor_img


def train_model(
    epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    debug_limit=None,
    load_cached_data=True,
):
    seed_everything(42)
    device = get_device()

    # Load Metadata
    train_df = pd.read_parquet("./metadata/train.parquet")
    val_df = pd.read_parquet("./metadata/val.parquet")

    # Process Data
    X_train, y_train, _ = process_dataset(
        train_df, "train", load_cached_data, debug_limit
    )
    X_val, y_val, _ = process_dataset(val_df, "val", load_cached_data, debug_limit)

    train_dataset = BrainTumorDataset(X_train, y_train)
    val_dataset = BrainTumorDataset(X_val, y_val)

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

    model = VAMSNet(drop_path_rate=0.2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                val_preds.extend(probs.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())

        val_loss /= len(val_dataset)
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_auc = 0.5

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Best Validation AUC: {best_auc}")
    return best_model_path


def generate_submission(model_path, batch_size=32, load_cached_data=True):
    seed_everything(42)
    device = get_device()

    test_df = pd.read_parquet("./metadata/test.parquet")
    X_test, _, ids_test = process_dataset(test_df, "test", load_cached_data)

    test_dataset = BrainTumorDataset(X_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = VAMSNet(drop_path_rate=0.0).to(device)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy().flatten())

    submission_df = pd.DataFrame({"BraTS21ID": ids_test, "MGMT_value": predictions})

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run():
    # Execute the pipeline
    best_model = train_model(epochs=10, batch_size=32, learning_rate=1e-4)
    generate_submission(best_model, batch_size=32)


# Execute
run()
