import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
import timm
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. Configuration & Constants
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_32"
SUBMISSION_PATH = "./submission/submission.csv"

# Hyperparameters
IMAGE_SIZE = 320
SLICES_PER_MODALITY = 16
TOTAL_CHANNELS = 64  # 4 modalities * 16 slices
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
SEED = 42
NUM_EPOCHS = 15
PATIENCE = 3  # Early stopping patience

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Ensure reproducibility
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()

# ==========================================
# 2. Data Processing & Caching
# ==========================================


def extract_slice_number(path):
    """Extracts the integer slice number from the filename (e.g., Image-10.dcm -> 10)."""
    basename = os.path.basename(path)
    # Match digits after 'Image-' and before '.dcm'
    match = re.search(r"Image-(\d+)\.dcm", basename)
    if match:
        return int(match.group(1))
    return 0


def load_dicom_slice(path, target_size=IMAGE_SIZE):
    """Reads a DICOM file, normalizes it locally if needed (though we do volume norm), and resizes."""
    full_path = os.path.join(INPUT_DIR, path)
    try:
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)

        # Resize
        if img.shape != (target_size, target_size):
            img = cv2.resize(
                img, (target_size, target_size), interpolation=cv2.INTER_CUBIC
            )

        return img
    except Exception as e:
        # Return zero slice on failure
        return np.zeros((target_size, target_size), dtype=np.float32)


def process_modality_volume(
    paths, num_slices=SLICES_PER_MODALITY, target_size=IMAGE_SIZE
):
    """
    Processes a list of paths for a single modality:
    1. Sorts by slice number.
    2. Uniformly samples `num_slices`.
    3. Loads and resizes images.
    4. Performs View-Adaptive Per-Modality Normalization.
    """
    if not paths or len(paths) == 0:
        return np.zeros((num_slices, target_size, target_size), dtype=np.float32)

    # 1. Sort
    # Create (slice_num, path) tuples and sort
    sorted_paths = sorted(
        [(extract_slice_number(p), p) for p in paths], key=lambda x: x[0]
    )
    sorted_paths = [p[1] for p in sorted_paths]

    # 2. Uniform Sampling (10% to 90% depth to avoid edges, or full range if few slices)
    total_slices = len(sorted_paths)
    if total_slices < num_slices:
        # Pad if not enough slices
        indices = np.linspace(0, total_slices - 1, num_slices).astype(int)
    else:
        # Sample
        indices = np.linspace(0, total_slices - 1, num_slices).astype(int)

    selected_paths = [sorted_paths[i] for i in indices]

    # 3. Load Images
    volume = []
    for p in selected_paths:
        img = load_dicom_slice(p, target_size)
        volume.append(img)

    volume = np.array(volume)  # Shape: (16, 320, 320)

    # 4. View-Adaptive Per-Modality Normalization
    v_min = volume.min()
    v_max = volume.max()

    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    return volume


def process_patient(row):
    """
    Processes all 4 modalities for a patient and stacks them.
    Returns tensor of shape (64, 320, 320).
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    chunks = []

    for mod in modalities:
        col_name = f"{mod}_paths"
        paths = row[col_name]
        # Handle None or empty lists
        if paths is None:
            paths = []
        # Convert numpy array to list if necessary
        if isinstance(paths, np.ndarray):
            paths = paths.tolist()

        mod_vol = process_modality_volume(paths, SLICES_PER_MODALITY, IMAGE_SIZE)
        chunks.append(mod_vol)

    # Stack: (4*16, 320, 320) -> (64, 320, 320)
    full_volume = np.concatenate(chunks, axis=0)
    return full_volume


def load_processed_data(split_name, load_cached_data=True):
    """
    Loads data for a specific split ('train', 'val', 'test').
    Uses caching mechanism.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_X_path = os.path.join(WORKING_DIR, f"cached_{split_name}_X.npy")
    cache_y_path = os.path.join(WORKING_DIR, f"cached_{split_name}_y.npy")
    cache_ids_path = os.path.join(WORKING_DIR, f"cached_{split_name}_ids.npy")

    # Check cache
    if load_cached_data and os.path.exists(cache_X_path):
        print(f"Loading cached {split_name} data from {WORKING_DIR}...")
        X = np.load(cache_X_path)
        ids = np.load(cache_ids_path, allow_pickle=True)
        if split_name != "test":
            y = np.load(cache_y_path)
            return X, y, ids
        return X, None, ids

    # Process from scratch
    print(f"Processing {split_name} data from scratch...")
    meta_path = os.path.join(METADATA_DIR, f"{split_name}.parquet")
    df = pd.read_parquet(meta_path)

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        vol = process_patient(row)
        X_list.append(vol)
        ids_list.append(str(row["BraTS21ID"]))

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save cache
    np.save(cache_X_path, X)
    np.save(cache_ids_path, ids)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
        np.save(cache_y_path, y)
        return X, y, ids

    return X, None, ids


# ==========================================
# 3. Dataset & Model
# ==========================================


class MGMTDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X shape: (C, H, W)
        img = self.X[idx]
        if self.y is not None:
            label = self.y[idx]
            return torch.tensor(img, dtype=torch.float32), torch.tensor(
                label, dtype=torch.float32
            )
        return torch.tensor(img, dtype=torch.float32)


class HRLNNet(nn.Module):
    """
    High-Resolution Layer-Normalized 2.5D Network.
    Backbone: ConvNeXt-Tiny (LayerNorm based).
    Input: (B, 64, 320, 320).
    """

    def __init__(self):
        super(HRLNNet, self).__init__()
        # Load ConvNeXt Tiny
        # in_chans=64 triggers timm to adapt the first conv layer weights
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            in_chans=TOTAL_CHANNELS,
            num_classes=0,  # Remove head
            drop_path_rate=0.2,
        )

        # Get feature dimension (usually 768 for tiny)
        self.num_features = self.backbone.num_features

        # Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features), nn.Linear(self.num_features, 1)
        )

    def forward(self, x):
        # x: (B, 64, 320, 320)
        features = self.backbone(x)
        logits = self.head(features)
        return logits.squeeze(1)


# ==========================================
# 4. Training & Inference Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def train_model(load_cached_data=True):
    # 1. Load Data
    X_train, y_train, _ = load_processed_data("train", load_cached_data)
    X_val, y_val, _ = load_processed_data("val", load_cached_data)

    train_dataset = MGMTDataset(X_train, y_train)
    val_dataset = MGMTDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Setup Model
    model = HRLNNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 3. Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_auc = validate(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    return best_model_path


def predict_and_submit(model_path, load_cached_data=True):
    # 1. Load Test Data
    X_test, _, ids_test = load_processed_data("test", load_cached_data)
    test_dataset = MGMTDataset(X_test, None)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # 2. Load Model
    model = HRLNNet().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(
            "Warning: No trained model found. Using random initialization for prediction."
        )

    model.eval()
    predictions = []

    # 3. Inference
    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs)

    # 4. Save Submission
    submission_df = pd.DataFrame({"BraTS21ID": ids_test, "MGMT_value": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


def run_pipeline(load_cached_data=True):
    """
    Main entry point to run the full pipeline.
    """
    print("Initializing HRLN-Net Pipeline...")
    model_path = train_model(load_cached_data)
    predict_and_submit(model_path, load_cached_data)
    print("Pipeline completed successfully.")
