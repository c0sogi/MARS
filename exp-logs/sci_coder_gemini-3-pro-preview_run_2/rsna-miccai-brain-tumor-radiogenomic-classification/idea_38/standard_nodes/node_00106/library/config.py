import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import roc_auc_score
import random


# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
class Config:
    # Paths
    INPUT_DIR = "./input"
    TRAIN_LABELS = "./input/train_labels.csv"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_38"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Data Generation
    IMG_SIZE = 224
    NUM_SLICES = 3  # Depth per modality
    STRIDE = 5
    CHANNELS = 12  # 4 modalities * 3 slices

    # Training
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 3
    SEED = 42

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup():
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Set seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)


# ------------------------------------------------------------------------------
# 2. Data Processing Utilities
# ------------------------------------------------------------------------------
def read_dicom(path):
    """
    Robust DICOM reader. Tries OpenCV first, falls back to raw binary read.
    Assumes 512x512 uint16 if raw read is necessary based on file size.
    """
    # Method 1: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except:
        pass

    # Method 2: Raw Binary Tail-Read
    # 512 * 512 * 2 bytes = 524,288 bytes
    expected_bytes = 512 * 512 * 2
    try:
        file_size = os.path.getsize(path)
        if file_size >= expected_bytes:
            with open(path, "rb") as f:
                f.seek(-expected_bytes, os.SEEK_END)
                buffer = f.read(expected_bytes)
                img = np.frombuffer(buffer, dtype=np.uint16).reshape(512, 512)
                return img
    except:
        pass

    return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint16)


def load_subject_volume(subject_id, split_type, metadata_df):
    """
    Loads and aligns volume data for a single subject using Strict Geometric Alignment.
    """
    row = metadata_df[metadata_df["BraTS21ID"] == subject_id].iloc[0]

    # Modalities
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
    paths = {
        mod: os.path.join(Config.INPUT_DIR, row[f"path_{mod}"]) for mod in modalities
    }

    # 1. Analyze FLAIR to find anchor
    flair_files = sorted(
        glob.glob(os.path.join(paths["FLAIR"], "*.dcm")),
        key=lambda x: int(x.split("-")[-1].split(".")[0]),
    )

    if not flair_files:
        return np.zeros(
            (Config.CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    # Calculate integral for ROI selection (15% - 85% depth)
    n_files = len(flair_files)
    start_idx = int(n_files * 0.15)
    end_idx = int(n_files * 0.85)

    if start_idx >= end_idx:
        start_idx, end_idx = 0, n_files

    max_integral = -1
    anchor_idx = n_files // 2  # Default

    # Optimization: Read strided to find anchor faster if many files
    scan_step = 1 if n_files < 50 else 2

    for i in range(start_idx, end_idx, scan_step):
        img = read_dicom(flair_files[i])
        integral = np.sum(img)
        if integral > max_integral:
            max_integral = integral
            anchor_idx = i

    # 2. Define Target Indices
    # [Anchor-5, Anchor, Anchor+5]
    offsets = [-Config.STRIDE, 0, Config.STRIDE]
    target_indices = [anchor_idx + o for o in offsets]

    # 3. Load Data with Strict Alignment
    volume = []

    for mod in modalities:
        mod_dir = paths[mod]
        # Map slice index to filename: Image-{i+1}.dcm usually, but let's be robust
        # We list all files and parse IDs
        files = glob.glob(os.path.join(mod_dir, "*.dcm"))
        file_map = {}  # idx -> path
        for f in files:
            try:
                idx = int(os.path.basename(f).split("-")[-1].split(".")[0])
                # Adjust for 0-based indexing if files are 1-based (usually they are)
                # But our anchor_idx comes from the sorted list index of FLAIR.
                # We need to map list index to file ID for FLAIR, then use that ID for others?
                # NO. The prompt says "Spatial Continuity" vs "Cross-Modality Alignment".
                # Usually, Image-N in FLAIR corresponds to Image-N in T1w.
                # So we should use the integer ID from the filename.
                file_map[idx] = f
            except:
                pass

        # Get FLAIR IDs corresponding to the list indices we chose
        # We need the actual file IDs from the FLAIR list at those indices
        # If we used list indices, we need to convert to file IDs
        pass

    # RE-EVALUATING ALIGNMENT STRATEGY:
    # The list `flair_files` is sorted. `flair_files[anchor_idx]` has a specific ID, e.g., Image-105.dcm.
    # We assume the "Z-axis" corresponds to the sorted order.
    # For FLAIR, we use the list indices `target_indices`.
    # For other modalities, we must find the file that spatially corresponds.
    # In BraTS, usually files with same ID are registered.
    # So we get the ID from the FLAIR file, and look for the SAME ID in other modalities.

    # Get IDs for the chosen FLAIR slices
    flair_ids = []
    for ti in target_indices:
        # Clamp for FLAIR (Spatial Continuity)
        clamped_idx = max(0, min(n_files - 1, ti))
        f_path = flair_files[clamped_idx]
        try:
            fid = int(os.path.basename(f_path).split("-")[-1].split(".")[0])
            flair_ids.append(fid)
        except:
            flair_ids.append(-1)

    # Now load for all modalities
    for mod in modalities:
        mod_dir = paths[mod]
        mod_volume = []

        for fid in flair_ids:
            # Construct expected filename pattern or search
            # Fast lookup
            expected_path = os.path.join(mod_dir, f"Image-{fid}.dcm")

            img = None
            if os.path.exists(expected_path):
                img = read_dicom(expected_path)

            # Cross-Modality: Zero Padding if missing (NO Clamping)
            if img is None:
                img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
            else:
                # Resize
                img = cv2.resize(
                    img,
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                img = img.astype(np.float32)

            # Normalize [0, 1] per channel
            if np.max(img) > 0:
                img = (img - np.min(img)) / (np.max(img) - np.min(img))

            mod_volume.append(img)

        volume.extend(mod_volume)

    # Stack -> (12, 224, 224)
    return np.array(volume, dtype=np.float32)


def process_dataset(metadata_path, cache_name, load_cached_data=True):
    """
    Processes dataset with caching mechanism.
    """
    cache_path_data = os.path.join(Config.WORKING_DIR, f"{cache_name}_data.npy")
    cache_path_labels = os.path.join(Config.WORKING_DIR, f"{cache_name}_labels.npy")

    if (
        load_cached_data
        and os.path.exists(cache_path_data)
        and os.path.exists(cache_path_labels)
    ):
        print(f"Loading {cache_name} from cache...")
        data = np.load(cache_path_data)
        labels = np.load(cache_path_labels)
        return data, labels

    print(f"Processing {cache_name} from scratch...")
    df = pd.read_csv(metadata_path)

    data_list = []
    labels_list = []

    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]
        vol = load_subject_volume(sid, "train", df)
        data_list.append(vol)

        if "MGMT_value" in row:
            labels_list.append(row["MGMT_value"])
        else:
            labels_list.append(0.5)  # Dummy for test

    data = np.array(data_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)

    np.save(cache_path_data, data)
    np.save(cache_path_labels, labels)

    return data, labels


# ------------------------------------------------------------------------------
# 3. Dataset & Model
# ------------------------------------------------------------------------------
class BraTSDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        # Augmentations (Geometric only)
        if self.transform:
            # x is (C, H, W), convert to (H, W, C) for cv2
            x_np = x.transpose(1, 2, 0)

            # HFlip
            if random.random() > 0.5:
                x_np = cv2.flip(x_np, 1)
            # VFlip
            if random.random() > 0.5:
                x_np = cv2.flip(x_np, 0)
            # Rotate
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                h, w = x_np.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
                x_np = cv2.warpAffine(x_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            x = x_np.transpose(2, 0, 1)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


class EfficientNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True)

        # Modify Stem: 12 channels, Groups=4
        original_stem = self.backbone.conv_stem
        new_stem = nn.Conv2d(
            in_channels=12,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=False,
            groups=4,
        )

        # Direct Asymmetric Initialization
        # Copy weights: (32, 3, 3, 3) -> (32, 3, 3, 3)
        # Since groups=4, input channels per group is 12/4=3.
        # The weight tensor shape matches. We just copy.
        with torch.no_grad():
            new_stem.weight.data = original_stem.weight.data.clone()

        self.backbone.conv_stem = new_stem

        # Modify Head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.backbone.classifier.in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)


# ------------------------------------------------------------------------------
# 4. Training & Inference Logic
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    probs = []
    targets = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.5
    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    probs = []
    targets = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            probs.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.5
    return epoch_loss, epoch_auc


def predict_tta(model, inputs, device):
    """
    Test Time Augmentation: Original, HFlip, VFlip
    """
    model.eval()
    with torch.no_grad():
        # Original
        out1 = torch.sigmoid(model(inputs.to(device))).cpu().numpy()

        # HFlip
        inputs_h = torch.flip(inputs, [3])
        out2 = torch.sigmoid(model(inputs_h.to(device))).cpu().numpy()

        # VFlip
        inputs_v = torch.flip(inputs, [2])
        out3 = torch.sigmoid(model(inputs_v.to(device))).cpu().numpy()

    return (out1 + out2 + out3) / 3.0


def run_task():
    Config.setup()

    # 1. Load Data
    train_data, train_labels = process_dataset(
        os.path.join(Config.METADATA_DIR, "train.csv"), "train_cache"
    )
    val_data, val_labels = process_dataset(
        os.path.join(Config.METADATA_DIR, "val.csv"), "val_cache"
    )

    # 2. Datasets & Loaders
    train_dataset = BraTSDataset(train_data, train_labels, transform=True)
    val_dataset = BraTSDataset(val_data, val_labels, transform=False)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 3. Model Setup
    model = EfficientNetModel().to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss:.6f} AUC: {train_auc:.6f} | Val Loss: {val_loss:.6f} AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # 5. Inference
    print("Starting Inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    # Load test metadata
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Process test data (No caching for test usually, or separate cache)
    # We process on the fly to save memory or use same function
    test_data, _ = process_dataset(
        os.path.join(Config.METADATA_DIR, "test.csv"),
        "test_cache",
        load_cached_data=True,
    )

    predictions = []
    # Batch inference
    test_dataset = BraTSDataset(test_data, np.zeros(len(test_data)), transform=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    for inputs, _ in test_loader:
        preds = predict_tta(model, inputs, Config.DEVICE)
        predictions.extend(preds.flatten().tolist())

    # 6. Submission
    submission = pd.DataFrame(
        {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": predictions}
    )
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
