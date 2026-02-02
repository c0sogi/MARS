import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
import timm
from sklearn.metrics import roc_auc_score


# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Hyperparameters
    SEED = 42
    # Cite solution_lesson_node_00013: Reduce slices to 16 to prevent overfitting on small data
    NUM_SLICES = 16
    # Cite solution_lesson_node_00015: Reduce resolution to 224x224 (sufficient for signal, faster)
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    DROP_PATH_RATE = 0.2
    NUM_WORKERS = 4

    # Paths
    INPUT_DIR = "./input"
    TRAIN_META_PATH = "./metadata/train.parquet"
    VAL_META_PATH = "./metadata/val.parquet"
    TEST_META_PATH = "./metadata/test.parquet"

    # Output
    WORKING_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache
    CACHE_DIR = WORKING_DIR


# Ensure directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


# ==========================================
# DATASET & PROCESSING
# ==========================================
def load_dicom_volume(paths, size=256):
    """
    Loads a list of DICOM paths into a 3D numpy array (Depth, H, W).
    Performs resizing and global min-max normalization.
    """
    slices = []
    # Paths in metadata are relative to INPUT_DIR
    for p in paths:
        full_path = os.path.join(Config.INPUT_DIR, p)
        try:
            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array.astype(np.float32)
            # Resize
            if img.shape != (size, size):
                img = cv2.resize(img, (size, size))
            slices.append(img)
        except Exception as e:
            # Handle corrupt files by appending a black slice
            slices.append(np.zeros((size, size), dtype=np.float32))

    if not slices:
        return np.zeros((1, size, size), dtype=np.float32)

    volume = np.array(slices)

    # Global Min-Max Normalization
    min_val = volume.min()
    max_val = volume.max()
    if max_val - min_val > 0:
        volume = (volume - min_val) / (max_val - min_val)
    else:
        volume = np.zeros_like(volume)

    return volume


def get_indices(total_slices, num_selected):
    """
    Selects indices uniformly from the 10%-90% range of the volume.
    """
    if total_slices < num_selected:
        # Padding logic if volume is too small: repeat indices
        indices = np.linspace(0, total_slices - 1, num_selected).astype(int)
    else:
        # Exclude top/bottom 10%
        start = int(total_slices * 0.1)
        end = int(total_slices * 0.9)
        if end <= start:
            start, end = 0, total_slices

        indices = np.linspace(start, end - 1, num_selected).astype(int)

    return indices


class BraTSDataset(Dataset):
    def __init__(
        self, metadata_path, cache_name=None, load_cached_data=True, is_train=True
    ):
        self.df = pd.read_parquet(metadata_path)
        self.is_train = is_train
        self.cache_path_X = (
            os.path.join(Config.CACHE_DIR, f"{cache_name}_X.npy")
            if cache_name
            else None
        )
        self.cache_path_y = (
            os.path.join(Config.CACHE_DIR, f"{cache_name}_y.npy")
            if cache_name
            else None
        )
        self.cache_path_ids = (
            os.path.join(Config.CACHE_DIR, f"{cache_name}_ids.npy")
            if cache_name
            else None
        )

        self.X = None
        self.y = None
        self.ids = None

        if load_cached_data and self._check_cache():
            print(f"Loading cached data from {Config.CACHE_DIR}...")
            self.X = np.load(self.cache_path_X)
            self.ids = np.load(self.cache_path_ids, allow_pickle=True)
            if self.is_train:
                self.y = np.load(self.cache_path_y)
        else:
            print(f"Processing data from {metadata_path}...")
            self._process_and_cache()

    def _check_cache(self):
        if not self.cache_path_X:
            return False
        if not os.path.exists(self.cache_path_X):
            return False
        if not os.path.exists(self.cache_path_ids):
            return False
        if self.is_train and not os.path.exists(self.cache_path_y):
            return False
        return True

    def _process_and_cache(self):
        X_list = []
        y_list = []
        ids_list = []

        for idx, row in self.df.iterrows():
            bra_id = row["BraTS21ID"]

            # Load volumes for each modality
            paths_flair = row.get("flair_paths", [])
            paths_t1 = row.get("t1w_paths", [])
            paths_t1ce = row.get("t1wce_paths", [])
            paths_t2 = row.get("t2w_paths", [])

            vol_flair = load_dicom_volume(paths_flair, Config.IMG_SIZE)
            vol_t1 = load_dicom_volume(paths_t1, Config.IMG_SIZE)
            vol_t1ce = load_dicom_volume(paths_t1ce, Config.IMG_SIZE)
            vol_t2 = load_dicom_volume(paths_t2, Config.IMG_SIZE)

            # Construct 128-channel input
            # Interleaved: [F_0, T1_0, T1c_0, T2_0, F_1, ...]
            channels = []
            for vol in [vol_flair, vol_t1, vol_t1ce, vol_t2]:
                indices = get_indices(vol.shape[0], Config.NUM_SLICES)
                selected_slices = vol[indices]  # (32, 256, 256)
                channels.append(selected_slices)

            # Stack to (4, 32, 256, 256)
            stacked = np.stack(channels, axis=0)
            # Transpose to (32, 4, 256, 256) to get slice-first ordering
            stacked = stacked.transpose(1, 0, 2, 3)
            # Reshape to (128, 256, 256)
            combined = stacked.reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE)

            X_list.append(combined)
            ids_list.append(bra_id)
            if self.is_train:
                y_list.append(row["MGMT_value"])

        self.X = np.array(X_list, dtype=np.float32)
        self.ids = np.array(ids_list)
        if self.is_train:
            self.y = np.array(y_list, dtype=np.float32)

        # Save to cache
        if self.cache_path_X:
            np.save(self.cache_path_X, self.X)
            np.save(self.cache_path_ids, self.ids)
            if self.is_train:
                np.save(self.cache_path_y, self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.is_train:
            y = self.y[idx]
            return torch.tensor(x), torch.tensor(y).float()
        else:
            return torch.tensor(x), self.ids[idx]


# ==========================================
# MODEL
# ==========================================
class SHDNet(nn.Module):
    def __init__(self):
        super(SHDNet, self).__init__()

        # Stabilized Compression Stem
        # Compresses 128 channels (32 slices * 4 mods) to 64 channels
        self.stem = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Initialize Stem with Kaiming Normal
        nn.init.kaiming_normal_(
            self.stem[0].weight, mode="fan_out", nonlinearity="relu"
        )

        # Backbone: EfficientNet-B0
        # in_chans=64 to match stem output
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=64,
            drop_path_rate=Config.DROP_PATH_RATE,
            num_classes=0,  # Remove head
        )

        # Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        # x: (B, 128, 256, 256)
        x = self.stem(x)  # (B, 64, 256, 256)
        x = self.backbone.forward_features(x)  # (B, C, H, W)
        x = self.global_pool(x).flatten(1)  # (B, C)
        x = self.fc(x)  # (B, 1)
        return x


# ==========================================
# EXECUTION
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_workflow():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Datasets
    print("Initializing Datasets...")
    train_dataset = BraTSDataset(
        Config.TRAIN_META_PATH, cache_name="train", is_train=True
    )
    val_dataset = BraTSDataset(Config.VAL_META_PATH, cache_name="val", is_train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Model Setup
    model = SHDNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 3. Training Loop
    best_auc = 0.0
    print("Starting Training...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New Best Model Saved! (AUC: {best_auc:.6f})")

    # 4. Inference
    print("Starting Inference...")
    # Load Best Model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    test_dataset = BraTSDataset(
        Config.TEST_META_PATH, cache_name="test", is_train=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    results = []
    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            for bid, prob in zip(ids, probs):
                results.append({"BraTS21ID": bid, "MGMT_value": prob})

    # 5. Submission
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
