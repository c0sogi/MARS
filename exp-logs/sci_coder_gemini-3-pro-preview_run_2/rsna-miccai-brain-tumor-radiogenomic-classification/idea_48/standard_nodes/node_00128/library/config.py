import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score

# -----------------------------------------------------------------------------
# 1. Configuration & Constants
# -----------------------------------------------------------------------------

# Paths
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_48"
SUBMISSION_DIR = "./submission"
CACHE_FILE = os.path.join(WORKING_DIR, "roi_cache.parquet")

# Hyperparameters
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 3

# Model Settings
BACKBONE = "efficientnet_b0"
INPUT_CHANNELS = 12  # 4 modalities * 3 slices
GROUPS = 4
SLICE_STRIDE = 3
ROI_DEPTH_MIN = 0.15
ROI_DEPTH_MAX = 0.85
IMG_SIZE = 224
NUM_SLICES = 3

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed()

# -----------------------------------------------------------------------------
# 3. Data Processing & Caching
# -----------------------------------------------------------------------------


def load_dicom_slice(path):
    """
    Reads a DICOM file using OpenCV.
    """
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("OpenCV returned None")
        return img
    except Exception:
        # Fallback for corrupt/unreadable files to maintain pipeline stability
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)


def compute_roi_anchors(metadata_df, load_cached_data=True):
    """
    Computes or loads the anchor slice index for each subject.
    The anchor is the slice with maximum intensity sum in the FLAIR modality,
    restricted to the middle 15-85% of the volume.
    """
    if load_cached_data and os.path.exists(CACHE_FILE):
        print(f"Loading ROI cache from {CACHE_FILE}")
        return (
            pd.read_parquet(CACHE_FILE).set_index("BraTS21ID")["anchor_idx"].to_dict()
        )

    print("Computing ROI anchors (this may take a while)...")
    anchors = {}

    for idx, row in metadata_df.iterrows():
        subject_id = row["BraTS21ID"]
        flair_path = os.path.join(INPUT_DIR, row["path_FLAIR"])

        if not os.path.exists(flair_path):
            anchors[subject_id] = 0
            continue

        files = sorted(
            [f for f in os.listdir(flair_path) if f.endswith(".dcm")],
            key=lambda x: int(x.split("-")[-1].split(".")[0]) if "-" in x else 0,
        )

        num_slices = len(files)
        if num_slices == 0:
            anchors[subject_id] = 0
            continue

        start_idx = int(num_slices * ROI_DEPTH_MIN)
        end_idx = int(num_slices * ROI_DEPTH_MAX)

        # If volume is too small, use full range
        if start_idx >= end_idx:
            start_idx, end_idx = 0, num_slices

        max_intensity = -1
        best_idx = 0

        # Iterate through the valid range
        for i in range(start_idx, end_idx):
            img_path = os.path.join(flair_path, files[i])
            img = load_dicom_slice(img_path)
            current_intensity = np.sum(img)

            if current_intensity > max_intensity:
                max_intensity = current_intensity
                best_idx = i

        anchors[subject_id] = best_idx

    # Save cache
    print(f"Saving ROI cache to {CACHE_FILE}")
    df_cache = pd.DataFrame(list(anchors.items()), columns=["BraTS21ID", "anchor_idx"])
    df_cache.to_parquet(CACHE_FILE)

    return anchors


# -----------------------------------------------------------------------------
# 4. Dataset Class
# -----------------------------------------------------------------------------


class BraTSDataset(Dataset):
    def __init__(self, df, anchor_dict, phase="train", transform=None):
        self.df = df
        self.anchor_dict = anchor_dict
        self.phase = phase
        self.transform = transform
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]
        anchor_idx = self.anchor_dict.get(subject_id, 0)

        # Calculate slice indices: [-6, -3, 0, +3, +6] relative to anchor
        offsets = [
            -2 * SLICE_STRIDE,
            -1 * SLICE_STRIDE,
            0,
            1 * SLICE_STRIDE,
            2 * SLICE_STRIDE,
        ]
        slice_indices = [anchor_idx + o for o in offsets]

        channels = []

        for mod in self.modalities:
            mod_path = os.path.join(INPUT_DIR, row[f"path_{mod}"])
            files = sorted(
                [f for f in os.listdir(mod_path) if f.endswith(".dcm")],
                key=lambda x: int(x.split("-")[-1].split(".")[0]) if "-" in x else 0,
            )
            num_files = len(files)

            for s_idx in slice_indices:
                # Edge clamping
                read_idx = max(0, min(s_idx, num_files - 1))

                if num_files > 0:
                    img_path = os.path.join(mod_path, files[read_idx])
                    img = load_dicom_slice(img_path)
                else:
                    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

                # Resize
                img = cv2.resize(
                    img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
                )

                # Normalize [0, 1] per channel
                if np.max(img) > 0:
                    img = img.astype(np.float32) / np.max(img)
                else:
                    img = img.astype(np.float32)

                channels.append(img)

        # Stack channels: (20, 224, 224)
        # Order: FLAIR(5), T1w(5), T1wCE(5), T2w(5)
        tensor = np.stack(channels, axis=0)  # (C, H, W)
        tensor = torch.from_numpy(tensor)

        # Augmentation (Geometric only)
        if self.phase == "train":
            # Random Horizontal Flip
            if random.random() > 0.5:
                tensor = transforms.functional.hflip(tensor)

            # Random Vertical Flip
            if random.random() > 0.5:
                tensor = transforms.functional.vflip(tensor)

            # Random Rotation +/- 15
            angle = random.uniform(-15, 15)
            tensor = transforms.functional.rotate(
                tensor, angle, interpolation=transforms.InterpolationMode.BILINEAR
            )

        target = row["MGMT_value"] if "MGMT_value" in row else 0.5
        return tensor, torch.tensor(target, dtype=torch.float32)


# -----------------------------------------------------------------------------
# 5. Model Architecture
# -----------------------------------------------------------------------------


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class AsymmetricEfficientNet(nn.Module):
    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # Load backbone
        base_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # 1. Modify Stem
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        old_stem = base_model.features[0][0]

        # New Stem: 20 input channels, Groups=4
        new_stem = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=GROUPS,
            bias=False,
        )

        # 2. Asymmetric Initialization (Central Replication)
        # Old weights: (32, 3, 3, 3) -> (Out, In, K, K)
        # New weights: (32, 5, 3, 3) -> (Out, In/Groups, K, K)
        w_old = old_stem.weight.data
        w_new = torch.zeros(32, 5, 3, 3)

        # Replicate: [W0, W1, W1, W1, W2]
        # We assume the original 3 channels correspond to standard RGB structure.
        # We map: 0->0, 1->1,2,3, 2->4
        w_new[:, 0, :, :] = w_old[:, 0, :, :]
        w_new[:, 1, :, :] = w_old[:, 1, :, :]
        w_new[:, 2, :, :] = w_old[:, 1, :, :]
        w_new[:, 3, :, :] = w_old[:, 1, :, :]
        w_new[:, 4, :, :] = w_old[:, 2, :, :]

        new_stem.weight.data = w_new

        # Replace stem convolution
        base_model.features[0][0] = new_stem

        # 3. Insert SE Block after Stem
        # base_model.features is a Sequential. We reconstruct it.
        layers = list(base_model.features.children())
        # layers[0] is the Stem (ConvBNAct). Insert SE after it.
        layers.insert(1, SEBlock(32, reduction=4))

        self.features = nn.Sequential(*layers)
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier

        # Modify classifier for binary output
        # EfficientNet B0 classifier: Dropout -> Linear(1280, 1000)
        in_features = self.classifier[1].in_features
        self.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# -----------------------------------------------------------------------------
# 6. Training & Evaluation Functions
# -----------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training():
    # Load Metadata
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))

    # Compute Anchors (with caching)
    # Combine dfs for anchor computation to cover all subjects
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    anchors = compute_roi_anchors(df_all, load_cached_data=True)

    # Datasets & Loaders
    train_dataset = BraTSDataset(df_train, anchors, phase="train")
    val_dataset = BraTSDataset(df_val, anchors, phase="val")

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

    # Model Setup
    model = AsymmetricEfficientNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Starting training on {DEVICE}...")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

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


# -----------------------------------------------------------------------------
# 7. Inference & Submission
# -----------------------------------------------------------------------------


def predict_and_submit(model_path):
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Compute Anchors for test set
    anchors = compute_roi_anchors(df_test, load_cached_data=True)

    test_dataset = BraTSDataset(df_test, anchors, phase="test")
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = AsymmetricEfficientNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    predictions = []

    print("Starting inference with TTA...")

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(DEVICE)

            # TTA: Original, HFlip, VFlip
            # 1. Original
            out1 = torch.sigmoid(model(inputs))

            # 2. HFlip
            inputs_h = transforms.functional.hflip(inputs)
            out2 = torch.sigmoid(model(inputs_h))

            # 3. VFlip
            inputs_v = transforms.functional.vflip(inputs)
            out3 = torch.sigmoid(model(inputs_v))

            # Average
            avg_preds = (out1 + out2 + out3) / 3.0
            predictions.extend(avg_preds.cpu().numpy().flatten())

    # Prepare submission
    submission_df = df_test[["BraTS21ID"]].copy()
    submission_df["MGMT_value"] = predictions

    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
