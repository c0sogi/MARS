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
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import gc

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================

# Paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_31"
CACHE_DIR = WORKING_DIR  # Alias for clarity
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SUBMISSION_PATH = "./submission/submission.csv"
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# RN-WIV Strategy Settings
NUM_CHANNELS = 9
ROI_DEPTHS = [0.4, 0.5, 0.6]  # Relative depths: 40%, 50%, 60%
WEIGHT_INFLATION_RATIOS = {"center": 0.5, "periphery": 0.25}
INPUT_DROPOUT_PROB = 0.2

# Model Settings
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 1
DROPOUT_RATE = 0.3
IMG_SIZE = 224

# Training Settings
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 5
NUM_FOLDS = 5
NUM_WORKERS = 4
USE_AMP = True  # Automatic Mixed Precision

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Utilities & Caching
# ==========================================


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom_image(path):
    """
    Reads a DICOM file. Tries pydicom first, then cv2.
    Returns a numpy array or None.
    """
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        return img
    except:
        pass

    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except:
        pass
    return None


def get_roi_boundaries(df, load_cached_data=True):
    """
    Determines the start and end indices of the brain ROI for each subject/modality.
    Scans files to find where pixel max > 0.
    Uses caching to store results in parquet format.
    """
    cache_path = os.path.join(CACHE_DIR, "roi_boundaries_cache.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading ROI boundaries from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Computing ROI boundaries (this may take a while)...")
    results = []
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    # Process each subject
    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]
        subject_res = {"BraTS21ID": sid}

        for mod in modalities:
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            if not os.path.exists(full_path):
                subject_res[f"{mod}_start"] = 0
                subject_res[f"{mod}_end"] = 0
                continue

            files = sorted(
                [f for f in os.listdir(full_path) if f.endswith(".dcm")],
                key=lambda x: int(x.split("-")[-1].split(".")[0]),
            )

            if not files:
                subject_res[f"{mod}_start"] = 0
                subject_res[f"{mod}_end"] = 0
                continue

            # Efficient scanning: Check every 5th slice to find boundaries
            # Then refine? For speed, we just use the first/last non-empty found in stride.
            # A more robust way for 24h limit:
            # Read middle to check if valid.
            # Then binary search or linear scan with stride.

            # Linear scan with stride 5
            start_idx = 0
            end_idx = len(files) - 1

            # Find start
            found_start = False
            for i in range(0, len(files), 3):  # Stride 3 for better resolution
                img = load_dicom_image(os.path.join(full_path, files[i]))
                if img is not None and np.max(img) > 0:
                    start_idx = i
                    found_start = True
                    break

            # Find end
            found_end = False
            for i in range(len(files) - 1, -1, -3):
                img = load_dicom_image(os.path.join(full_path, files[i]))
                if img is not None and np.max(img) > 0:
                    end_idx = i
                    found_end = True
                    break

            if not found_start:
                start_idx = 0
            if not found_end:
                end_idx = len(files) - 1

            if end_idx < start_idx:
                start_idx, end_idx = 0, len(files) - 1

            subject_res[f"{mod}_start"] = start_idx
            subject_res[f"{mod}_end"] = end_idx
            subject_res[f"{mod}_count"] = len(files)

        results.append(subject_res)

    df_res = pd.DataFrame(results)
    df_res.to_parquet(cache_path)
    print(f"ROI boundaries saved to {cache_path}")
    return df_res


# ==========================================
# 3. Dataset & Transforms
# ==========================================


class RNWIVDataset(Dataset):
    def __init__(self, df, roi_df, transform=None, is_train=False):
        self.df = df
        self.roi_df = roi_df.set_index("BraTS21ID")
        self.transform = transform
        self.is_train = is_train
        self.modalities = ["flair", "t1wce", "t2w"]  # As per Idea: FLAIR, T1wCE, T2w
        # Note: Idea specifies 9 channels from these 3 modalities.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["BraTS21ID"]
        roi_data = self.roi_df.loc[sid]

        channels = []

        # We need 3 modalities x 3 depths = 9 channels
        # Order:
        # Depth 40%: [FLAIR, T1wCE, T2w]
        # Depth 50%: [FLAIR, T1wCE, T2w]
        # Depth 60%: [FLAIR, T1wCE, T2w]

        target_depths = ROI_DEPTHS  # [0.4, 0.5, 0.6]

        for depth_ratio in target_depths:
            for mod in self.modalities:
                # Get ROI info
                start = roi_data[f"{mod}_start"]
                end = roi_data[f"{mod}_end"]
                count = roi_data[f"{mod}_count"]

                # Calculate index
                roi_len = end - start
                if roi_len < 1:
                    target_idx = 0
                else:
                    target_idx = int(start + roi_len * depth_ratio)

                # Clamp
                target_idx = max(0, min(target_idx, count - 1))

                # Get file path
                rel_path = row[f"{mod}_path"]
                full_path = os.path.join(INPUT_DIR, rel_path)
                files = sorted(
                    [f for f in os.listdir(full_path) if f.endswith(".dcm")],
                    key=lambda x: int(x.split("-")[-1].split(".")[0]),
                )

                if not files:
                    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                else:
                    file_path = os.path.join(full_path, files[target_idx])
                    img = load_dicom_image(file_path)

                    if img is None:
                        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                    else:
                        img = img.astype(np.float32)
                        # Independent Channel Min-Max Scaling
                        if img.max() > 0:
                            img = (img - img.min()) / (img.max() - img.min())
                        else:
                            img = np.zeros_like(img)

                        # Resize
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                channels.append(img)

        # Stack to (H, W, 9) for Albumentations
        image = np.stack(channels, axis=-1)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # (9, H, W) via ToTensorV2
        else:
            # Manual ToTensor
            image = torch.from_numpy(image.transpose(2, 0, 1))  # (9, H, W)

        if "MGMT_value" in row:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return image, target
        else:
            return image, torch.tensor(-1.0)  # Dummy for test


def get_transforms(phase):
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(p=0.2),
                # No RandomBrightnessContrast as we did min-max norm?
                # Actually, intensity augs are okay, but translation is banned.
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


# ==========================================
# 4. Model Architecture
# ==========================================


class StructuredInputDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # x shape: (B, 9, H, W)
        # Groups:
        # 0: [0, 1, 2] (Depth 40%)
        # 1: [3, 4, 5] (Depth 50% - Center)
        # 2: [6, 7, 8] (Depth 60%)

        B = x.shape[0]
        mask = torch.ones_like(x)

        # For each sample in batch
        for i in range(B):
            if torch.rand(1).item() < self.p:
                # Decide whether to drop center or periphery
                if torch.rand(1).item() < 0.5:
                    # Drop Center (3,4,5)
                    mask[i, 3:6, :, :] = 0
                else:
                    # Drop Periphery (0,1,2 and 6,7,8)
                    mask[i, 0:3, :, :] = 0
                    mask[i, 6:9, :, :] = 0

        return x * mask


class RNWIVEfficientNet(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.input_dropout = StructuredInputDropout(p=INPUT_DROPOUT_PROB)

        # Create backbone
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Modify first layer to accept 9 channels
        # EfficientNet usually has conv_stem as first layer
        if hasattr(self.backbone, "conv_stem"):
            old_conv = self.backbone.conv_stem
            new_conv = nn.Conv2d(
                in_channels=NUM_CHANNELS,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            # Gaussian Weight Inflation
            # Copy weights
            w_orig = old_conv.weight.data  # (Out, 3, K, K)
            w_new = torch.zeros_like(new_conv.weight.data)  # (Out, 9, K, K)

            # Center (Channels 3,4,5) -> 50% energy
            w_new[:, 3:6, :, :] = w_orig * WEIGHT_INFLATION_RATIOS["center"]

            # Periphery (Channels 0,1,2) -> 25% energy
            w_new[:, 0:3, :, :] = w_orig * WEIGHT_INFLATION_RATIOS["periphery"]

            # Periphery (Channels 6,7,8) -> 25% energy
            w_new[:, 6:9, :, :] = w_orig * WEIGHT_INFLATION_RATIOS["periphery"]

            new_conv.weight.data = w_new
            self.backbone.conv_stem = new_conv
        else:
            raise AttributeError(
                "Backbone does not have 'conv_stem'. Check model architecture."
            )

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x):
        x = self.input_dropout(x)
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ==========================================
# 5. Training Logic
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
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
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(images)
                loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training():
    seed_everything(SEED)

    # Load Metadata
    df_train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(VAL_METADATA_PATH)

    # Combine for CV (The provided metadata splits are fixed, but we use 5-Fold on full train data as per Idea)
    # The Idea says: "5-Fold Cross-Validation grouped by Subject ID"
    # So we combine train and val metadata provided by the system to perform our own CV.
    df_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Get ROI Boundaries (Cached)
    roi_df = get_roi_boundaries(df_full, load_cached_data=True)

    # CV
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(df_full))

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_full, df_full["MGMT_value"])
    ):
        print(f"\n=== Fold {fold} ===")

        train_sub = df_full.iloc[train_idx].reset_index(drop=True)
        val_sub = df_full.iloc[val_idx].reset_index(drop=True)

        train_ds = RNWIVDataset(
            train_sub, roi_df, transform=get_transforms("train"), is_train=True
        )
        val_ds = RNWIVDataset(
            val_sub, roi_df, transform=get_transforms("val"), is_train=False
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        model = RNWIVEfficientNet().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

        best_auc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")

        for epoch in range(NUM_EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, DEVICE
            )
            val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # Load best model for OOF
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        model.eval()

        # OOF Inference
        fold_preds = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    fold_preds.extend(probs)

        oof_preds[val_idx] = np.array(fold_preds).flatten()

    total_auc = roc_auc_score(df_full["MGMT_value"], oof_preds)
    print(f"\nOverall CV AUC: {total_auc:.8f}")


def predict_and_submit():
    print("\nStarting Inference on Test Set...")
    seed_everything(SEED)

    df_test = pd.read_csv(TEST_METADATA_PATH)
    roi_df = get_roi_boundaries(df_test, load_cached_data=True)

    test_ds = RNWIVDataset(
        df_test, roi_df, transform=get_transforms("test"), is_train=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Ensemble predictions from all folds
    fold_models = []
    for fold in range(NUM_FOLDS):
        path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")
        if os.path.exists(path):
            model = RNWIVEfficientNet().to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            fold_models.append(model)

    if not fold_models:
        print("No models found. Check training.")
        return

    avg_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)
            batch_preds = []

            for model in fold_models:
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    batch_preds.append(probs)

            # Average across folds
            batch_avg = np.mean(batch_preds, axis=0)
            avg_preds.extend(batch_avg)

    # Create submission
    submission = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": np.array(avg_preds).flatten()}
    )

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


# Entry point function (optional, if external runner calls it)
def run():
    run_training()
    predict_and_submit()
