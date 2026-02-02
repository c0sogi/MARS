import os
import glob
import random
import time
import warnings
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==========================================
# 1. Configuration & Constants
# ==========================================
class Config:
    # Paths
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")
    INPUT_DIR = "./input"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_24")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Data Parameters
    IMAGE_SIZE = 224
    NUM_CHANNELS = 3  # 3 modalities (FLAIR, T1wCE, T2w)
    STRIDE = 0  # Not used for single slice
    MODALITIES = [
        "flair",
        "t1wce",
        "t2w",
    ]  # T1w excluded to save channels/compute if redundant, or use all.
    # Note: Prompt mentions 9 channels. 3 modalities * 3 depths = 9.
    # We select FLAIR, T1wCE, T2w as they are most informative for tumor core/edema.

    # Training Parameters
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    DROPOUT_RATE = 0.3
    N_FOLDS = 5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    BACKBONE = "efficientnet_b0"


# Ensure cache directory exists
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)


# ==========================================
# 2. Utilities
# ==========================================
def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dicom_image(path, size=Config.IMAGE_SIZE):
    """
    Reads a DICOM file. Uses cv2 as primary, falls back to simple reading if needed.
    Returns a normalized float32 image (H, W).
    """
    # Try reading with OpenCV (often works for simple DICOMs if library supports it)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        # Fallback: Create a black image if read fails (should not happen often)
        return np.zeros((size, size), dtype=np.float32)

    # Normalize to 0-1
    if img.max() > 0:
        img = img.astype(np.float32) / img.max()
    else:
        img = img.astype(np.float32)

    # Resize
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    return img


def get_subject_centroid(subject_row, input_dir, modalities):
    """
    Calculates the anatomical centroid (Z-axis index) for each modality.
    Returns a dict {modality: centroid_index}.
    """
    centroids = {}

    for mod in modalities:
        rel_path = subject_row[f"{mod}_path"]
        full_path = os.path.join(input_dir, rel_path)

        if not os.path.exists(full_path):
            centroids[mod] = 0
            continue

        files = sorted(
            glob.glob(os.path.join(full_path, "*.dcm")),
            key=lambda x: int(x.split("-")[-1].split(".")[0]),
        )

        if not files:
            centroids[mod] = 0
            continue

        # Optimization: Sample every 5th slice to speed up centroid finding
        # We need to find where the brain is.
        sample_step = max(1, len(files) // 20)
        indices = []
        masses = []

        for i in range(0, len(files), sample_step):
            f_path = files[i]
            img = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.max() > 0:
                # Simple mass calculation: sum of pixels > threshold
                # Threshold is dynamic: 10% of max intensity
                thresh = img.max() * 0.1
                mass = np.sum(img > thresh)
                if mass > 100:  # Filter noise
                    indices.append(i)
                    masses.append(mass)

        if indices:
            # Weighted average of indices
            centroid_idx = int(np.average(indices, weights=masses))
            # Map back to file list index
            centroids[mod] = centroid_idx
        else:
            # Fallback to middle slice
            centroids[mod] = len(files) // 2

    return centroids


def get_cached_centroids(df, input_dir, cache_name, load_cached_data=True):
    """
    Computes or loads cached centroids for the dataset.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"centroids_{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached centroids from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Computing centroids for {cache_name} (this may take a while)...")
    results = []
    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]
        centroids = get_subject_centroid(row, input_dir, Config.MODALITIES)
        res = {"BraTS21ID": sid}
        for mod in Config.MODALITIES:
            res[f"{mod}_centroid"] = centroids[mod]
        results.append(res)

    df_res = pd.DataFrame(results)
    df_res.to_parquet(cache_path)
    return df_res


# ==========================================
# 3. Dataset
# ==========================================
class MGMTVolumetricDataset(Dataset):
    def __init__(self, df, centroids_df, input_dir, transform=None, mode="train"):
        self.df = df
        self.centroids_df = centroids_df.set_index("BraTS21ID")
        self.input_dir = input_dir
        self.transform = transform
        self.mode = mode
        self.modalities = Config.MODALITIES
        self.stride = Config.STRIDE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["BraTS21ID"]

        # Get centroids
        if sid in self.centroids_df.index:
            centroids = self.centroids_df.loc[sid]
        else:
            # Fallback
            centroids = {f"{m}_centroid": 0 for m in self.modalities}

        channels = []

        # Order: [FLAIR(z-d), FLAIR(z), FLAIR(z+d), T1wCE(z-d)...]
        # But prompt strategy says:
        # Ch 0-2: [FLAIR, T1wCE, T2w] at z-delta
        # Ch 3-5: [FLAIR, T1wCE, T2w] at z
        # Ch 6-8: [FLAIR, T1wCE, T2w] at z+delta

        offsets = [-self.stride, 0, self.stride]

        # We construct the 9 channels by iterating offsets then modalities
        # This groups them spatially: all z-delta, then all z, then all z+delta

        for offset in offsets:
            for mod in self.modalities:
                rel_path = row[f"{mod}_path"]
                full_path = os.path.join(self.input_dir, rel_path)

                # Get file list
                files = sorted(
                    glob.glob(os.path.join(full_path, "*.dcm")),
                    key=lambda x: int(x.split("-")[-1].split(".")[0]),
                )

                if not files:
                    img = np.zeros(
                        (Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
                    )
                else:
                    # Determine index
                    center_idx = (
                        int(centroids[f"{mod}_centroid"])
                        if sid in self.centroids_df.index
                        else len(files) // 2
                    )
                    target_idx = center_idx + offset

                    # Clip to bounds
                    target_idx = max(0, min(target_idx, len(files) - 1))

                    img = load_dicom_image(files[target_idx])

                channels.append(img)

        # Stack: (H, W, 9)
        img_stack = np.stack(channels, axis=-1)

        if self.transform:
            augmented = self.transform(image=img_stack)
            img_tensor = augmented["image"]  # (9, H, W) via ToTensorV2
        else:
            # Manual to tensor
            img_tensor = torch.tensor(img_stack.transpose(2, 0, 1), dtype=torch.float32)

        if self.mode != "test":
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return img_tensor, target
        else:
            return img_tensor, sid


# ==========================================
# 4. Model: CA-WIV EfficientNet
# ==========================================
class CAWIVEfficientNet(nn.Module):
    def __init__(self, model_name=Config.BACKBONE, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1
        )

        # Adapt first layer for 9 channels with Gaussian Weight Inflation
        # EfficientNet stem is usually named 'conv_stem'
        if hasattr(self.backbone, "conv_stem"):
            old_layer = self.backbone.conv_stem
            in_channels = 9
            out_channels = old_layer.out_channels
            kernel_size = old_layer.kernel_size
            stride = old_layer.stride
            padding = old_layer.padding
            bias = old_layer.bias is not None

            new_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            )

            # Initialize weights
            # Shape: (Out, In, K, K) -> (32, 9, 3, 3)
            with torch.no_grad():
                original_weights = old_layer.weight  # (32, 3, 3, 3)

                # Modalities: FLAIR, T1wCE, T2w
                # Channels 0-2: z-delta (Peripheral) -> Weight 0.25
                # Channels 3-5: z (Center)       -> Weight 0.50
                # Channels 6-8: z+delta (Peripheral) -> Weight 0.25

                new_weights = torch.zeros_like(new_layer.weight)

                # We assume original weights correspond to RGB or 3-channel generic features.
                # We distribute this energy.

                # z-delta
                new_weights[:, 0:3, :, :] = original_weights * 0.25
                # z (Center)
                new_weights[:, 3:6, :, :] = original_weights * 0.50
                # z+delta
                new_weights[:, 6:9, :, :] = original_weights * 0.25

                new_layer.weight.copy_(new_weights)

            self.backbone.conv_stem = new_layer

        # Add Dropout to classifier
        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE),
                nn.Linear(self.backbone.classifier.in_features, 1),
            )

    def forward(self, x):
        return self.backbone(x)


# ==========================================
# 5. Training & Inference Engine
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
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
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def predict(model, loader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_preds.extend(probs.flatten())
            all_ids.extend(ids.numpy())

    return all_ids, all_preds


def run_pipeline(load_cached_data=True):
    set_seed()

    # 1. Load Metadata
    df_train_full = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # Combine train and val for CV splitting if desired, or use fixed split.
    # The prompt suggests 5-fold CV. We will merge and re-split or just use the provided split + CV on train.
    # To maximize data, we combine train+val and do 5-fold CV.
    df_all_train = pd.concat([df_train_full, df_val]).reset_index(drop=True)

    # 2. Prepare Centroids (Cache)
    centroids_train = get_cached_centroids(
        df_all_train, Config.INPUT_DIR, "train_val", load_cached_data
    )
    centroids_test = get_cached_centroids(
        df_test, Config.INPUT_DIR, "test", load_cached_data
    )

    # 3. Augmentations (No Translation/Scaling)
    train_transform = A.Compose(
        [
            A.Rotate(limit=15, p=0.5),
            A.GridDistortion(p=0.5),
            A.ElasticTransform(p=0.5),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.5),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # 4. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(df_all_train))
    test_preds_accum = np.zeros(len(df_test))

    print(f"Starting Training on {len(df_all_train)} samples...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_all_train, df_all_train["MGMT_value"])
    ):
        print(f"\n=== Fold {fold} ===")

        train_sub = df_all_train.iloc[train_idx]
        val_sub = df_all_train.iloc[val_idx]

        train_ds = MGMTVolumetricDataset(
            train_sub, centroids_train, Config.INPUT_DIR, transform=train_transform
        )
        val_ds = MGMTVolumetricDataset(
            val_sub, centroids_train, Config.INPUT_DIR, transform=val_transform
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model = CAWIVEfficientNet().to(Config.DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.NUM_EPOCHS
        )

        best_auc = 0.0
        best_model_path = os.path.join(Config.CACHE_DIR, f"model_fold{fold}.pth")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, Config.DEVICE
            )
            val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        print(f"Best AUC for Fold {fold}: {best_auc:.6f}")

        # Load best model for inference
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

        # OOF Predictions
        _, val_preds = predict(model, val_loader, Config.DEVICE)
        oof_preds[val_idx] = val_preds

        # Test Predictions
        test_ds = MGMTVolumetricDataset(
            df_test,
            centroids_test,
            Config.INPUT_DIR,
            transform=val_transform,
            mode="test",
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )
        _, fold_test_preds = predict(model, test_loader, Config.DEVICE)
        test_preds_accum += np.array(fold_test_preds) / Config.N_FOLDS

    overall_auc = roc_auc_score(df_all_train["MGMT_value"], oof_preds)
    print(f"\nOverall OOF AUC: {overall_auc:.6f}")

    # 5. Save Submission
    df_submission = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": test_preds_accum}
    )
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
