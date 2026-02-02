import os
import time
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from library.config import Config
from library.utils import setup_logger, calculate_weighted_log_loss, window_dicom
from library.models import UNetLocalizer, DualStreamEncoder, SpinalGraphAggregator
from library.data import (
    SegmentationDataset,
    DualStreamSliceDataset,
    FeatureSequenceDataset,
    prepare_slice_dataframe,
    load_dicom_array,
    HAS_NIBABEL,
)

# ---------------------------------------------------------
# Losses & Metrics
# ---------------------------------------------------------


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W)
        # targets: (B, H, W) with class indices 0-C

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Calculate Dice for each class
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average dice loss across classes (1 - dice)
        return 1.0 - dice.mean()


class WeightedMultilabelLoss(nn.Module):
    def __init__(self):
        super(WeightedMultilabelLoss, self).__init__()
        # Weights: Patient Overall = 1.0, C1-C7 = 1/7
        self.weights = torch.tensor([1.0] + [1.0 / 7.0] * 7)

    def forward(self, y_pred_vert, y_pred_patient, y_true):
        # y_pred_vert: (B, 7) probabilities
        # y_pred_patient: (B, 1) probabilities
        # y_true: (B, 8) -> [patient_overall, C1, ..., C7]

        device = y_pred_vert.device
        weights = self.weights.to(device)

        # Concatenate predictions to match y_true shape: [patient, C1...C7]
        y_pred = torch.cat([y_pred_patient, y_pred_vert], dim=1)

        # Clamp for stability
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1.0 - eps)

        # Binary Cross Entropy
        # L = - [y * log(p) + (1-y) * log(1-p)]
        loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

        # Apply weights
        weighted_loss = loss * weights.unsqueeze(0)

        # Average over all elements (Batch * 8)
        return weighted_loss.mean()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------
# Trainer Class
# ---------------------------------------------------------


class Trainer:
    def __init__(self):
        self.logger = setup_logger("Trainer")
        self.device = Config.DEVICE

        # Create directories
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Paths for checkpoints
        self.path_seg = os.path.join(Config.CHECKPOINT_DIR, "stage1_unet.pth")
        self.path_enc = os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth")
        self.path_agg = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, model, path):
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
            self.logger.info(f"Loaded checkpoint: {path}")
        else:
            self.logger.warning(f"Checkpoint not found: {path}")

    # ---------------------------------------------------------
    # Stage 1: Localizer Training
    # ---------------------------------------------------------
    def train_localizer(self, epochs=Config.EPOCHS_SEG):
        self.logger.info("Starting Stage 1: Localizer Training")

        train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)

        # Filter Logic: Use segmentation if available, else use bounding box if nibabel missing
        if HAS_NIBABEL:
            seg_meta = train_meta[train_meta["has_segmentation"]].reset_index(drop=True)
            self.logger.info(f"Using {len(seg_meta)} samples with NIFTI segmentations.")
        else:
            self.logger.warning(
                "Nibabel missing. Using samples with Bounding Boxes for fallback."
            )
            # Ensure we have has_bounding_box column
            if "has_bounding_box" in train_meta.columns:
                seg_meta = train_meta[train_meta["has_bounding_box"]].reset_index(
                    drop=True
                )
            else:
                self.logger.error("No has_bounding_box column found.")
                seg_meta = pd.DataFrame()
            self.logger.info(f"Using {len(seg_meta)} samples with Bounding Boxes.")

        if len(seg_meta) == 0:
            self.logger.error("No training data available for Localizer.")
            return

        val_size = max(1, int(len(seg_meta) * 0.2))
        train_seg_df = seg_meta.iloc[:-val_size]
        val_seg_df = seg_meta.iloc[-val_size:]

        train_dataset = SegmentationDataset(train_seg_df)
        val_dataset = SegmentationDataset(val_seg_df)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE_SEG,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE_SEG,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Model
        model = UNetLocalizer(n_classes=8).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.LR_SEG)

        criterion_ce = nn.CrossEntropyLoss()
        criterion_dice = DiceLoss()

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = AverageMeter()

            for imgs, masks in train_loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                logits = model(imgs)

                loss_ce = criterion_ce(logits, masks)
                loss_dice = criterion_dice(logits, masks)
                loss = loss_ce + loss_dice

                loss.backward()
                optimizer.step()

                train_loss.update(loss.item(), imgs.size(0))

            # Val
            model.eval()
            val_loss = AverageMeter()
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device)

                    logits = model(imgs)
                    loss = criterion_ce(logits, masks) + criterion_dice(logits, masks)
                    val_loss.update(loss.item(), imgs.size(0))

            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss.avg:.6f} | Val Loss: {val_loss.avg:.6f}"
            )

            if val_loss.avg < best_val_loss:
                best_val_loss = val_loss.avg
                self.save_checkpoint(model, self.path_seg)

    # ---------------------------------------------------------
    # Stage 2: Encoder Training
    # ---------------------------------------------------------
    def train_encoder(self, epochs=Config.EPOCHS_CLS):
        self.logger.info("Starting Stage 2: Encoder Training")

        # Data
        slice_df = prepare_slice_dataframe(load_cached_data=True)
        # Split train/val based on StudyInstanceUID to prevent leakage
        uids = slice_df["StudyInstanceUID"].unique()
        np.random.shuffle(uids)
        split = int(len(uids) * 0.8)
        train_uids = set(uids[:split])

        train_df = slice_df[slice_df["StudyInstanceUID"].isin(train_uids)].reset_index(
            drop=True
        )
        val_df = slice_df[~slice_df["StudyInstanceUID"].isin(train_uids)].reset_index(
            drop=True
        )

        train_dataset = DualStreamSliceDataset(train_df, phase="train")
        val_dataset = DualStreamSliceDataset(val_df, phase="test")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE_CLS,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE_CLS,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Model
        model = DualStreamEncoder(feature_dim=1280).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.LR_CLS)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = AverageMeter()

            for batch in train_loader:
                local_img = batch["local"].to(self.device)
                global_img = batch["global"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                # Forward returns features, but we need logits for training.
                # The DualStreamEncoder returns projected features.
                # We need a temporary classification head for this stage.
                # However, the provided model in models.py does NOT have a classification head in forward().
                # It returns 'out' which is features.
                # We will add a temporary linear layer here or assume the model provided is for feature extraction
                # and we need to attach a head.
                # Given strict instructions not to modify models.py, we attach a head here.

                features = model(local_img, global_img)
                # Simple linear probe for training
                # We can't save this head as part of the model state dict if we want to load it later strictly
                # But for training the encoder weights, we need backprop.
                # We'll create a head and optimize it together.
                if not hasattr(self, "cls_head"):
                    self.cls_head = nn.Linear(1280, 1).to(self.device)
                    # Add head params to optimizer
                    optimizer.add_param_group({"params": self.cls_head.parameters()})

                logits = self.cls_head(features)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                train_loss.update(loss.item(), local_img.size(0))

            # Val
            model.eval()
            val_loss = AverageMeter()
            with torch.no_grad():
                for batch in val_loader:
                    local_img = batch["local"].to(self.device)
                    global_img = batch["global"].to(self.device)
                    labels = batch["label"].to(self.device).unsqueeze(1)

                    features = model(local_img, global_img)
                    logits = self.cls_head(features)
                    loss = criterion(logits, labels)
                    val_loss.update(loss.item(), local_img.size(0))

            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss.avg:.6f} | Val Loss: {val_loss.avg:.6f}"
            )

            if val_loss.avg < best_val_loss:
                best_val_loss = val_loss.avg
                self.save_checkpoint(model, self.path_enc)

    # ---------------------------------------------------------
    # Feature Extraction
    # ---------------------------------------------------------
    def extract_features(self, load_cached_data=True):
        self.logger.info("Starting Feature Extraction")

        feature_dir = os.path.join(Config.CACHE_DIR, "features")
        os.makedirs(feature_dir, exist_ok=True)

        # Gather all UIDs
        train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

        all_uids = pd.concat([train_meta, val_meta, test_meta])[
            "StudyInstanceUID"
        ].unique()

        # Check if cache exists
        if load_cached_data:
            existing = [
                f.replace(".npy", "")
                for f in os.listdir(feature_dir)
                if f.endswith(".npy")
            ]
            remaining = set(all_uids) - set(existing)
            if len(remaining) == 0:
                self.logger.info("All features cached.")
                return
            self.logger.info(
                f"Found {len(existing)} cached, extracting {len(remaining)}..."
            )
            uids_to_process = list(remaining)
        else:
            uids_to_process = all_uids

        # Load Models
        seg_model = UNetLocalizer(n_classes=8).to(self.device)
        self.load_checkpoint(seg_model, self.path_seg)
        seg_model.eval()

        enc_model = DualStreamEncoder(feature_dim=1280).to(self.device)
        self.load_checkpoint(enc_model, self.path_enc)
        enc_model.eval()

        # Processing Loop
        for uid in tqdm(uids_to_process, desc="Extracting Features"):
            # Locate folder
            # Try train then test
            img_dir = os.path.join(Config.TRAIN_IMAGES_DIR, uid)
            if not os.path.exists(img_dir):
                img_dir = os.path.join(Config.TEST_IMAGES_DIR, uid)
            if not os.path.exists(img_dir):
                continue

            # Load all slices
            slice_files = sorted(
                glob.glob(os.path.join(img_dir, "*.dcm")),
                key=lambda x: int(os.path.basename(x).replace(".dcm", "")),
            )

            if not slice_files:
                continue

            # We process in batches to save memory
            batch_size = 16

            study_features = []
            study_probs = []

            for i in range(0, len(slice_files), batch_size):
                batch_files = slice_files[i : i + batch_size]

                # 1. Load Original Images
                orig_imgs = [
                    load_dicom_array(f, size=Config.IMG_SIZE_ORIG) for f in batch_files
                ]
                orig_imgs_np = np.array(orig_imgs)  # (B, 512, 512)

                # 2. Stage 1: Localization
                # Resize for UNet (256)
                imgs_seg = []
                for img in orig_imgs_np:
                    resized = cv2.resize(
                        img, (Config.IMG_SIZE_MODEL, Config.IMG_SIZE_MODEL)
                    )
                    imgs_seg.append(resized)
                imgs_seg = np.array(imgs_seg)
                imgs_seg = (imgs_seg - Config.PIXEL_MEAN) / Config.PIXEL_STD
                imgs_seg_t = (
                    torch.tensor(imgs_seg, dtype=torch.float32)
                    .unsqueeze(1)
                    .to(self.device)
                )

                with torch.no_grad():
                    logits_seg = seg_model(imgs_seg_t)
                    probs_seg = F.softmax(logits_seg, dim=1)  # (B, 8, 256, 256)

                # Calculate Anatomical Probs (Global Average Pooling of masks)
                # (B, 8)
                anat_probs = probs_seg.mean(dim=(2, 3)).cpu().numpy()
                study_probs.append(anat_probs)

                # Calculate Crop Centers from masks
                # We use the combined bone mask (classes 1-7)
                bone_mask = probs_seg[:, 1:, :, :].sum(dim=1)  # (B, 256, 256)

                # 3. Prepare Stage 2 Inputs
                local_batch = []
                global_batch = []

                for b in range(len(batch_files)):
                    # Global
                    g_img = imgs_seg[b]  # Already resized and normalized
                    global_batch.append(g_img)

                    # Local Crop
                    # Find center of mass of bone mask
                    mask_b = bone_mask[b].cpu().numpy()
                    M = cv2.moments(mask_b)
                    if M["m00"] > 0:
                        cx_model = int(M["m10"] / M["m00"])
                        cy_model = int(M["m01"] / M["m00"])
                        # Scale back to 512
                        scale = Config.IMG_SIZE_ORIG / Config.IMG_SIZE_MODEL
                        cx = int(cx_model * scale)
                        cy = int(cy_model * scale)
                    else:
                        cx, cy = Config.IMG_SIZE_ORIG // 2, Config.IMG_SIZE_ORIG // 2

                    # Crop from Original (512)
                    crop_size = Config.IMG_SIZE_MODEL
                    half = crop_size // 2
                    start_x = int(
                        np.clip(cx - half, 0, Config.IMG_SIZE_ORIG - crop_size)
                    )
                    start_y = int(
                        np.clip(cy - half, 0, Config.IMG_SIZE_ORIG - crop_size)
                    )

                    l_img = orig_imgs_np[
                        b, start_y : start_y + crop_size, start_x : start_x + crop_size
                    ]
                    if l_img.shape != (crop_size, crop_size):
                        l_img = cv2.resize(l_img, (crop_size, crop_size))

                    l_img = (l_img - Config.PIXEL_MEAN) / Config.PIXEL_STD

                    # Local Mask heuristic (thresholding local image)
                    l_mask = (l_img > 0.2).astype(np.float32)

                    l_combined = np.stack([l_img, l_mask], axis=0)  # (2, 256, 256)
                    local_batch.append(l_combined)

                local_t = torch.tensor(np.array(local_batch), dtype=torch.float32).to(
                    self.device
                )
                global_t = (
                    torch.tensor(np.array(global_batch), dtype=torch.float32)
                    .unsqueeze(1)
                    .to(self.device)
                )

                with torch.no_grad():
                    feats = enc_model(local_t, global_t)  # (B, 1280)
                    study_features.append(feats.cpu().numpy())

            # Save
            if study_features:
                full_feats = np.concatenate(study_features, axis=0)
                full_probs = np.concatenate(study_probs, axis=0)

                save_path = os.path.join(feature_dir, f"{uid}.npy")
                np.save(save_path, {"features": full_feats, "probs": full_probs})

    # ---------------------------------------------------------
    # Stage 3: Aggregator Training
    # ---------------------------------------------------------
    def train_aggregator(self, epochs=Config.EPOCHS_SEQ):
        self.logger.info("Starting Stage 3: Aggregator Training")

        # Ensure features are extracted
        self.extract_features(load_cached_data=True)

        feature_dir = os.path.join(Config.CACHE_DIR, "features")

        # Data
        train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

        train_dataset = FeatureSequenceDataset(train_meta, feature_dir, phase="train")
        val_dataset = FeatureSequenceDataset(
            val_meta, feature_dir, phase="train"
        )  # 'train' phase returns labels

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE_SEQ,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE_SEQ,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Model
        model = SpinalGraphAggregator(
            input_dim=1280,
            hidden_dim=Config.GRU_HIDDEN_DIM,
            gcn_dim=Config.GCN_HIDDEN_DIM,
        ).to(self.device)
        optimizer = optim.Adam(
            model.parameters(), lr=Config.LR_SEQ, weight_decay=Config.WEIGHT_DECAY
        )
        criterion = WeightedMultilabelLoss()

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = AverageMeter()

            for feats, probs, labels in train_loader:
                feats = feats.to(self.device)
                probs = probs.to(self.device)
                labels = labels.to(self.device)  # (B, 8)

                optimizer.zero_grad()
                vert_probs, patient_prob = model(feats, probs)

                loss = criterion(vert_probs, patient_prob, labels)
                loss.backward()
                optimizer.step()

                train_loss.update(loss.item(), feats.size(0))

            # Val
            model.eval()
            val_loss = AverageMeter()
            with torch.no_grad():
                for feats, probs, labels in val_loader:
                    feats = feats.to(self.device)
                    probs = probs.to(self.device)
                    labels = labels.to(self.device)

                    vert_probs, patient_prob = model(feats, probs)
                    loss = criterion(vert_probs, patient_prob, labels)
                    val_loss.update(loss.item(), feats.size(0))

            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss.avg:.6f} | Val Loss: {val_loss.avg:.6f}"
            )

            if val_loss.avg < best_val_loss:
                best_val_loss = val_loss.avg
                self.save_checkpoint(model, self.path_agg)
