import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import glob

from library.config import Config
from library.utils import (
    WeightedLogLoss,
    save_features,
    get_roi_coordinates,
    load_dicom,
    apply_windowing,
)
from library.data import (
    SegmentationDataset,
    DualStreamDataset,
    SequenceDataset,
    process_segmentation_data,
    process_classification_data,
    get_transforms,
)
from library.models import (
    AnatomicalLocalizer,
    DualBranchEncoder,
    HierarchicalAggregator,
)

# =============================================================================
# Helpers
# =============================================================================


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, path="checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)


def train_epoch(model, loader, criterion, optimizer, device, stage_name="Model"):
    model.train()
    running_loss = 0.0

    for batch_idx, batch_data in enumerate(loader):
        if len(batch_data) == 2:
            inputs, targets = batch_data
        elif len(batch_data) == 3:
            inputs, targets, _ = batch_data

        # Move to device
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = inputs.to(device)

        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(*inputs) if isinstance(inputs, list) else model(inputs)

        # Handle tuple outputs (e.g. Stage 1 returns mask, probs; Stage 2 returns emb, logits)
        if isinstance(outputs, tuple):
            # For Stage 1: outputs[0] is mask logits
            # For Stage 2: outputs[1] is logits
            if stage_name == "Stage1":
                loss = criterion(outputs[0], targets)
            elif stage_name == "Stage2":
                loss = criterion(outputs[1].squeeze(), targets)
            else:
                loss = criterion(outputs, targets)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_epoch(model, loader, criterion, device, stage_name="Model"):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 2:
                inputs, targets = batch_data
            elif len(batch_data) == 3:
                inputs, targets, _ = batch_data

            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(*inputs) if isinstance(inputs, list) else model(inputs)

            if isinstance(outputs, tuple):
                if stage_name == "Stage1":
                    loss = criterion(outputs[0], targets)
                elif stage_name == "Stage2":
                    loss = criterion(outputs[1].squeeze(), targets)
                else:
                    loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs, targets)

            running_loss += loss.item()

    return running_loss / len(loader)


# =============================================================================
# Stage 1 Trainer
# =============================================================================


class Stage1Trainer:
    def __init__(self):
        self.device = Config.DEVICE
        self.model = AnatomicalLocalizer(pretrained=True).to(self.device)
        self.checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "stage1_unet.pth")

    def train(self, epochs=None, batch_size=None):
        epochs = epochs if epochs is not None else Config.SEG_EPOCHS
        batch_size = batch_size if batch_size is not None else Config.SEG_BATCH_SIZE

        print(
            f"Starting Stage 1 Training (U-Net) on {self.device} (Epochs: {epochs})..."
        )

        # Data
        df_seg = process_segmentation_data(load_cached_data=True)
        if df_seg.empty:
            print("No segmentation data found. Skipping Stage 1 training.")
            return

        # Split (Simple random split since df_seg is slices, not patients)
        # Ideally we split by patient, but for simplicity here we just shuffle
        # In a real scenario, we'd filter by validation patients from metadata
        val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
        val_uids = set(val_meta["StudyInstanceUID"].unique())

        val_mask = df_seg["StudyInstanceUID"].isin(val_uids)
        train_df = df_seg[~val_mask]
        val_df = df_seg[val_mask]

        # If val is empty (e.g. segmentation subset doesn't overlap with val split), take random 10%
        if len(val_df) == 0:
            perm = np.random.permutation(len(df_seg))
            split = int(len(df_seg) * 0.9)
            train_df = df_seg.iloc[perm[:split]]
            val_df = df_seg.iloc[perm[split:]]

        train_ds = SegmentationDataset(
            train_df, transforms=get_transforms("train", Config.SEG_IMG_SIZE)
        )
        val_ds = SegmentationDataset(
            val_df, transforms=get_transforms("val", Config.SEG_IMG_SIZE)
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS
        )

        # Optimization
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.SEG_LR,
            weight_decay=Config.SEG_WEIGHT_DECAY,
        )
        early_stopping = EarlyStopping(patience=5, path=self.checkpoint_path)

        for epoch in range(epochs):
            train_loss = train_epoch(
                self.model,
                train_loader,
                criterion,
                optimizer,
                self.device,
                stage_name="Stage1",
            )
            val_loss = validate_epoch(
                self.model, val_loader, criterion, self.device, stage_name="Stage1"
            )

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # Load best
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        print("Stage 1 Training Complete.")


# =============================================================================
# Stage 2 Trainer
# =============================================================================


class Stage2Trainer:
    def __init__(self):
        self.device = Config.DEVICE
        self.model = DualBranchEncoder(pretrained=True).to(self.device)
        self.checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth")

    def train(self, epochs=None, batch_size=None):
        epochs = epochs if epochs is not None else Config.ENC_EPOCHS
        batch_size = batch_size if batch_size is not None else Config.ENC_BATCH_SIZE

        print(
            f"Starting Stage 2 Training (Dual Branch Encoder) on {self.device} (Epochs: {epochs})..."
        )

        # Data
        df_cls = process_classification_data(load_cached_data=True)

        # Split based on metadata
        val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
        val_uids = set(val_meta["StudyInstanceUID"].unique())

        val_mask = df_cls["StudyInstanceUID"].isin(val_uids)
        train_df = df_cls[~val_mask]
        val_df = df_cls[val_mask]

        train_ds = DualStreamDataset(
            train_df,
            transforms=get_transforms("train", Config.LOCAL_CROP_SIZE),
            mode="train",
        )
        val_ds = DualStreamDataset(
            val_df, transforms=get_transforms("val", Config.LOCAL_CROP_SIZE), mode="val"
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS
        )

        # Optimization
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.ENC_LR,
            weight_decay=Config.ENC_WEIGHT_DECAY,
        )
        early_stopping = EarlyStopping(patience=3, path=self.checkpoint_path)

        for epoch in range(epochs):
            train_loss = train_epoch(
                self.model,
                train_loader,
                criterion,
                optimizer,
                self.device,
                stage_name="Stage2",
            )
            val_loss = validate_epoch(
                self.model, val_loader, criterion, self.device, stage_name="Stage2"
            )

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        print("Stage 2 Training Complete.")


# =============================================================================
# Feature Extractor
# =============================================================================


class FeatureExtractor:
    def __init__(self):
        self.device = Config.DEVICE

        # Load Models
        self.seg_model = AnatomicalLocalizer(pretrained=False)
        seg_path = os.path.join(Config.CHECKPOINT_DIR, "stage1_unet.pth")
        if os.path.exists(seg_path):
            self.seg_model.load_state_dict(
                torch.load(seg_path, map_location=self.device)
            )
        self.seg_model.to(self.device).eval()

        self.enc_model = DualBranchEncoder(pretrained=False)
        enc_path = os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth")
        if os.path.exists(enc_path):
            self.enc_model.load_state_dict(
                torch.load(enc_path, map_location=self.device)
            )
        self.enc_model.to(self.device).eval()

        # Transforms for inference
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        self.seg_transform = A.Compose(
            [A.Resize(Config.SEG_IMG_SIZE[0], Config.SEG_IMG_SIZE[1]), ToTensorV2()]
        )
        self.global_transform = A.Compose(
            [A.Resize(Config.GLOBAL_SIZE[0], Config.GLOBAL_SIZE[1]), ToTensorV2()]
        )
        self.local_transform = A.Compose(
            [
                A.Resize(Config.LOCAL_CROP_SIZE[0], Config.LOCAL_CROP_SIZE[1]),
                ToTensorV2(),
            ]
        )

    def process_study(self, study_id, image_dir):
        # 1. Load all DICOMs
        dcm_files = glob.glob(os.path.join(image_dir, "*.dcm"))
        if not dcm_files:
            return None

        # Sort by slice number
        try:
            dcm_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except ValueError:
            dcm_files.sort()  # Fallback

        # Limit sequence length if too long (optional memory optimization)
        # But for accuracy we usually keep all.

        features_list = []

        # Batch processing slices to save time
        batch_size = 16

        for i in range(0, len(dcm_files), batch_size):
            batch_files = dcm_files[i : i + batch_size]

            # Load and Window
            imgs_hu = [load_dicom(f) for f in batch_files]
            imgs_raw = [apply_windowing(img) for img in imgs_hu]  # List of (H, W)

            # --- Stage 1: Segmentation ---
            seg_inputs = []
            for img in imgs_raw:
                # Resize for Seg
                aug = self.seg_transform(
                    image=img
                )  # Albumentations expects HWC usually, but img is HW.
                # Fix: Add channel dim for albumentations if needed, or just manual resize
                # Albumentations with grayscale:
                img_h, img_w = img.shape
                # Convert to HWC for A
                img_hwc = img[..., np.newaxis]
                aug = self.seg_transform(image=img_hwc)["image"]  # (1, 256, 256)
                seg_inputs.append(aug)

            seg_tensor = torch.stack(seg_inputs).to(self.device)

            with torch.no_grad():
                mask_logits, presence_probs = self.seg_model(seg_tensor)
                # mask_logits: (B, 8, 256, 256)
                # presence_probs: (B, 7)

            # Get masks for cropping
            masks_pred = torch.argmax(mask_logits, dim=1).cpu().numpy()  # (B, 256, 256)

            # --- Stage 2: Encoding ---
            local_inputs = []
            global_inputs = []

            for j, img in enumerate(imgs_raw):
                h, w = img.shape

                # Global Input
                img_3ch = np.stack([img, img, img], axis=-1)
                g_aug = self.global_transform(image=img_3ch)["image"]
                global_inputs.append(g_aug)

                # Local Input
                # Get ROI from mask (rescale mask to original size)
                mask_small = masks_pred[j]
                # Filter background (0)
                mask_binary = (mask_small > 0).astype(np.uint8)

                # Rescale mask to original image size for accurate ROI
                # Or calculate ROI on small mask and scale coordinates
                roi_ymin, roi_ymax, roi_xmin, roi_xmax = get_roi_coordinates(
                    mask_binary
                )

                # Scale factors
                scale_y = h / Config.SEG_IMG_SIZE[0]
                scale_x = w / Config.SEG_IMG_SIZE[1]

                cy = (roi_ymin + roi_ymax) / 2 * scale_y
                cx = (roi_xmin + roi_xmax) / 2 * scale_x

                # Crop logic
                crop_h, crop_w = Config.LOCAL_CROP_SIZE
                x1 = max(0, int(cx - crop_w // 2))
                y1 = max(0, int(cy - crop_h // 2))
                x2 = min(w, x1 + crop_w)
                y2 = min(h, y1 + crop_h)

                if x2 - x1 < crop_w:
                    x1 = max(0, w - crop_w)
                if y2 - y1 < crop_h:
                    y1 = max(0, h - crop_h)

                local_crop = img_3ch[y1 : y1 + crop_h, x1 : x1 + crop_w, :]
                if local_crop.shape[:2] != (crop_h, crop_w):
                    local_crop = cv2.resize(local_crop, (crop_w, crop_h))

                # Bone Mask Channel
                # We can use the upsampled predicted mask or heuristic.
                # Using heuristic for robustness as in Config
                mask_crop = (local_crop[..., 0] > 0.5).astype(np.float32)

                local_input_np = np.concatenate(
                    [local_crop, mask_crop[..., np.newaxis]], axis=-1
                )
                l_aug = self.local_transform(image=local_input_np)["image"]
                local_inputs.append(l_aug)

            local_tensor = torch.stack(local_inputs).to(self.device)
            global_tensor = torch.stack(global_inputs).to(self.device)

            with torch.no_grad():
                embeddings, _ = self.enc_model(local_tensor, global_tensor)
                # embeddings: (B, 512)

            # Concatenate Embedding + Anatomical Map
            # presence_probs: (B, 7)
            feats = torch.cat([embeddings, presence_probs], dim=1).cpu().numpy()
            features_list.append(feats)

        if not features_list:
            return None

        full_features = np.concatenate(features_list, axis=0)  # (Total_Slices, 519)
        return full_features

    def run(self, metadata_path, output_dir, image_root_dir):
        print(f"Running Feature Extraction for {metadata_path}...")
        df = pd.read_csv(metadata_path)
        studies = df["StudyInstanceUID"].unique()

        os.makedirs(output_dir, exist_ok=True)

        for study_id in studies:
            save_path = os.path.join(output_dir, f"{study_id}.npy")
            if os.path.exists(save_path):
                continue

            # Find image dir
            # The metadata has image_path relative to input
            # We construct full path
            # But df might have multiple rows. Get first.
            rel_path = df[df["StudyInstanceUID"] == study_id].iloc[0]["image_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            features = self.process_study(study_id, full_path)

            if features is not None:
                save_features(features, save_path)


# =============================================================================
# Stage 3 Trainer
# =============================================================================


class Stage3Trainer:
    def __init__(self):
        self.device = Config.DEVICE
        self.model = HierarchicalAggregator().to(self.device)
        self.checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, "fracture_aggregator.pth"
        )

    def train(self, epochs=None, batch_size=None):
        epochs = epochs if epochs is not None else Config.RNN_EPOCHS
        batch_size = batch_size if batch_size is not None else Config.RNN_BATCH_SIZE

        print(
            f"Starting Stage 3 Training (Bi-GRU Aggregator) on {self.device} (Epochs: {epochs})..."
        )

        feature_dir = os.path.join(Config.CACHE_DIR, "features")

        # Ensure features exist
        extractor = FeatureExtractor()
        extractor.run(Config.TRAIN_METADATA_PATH, feature_dir, Config.TRAIN_IMAGES_DIR)
        extractor.run(Config.VAL_METADATA_PATH, feature_dir, Config.TRAIN_IMAGES_DIR)

        # Load Dataframes
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)

        # Create Datasets
        # SequenceDataset handles loading .npy files
        train_ds = SequenceDataset(train_df, feature_dir, mode="train")
        val_ds = SequenceDataset(val_df, feature_dir, mode="val")

        # Collate function to handle variable sequence lengths
        def collate_fn(batch):
            # batch is list of (features, targets, study_id)
            features, targets, ids = zip(*batch)
            # Pad sequences
            features_padded = torch.nn.utils.rnn.pad_sequence(
                features, batch_first=True
            )
            targets_stacked = torch.stack(targets)
            return features_padded, targets_stacked, ids

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )  # num_workers 0 for safety with large arrays
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Optimization
        criterion = WeightedLogLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.RNN_LR,
            weight_decay=Config.RNN_WEIGHT_DECAY,
        )
        early_stopping = EarlyStopping(patience=5, path=self.checkpoint_path)

        for epoch in range(epochs):
            train_loss = train_epoch(
                self.model,
                train_loader,
                criterion,
                optimizer,
                self.device,
                stage_name="Stage3",
            )
            val_loss = validate_epoch(
                self.model, val_loader, criterion, self.device, stage_name="Stage3"
            )

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        print("Stage 3 Training Complete.")


# =============================================================================
# Submission Generator
# =============================================================================


class SubmissionGenerator:
    def __init__(self):
        self.device = Config.DEVICE
        self.model = HierarchicalAggregator().to(self.device)
        path = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def generate(self):
        print("Generating Submission...")

        # 1. Extract Test Features
        feature_dir = os.path.join(Config.CACHE_DIR, "test_features")
        extractor = FeatureExtractor()
        extractor.run(Config.TEST_METADATA_PATH, feature_dir, Config.TEST_IMAGES_DIR)

        # 2. Load Test Data
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        test_ds = SequenceDataset(test_meta, feature_dir, mode="test")

        def collate_fn(batch):
            features, targets, ids = zip(*batch)
            features_padded = torch.nn.utils.rnn.pad_sequence(
                features, batch_first=True
            )
            return features_padded, ids

        loader = DataLoader(
            test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0
        )

        results = []
        cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

        with torch.no_grad():
            for features, ids in loader:
                features = features.to(self.device)
                logits = self.model(features)
                probs = torch.sigmoid(logits).cpu().numpy()[0]  # (8,)

                study_id = ids[0]

                for i, col in enumerate(cols):
                    row_id = f"{study_id}_{col}"
                    results.append({"row_id": row_id, "fractured": probs[i]})

        # 3. Save
        sub_df = pd.DataFrame(results)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
