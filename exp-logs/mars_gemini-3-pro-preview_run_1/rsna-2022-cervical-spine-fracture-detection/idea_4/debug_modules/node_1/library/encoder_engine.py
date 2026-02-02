import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2

from library.config import Config
from library.models import MaskConditionedCNN
from library.datasets import CroppedSliceDataset
from library.utils import load_dicom_windowed, get_spine_crop_coords

# =========================================================================
# Helper Dataset for Inference
# =========================================================================


class PatientSliceInferenceDataset(Dataset):
    """
    Efficiently loads all slices for a single patient for feature extraction.
    Performs on-the-fly cropping based on the pre-computed masks.
    """

    def __init__(self, study_uid, image_dir, mask_dir, transform=None):
        self.study_uid = study_uid
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # List all DICOM files and sort by instance number
        self.files = []
        if os.path.exists(image_dir):
            raw_files = glob.glob(os.path.join(image_dir, "*.dcm"))
            # Sort by integer filename (1.dcm, 2.dcm, ...)
            try:
                raw_files.sort(
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                )
            except ValueError:
                raw_files.sort()

            for f in raw_files:
                # Extract index
                idx = int(os.path.splitext(os.path.basename(f))[0])
                self.files.append((idx, f))

        self.max_idx = self.files[-1][0] if self.files else 0

    def __len__(self):
        return len(self.files)

    def _load_slice(self, slice_idx):
        # Clamp index for context padding
        idx = max(1, min(slice_idx, self.max_idx))
        path = os.path.join(self.image_dir, f"{idx}.dcm")
        if os.path.exists(path):
            return load_dicom_windowed(path)
        return np.zeros(
            (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE), dtype=np.float32
        )

    def __getitem__(self, idx):
        slice_idx, dcm_path = self.files[idx]

        # 1. Load 3-slice stack (Context)
        img_prev = self._load_slice(slice_idx - 1)
        img_curr = self._load_slice(slice_idx)
        img_next = self._load_slice(slice_idx + 1)
        image_stack = np.stack([img_prev, img_curr, img_next], axis=-1)

        # 2. Load Mask
        mask = None
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.study_uid, f"{slice_idx}.npy")
            if os.path.exists(mask_path):
                mask = np.load(mask_path)

        if mask is None:
            mask = np.zeros(
                (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE),
                dtype=np.float32,
            )

        # 3. Crop
        y_min, y_max, x_min, x_max = get_spine_crop_coords(
            mask, image_size=Config.IMAGE_SIZE
        )

        image_crop = image_stack[y_min:y_max, x_min:x_max, :]
        mask_crop = mask[y_min:y_max, x_min:x_max]

        # Resize if necessary
        if image_crop.shape[:2] != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
            image_crop = cv2.resize(image_crop, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            mask_crop = cv2.resize(
                mask_crop,
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )

        # 4. Construct Input
        if Config.USE_MASK_INPUT:
            mask_crop = np.expand_dims(mask_crop, axis=-1)
            input_tensor = np.concatenate([image_crop, mask_crop], axis=-1)
        else:
            input_tensor = image_crop

        # To Tensor (C, H, W)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(2, 0, 1)

        return input_tensor


# =========================================================================
# Training Function
# =========================================================================


def train_encoder(train_df, val_df, mask_dir):
    """
    Trains the Stage 2 Mask-Conditioned CNN Encoder.

    Args:
        train_df: DataFrame with training metadata.
        val_df: DataFrame with validation metadata.
        mask_dir: Directory containing generated masks (from Stage 1).
    """
    print("Initializing Encoder Training...")

    # Create Checkpoint Directory
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Datasets
    # We use mode='train' for training to balance positives/negatives
    train_dataset = CroppedSliceDataset(train_df, mode="train", mask_dir=mask_dir)
    # We use mode='train' (or similar logic) for val to evaluate on a balanced subset efficiently,
    # or mode='val' to evaluate on everything. Given the size, let's use balanced for speed/metric relevance.
    # However, CroppedSliceDataset logic for 'val' takes all slices. Let's use 'train' mode for val
    # to keep it manageable and focused on classification capability, or 'val' if we want full scan coverage.
    # Let's stick to 'train' mode logic (balanced) for validation to monitor convergence on the classification task specifically.
    val_dataset = CroppedSliceDataset(val_df, mode="train", mask_dir=mask_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.CLS_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.CLS_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model Setup
    device = Config.DEVICE
    model = MaskConditionedCNN(pretrained=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.CLS_LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    model_save_path = os.path.join(Config.CHECKPOINT_DIR, "slice_encoder.pth")

    print(f"Starting training for {Config.CLS_EPOCHS} epochs on {device}...")

    for epoch in range(Config.CLS_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()
            outputs = model(inputs)  # Logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Accuracy
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss} | Val Acc: {val_acc}"
        )

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path}")

    print("Encoder training completed.")


# =========================================================================
# Feature Extraction Function
# =========================================================================


def extract_patient_features(metadata_df, mask_dir, load_cached_data=True):
    """
    Generates feature sequences for all patients in the metadata.

    Args:
        metadata_df: DataFrame containing study information.
        mask_dir: Directory containing generated masks.
        load_cached_data: If True, skips processing for studies that already have saved features.

    Returns:
        str: Path to the directory containing feature .npy files.
    """
    feature_dir = os.path.join(Config.CACHE_DIR, "features")
    os.makedirs(feature_dir, exist_ok=True)

    print(f"Extracting features for {len(metadata_df)} studies...")

    device = Config.DEVICE

    # Load Model
    model = MaskConditionedCNN(pretrained=False).to(device)
    weights_path = os.path.join(Config.CHECKPOINT_DIR, "slice_encoder.pth")

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded trained encoder weights.")
    else:
        print("WARNING: No trained encoder found. Using random weights.")

    model.eval()

    unique_studies = metadata_df["StudyInstanceUID"].unique()

    for study_uid in unique_studies:
        save_path = os.path.join(feature_dir, f"{study_uid}.npy")

        # Cache Check
        if load_cached_data and os.path.exists(save_path):
            continue

        # Get image directory
        row = metadata_df[metadata_df["StudyInstanceUID"] == study_uid].iloc[0]
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        if not os.path.exists(image_dir):
            continue

        # Create Dataset & Loader
        dataset = PatientSliceInferenceDataset(study_uid, image_dir, mask_dir)

        if len(dataset) == 0:
            # Handle empty study
            np.save(
                save_path, np.zeros((1, Config.ENCODER_HIDDEN_DIM), dtype=np.float32)
            )
            continue

        loader = DataLoader(
            dataset,
            batch_size=Config.CLS_BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        features_list = []

        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device)

                # Extract features (not logits)
                feats = model.forward_features(inputs)
                features_list.append(feats.cpu().numpy())

        if features_list:
            full_features = np.concatenate(features_list, axis=0)
            np.save(save_path, full_features)
        else:
            np.save(
                save_path, np.zeros((1, Config.ENCODER_HIDDEN_DIM), dtype=np.float32)
            )

    print("Feature extraction completed.")
    return feature_dir
