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
from library.models import SegmentationUNet
from library.datasets import SegmentationDataset
from library.utils import load_dicom_windowed

# =========================================================================
# Loss Function
# =========================================================================


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Flatten predictions and targets
        preds = torch.sigmoid(preds).view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


# =========================================================================
# Training Function
# =========================================================================


def train_segmenter(train_df, val_df):
    """
    Trains the Stage 1 U-Net Localizer using the subset of data with segmentation labels.
    """
    print("Initializing Segmentation Training...")

    # Create Checkpoint Directory
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Datasets and Loaders
    # Note: SegmentationDataset handles the caching of GT masks internally
    train_dataset = SegmentationDataset(train_df, transform=None, load_cached_data=True)
    val_dataset = SegmentationDataset(val_df, transform=None, load_cached_data=True)

    if len(train_dataset) == 0:
        print("No segmentation data found. Skipping training.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.SEG_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.SEG_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model Setup
    device = Config.DEVICE
    # Model expects 3 channels (ResNet backbone), input is 1 channel. We will repeat in loop.
    model = SegmentationUNet(n_channels=3, n_classes=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.SEG_LR)
    criterion = DiceLoss()

    best_dice = 0.0
    model_save_path = os.path.join(Config.CHECKPOINT_DIR, "spine_localizer.pth")

    print(f"Starting training for {Config.SEG_EPOCHS} epochs on {device}...")

    for epoch in range(Config.SEG_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Repeat grayscale to 3 channels for ResNet backbone compatibility
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_dice_sum = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate Dice Score for monitoring
                preds = torch.sigmoid(outputs).view(-1)
                targs = masks.view(-1)
                intersection = (preds * targs).sum()
                dice = (2.0 * intersection + 1.0) / (preds.sum() + targs.sum() + 1.0)
                val_dice_sum += dice.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice_sum / len(val_loader)

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Dice: {avg_val_dice:.6f}"
        )

        # Checkpoint
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path}")

    print("Training completed.")


# =========================================================================
# Inference / Generation Function
# =========================================================================


class InferenceSliceDataset(Dataset):
    """
    Simple dataset to load all slices for a specific study during inference.
    """

    def __init__(self, dcm_files):
        self.dcm_files = dcm_files

    def __len__(self):
        return len(self.dcm_files)

    def __getitem__(self, idx):
        path = self.dcm_files[idx]
        # Load and window
        img = load_dicom_windowed(path)

        # Resize to model input size (512)
        if img.shape != (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE):
            img = cv2.resize(
                img, (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE)
            )

        # To Tensor (1, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        # Return path to identify slice index later
        return img_tensor, path


def generate_dataset_masks(metadata_df, load_cached_data=True):
    """
    Runs inference on the provided dataset (metadata_df) to generate segmentation masks.
    Saves masks as .npy files in Config.WORKING_DIR/masks/<StudyUID>/<SliceIdx>.npy.
    """
    mask_output_dir = os.path.join(Config.WORKING_DIR, "masks")
    flag_file = os.path.join(mask_output_dir, "completed.flag")

    # Cache Check
    if load_cached_data and os.path.exists(flag_file):
        print(f"Loading cached masks from {mask_output_dir}")
        return mask_output_dir

    print(f"Generating segmentation masks for {len(metadata_df)} studies...")
    os.makedirs(mask_output_dir, exist_ok=True)

    device = Config.DEVICE

    # Load Model
    model = SegmentationUNet(n_channels=3, n_classes=1).to(device)
    weights_path = os.path.join(Config.CHECKPOINT_DIR, "spine_localizer.pth")

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded trained segmentation model.")
    else:
        print(
            "WARNING: No trained model found. Using random weights (Expect poor results)."
        )

    model.eval()

    # Process by Study
    unique_studies = metadata_df["StudyInstanceUID"].unique()

    for study_uid in unique_studies:
        # Check if study is already processed (partial cache)
        study_dir = os.path.join(mask_output_dir, study_uid)
        if os.path.exists(study_dir) and len(os.listdir(study_dir)) > 0:
            continue

        os.makedirs(study_dir, exist_ok=True)

        # Get image directory
        # We assume metadata_df has 'image_path'
        row = metadata_df[metadata_df["StudyInstanceUID"] == study_uid].iloc[0]
        rel_path = row["image_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        # List all dcm files
        dcm_files = glob.glob(os.path.join(full_path, "*.dcm"))
        if not dcm_files:
            continue

        # Create Dataset & Loader
        dataset = InferenceSliceDataset(dcm_files)
        loader = DataLoader(
            dataset,
            batch_size=Config.SEG_BATCH_SIZE * 2,  # Can use larger batch for inference
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        with torch.no_grad():
            for images, paths in loader:
                images = images.to(device)

                # Repeat channels
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)

                # Inference
                logits = model(images)
                probs = torch.sigmoid(logits)

                # Binarize (B, 1, H, W)
                masks = (probs > 0.5).float().cpu().numpy()

                # Save individual slices
                for i, path in enumerate(paths):
                    # Filename is usually "1.dcm", "100.dcm"
                    fname = os.path.basename(path)
                    slice_idx = os.path.splitext(fname)[0]

                    # Extract mask (H, W)
                    mask_np = masks[i, 0, :, :].astype(np.uint8)

                    # Save
                    save_path = os.path.join(study_dir, f"{slice_idx}.npy")
                    np.save(save_path, mask_np)

    # Write completion flag
    with open(flag_file, "w") as f:
        f.write("done")

    print("Mask generation completed.")
    return mask_output_dir
