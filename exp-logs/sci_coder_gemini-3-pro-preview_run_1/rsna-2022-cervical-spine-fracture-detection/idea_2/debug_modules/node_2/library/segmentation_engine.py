import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import read_dicom, save_checkpoint, load_checkpoint, seed_everything
from library.models import SpineLocalizer
from library.dataset import SegmentationDataset, process_slice_metadata

# Set device
DEVICE = Config.DEVICE


class LocalizerInferenceDataset(Dataset):
    """
    Simple dataset for running inference with the Localizer.
    Unlike SegmentationDataset, this does NOT filter for fractures and does NOT return masks.
    """

    def __init__(self, slice_df):
        self.df = slice_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(row["image_dir"], row["slice_file"])

        # Read image (H, W)
        image = read_dicom(image_path, Config.WINDOW_CENTER, Config.WINDOW_WIDTH)

        # Add channel dim: (1, H, W)
        image = torch.tensor(image).unsqueeze(0)

        return image, row["StudyInstanceUID"], row["slice_num"]


def train_localizer(
    num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Trains the SpineLocalizer U-Net model using bounding boxes as segmentation masks.
    """
    print(f"Starting Localizer Training on {DEVICE}...")
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        train_meta = train_meta.head(Config.DEBUG_DATASET_SIZE)
        val_meta = val_meta.head(Config.DEBUG_DATASET_SIZE)

    # 2. Process Slices
    # We rely on the library function to get the slice dataframe
    # Note: SegmentationDataset handles filtering for slices with bounding boxes internally
    train_slice_df = process_slice_metadata(
        train_meta, pd.read_csv(Config.TRAIN_BBOX_PATH), mode="train"
    )
    val_slice_df = process_slice_metadata(
        val_meta, pd.read_csv(Config.TRAIN_BBOX_PATH), mode="val"
    )

    # 3. Datasets & Loaders
    train_dataset = SegmentationDataset(train_slice_df)
    val_dataset = SegmentationDataset(val_slice_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples (fractured slices): {len(train_dataset)}")
    print(f"Val samples (fractured slices): {len(val_dataset)}")

    # 4. Model Setup
    model = SpineLocalizer(pretrained=Config.PREDICTION_DIR).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "spine_localizer.pth")

    # 5. Training Loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        # Training Step
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Validation Step
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_dataset)

        print(
            f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_path)
            print(f"  Saved new best model with val_loss: {best_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print("Localizer training complete.")


def generate_spine_coordinates(
    metadata_df, mode="test", load_cached_data=True, batch_size=Config.BATCH_SIZE * 2
):
    """
    Runs inference using the trained SpineLocalizer to find the center (x, y) of the spine
    for every slice in the provided metadata.

    Returns:
        dict: Mapping (StudyInstanceUID, slice_num) -> (x, y)
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"{mode}_spine_coords.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached spine coordinates from {cache_file}...")
        coords_df = pd.read_parquet(cache_file)
        # Convert to dictionary
        coords_map = {}
        for _, row in coords_df.iterrows():
            coords_map[(row["StudyInstanceUID"], row["slice_num"])] = (
                int(row["x"]),
                int(row["y"]),
            )
        return coords_map

    print(f"Generating spine coordinates for {mode} set...")

    # 2. Prepare Data
    # We need all slices, not just fractured ones.
    # We pass None for bbox_df because we don't need ground truth for inference dataset generation
    slice_df = process_slice_metadata(
        metadata_df, bbox_df=None, mode=mode, load_cached_data=load_cached_data
    )

    dataset = LocalizerInferenceDataset(slice_df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    model = SpineLocalizer(pretrained=False).to(DEVICE)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "spine_localizer.pth")

    if os.path.exists(checkpoint_path):
        model, _, _, _ = load_checkpoint(model, None, checkpoint_path, device=DEVICE)
        print("Loaded trained localizer weights.")
    else:
        print(
            "Warning: No trained localizer found. Using random weights (expect poor results)."
        )

    model.eval()

    # 4. Inference Loop
    results = []

    # Default center
    default_x, default_y = (
        Config.ORIGINAL_IMAGE_SIZE // 2,
        Config.ORIGINAL_IMAGE_SIZE // 2,
    )

    with torch.no_grad():
        # Using tqdm manually if allowed, otherwise silent or minimal print
        # Since instructions say "Only print the required information", we iterate silently or with simple log
        for batch_idx, (images, uids, slice_nums) in enumerate(loader):
            images = images.to(DEVICE)

            # Predict
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Move to CPU for processing
            probs = probs.cpu().numpy()  # (B, 1, H, W)

            for i in range(len(images)):
                mask = probs[i, 0]
                uid = uids[i]
                s_num = int(slice_nums[i])

                # Threshold
                binary_mask = (mask > 0.5).astype(np.uint8)

                # Calculate Center of Mass
                M = cv2.moments(binary_mask)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    # Fallback to center if nothing detected
                    cX, cY = default_x, default_y

                results.append(
                    {"StudyInstanceUID": uid, "slice_num": s_num, "x": cX, "y": cY}
                )

            if batch_idx % 100 == 0:
                print(f"Processed batch {batch_idx}/{len(loader)}")

    # 5. Save and Return
    coords_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    coords_df.to_parquet(cache_file)
    print(f"Saved coordinates to {cache_file}")

    coords_map = {}
    for _, row in coords_df.iterrows():
        coords_map[(row["StudyInstanceUID"], row["slice_num"])] = (
            int(row["x"]),
            int(row["y"]),
        )

    return coords_map
