import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import cv2
from torch.nn.utils.rnn import pad_sequence

from library.config import Config
from library.models import SpineLocalizer, SliceEncoder, SequenceAggregator
from library.datasets import FeatureSequenceDataset
from library.losses import WeightedLogLoss
from library.utils import process_dicom, crop_image, save_cache, load_cache


def get_spine_centers(images, model, device):
    """
    Predicts spine centers for a batch of images using the Localizer.
    Args:
        images (list of np.ndarray): List of 512x512 images.
        model (nn.Module): SpineLocalizer.
        device (torch.device): Device.
    Returns:
        list of tuples: (center_y, center_x) for each image.
    """
    # Resize to Localizer input size (256x256)
    input_size = Config.LOCALIZER_IMG_SIZE
    batch_tensor = []

    for img in images:
        resized = cv2.resize(img, (input_size[1], input_size[0]))
        # Add channel dim: (1, H, W)
        tensor = torch.from_numpy(resized).unsqueeze(0).float()
        batch_tensor.append(tensor)

    batch_tensor = torch.stack(batch_tensor).to(device)

    with torch.no_grad():
        # Output: (Batch, 1, H, W)
        masks = torch.sigmoid(model(batch_tensor))
        masks = (masks > 0.5).float().cpu().numpy()

    centers = []
    for i in range(len(images)):
        mask = masks[i, 0]
        if np.sum(mask) > 0:
            # Calculate center of mass
            indices = np.argwhere(mask)
            y_center = np.mean(indices[:, 0])
            x_center = np.mean(indices[:, 1])

            # Scale back to original 512x512 size
            # Localizer is 256, Original is 512 -> Scale factor 2
            scale_y = images[i].shape[0] / input_size[0]
            scale_x = images[i].shape[1] / input_size[1]

            centers.append((y_center * scale_y, x_center * scale_x))
        else:
            # Fallback to image center if no spine detected
            centers.append((images[i].shape[0] // 2, images[i].shape[1] // 2))

    return centers


def extract_features(metadata_df, subset_name, load_cached_data=True, debug=False):
    """
    Extracts features for all studies in the metadata using Localizer and Encoder.
    """
    cache_filename = f"{subset_name}_features.npy"

    # 1. Try Loading Cache
    if load_cached_data:
        cached_data = load_cache(cache_filename, use_parquet=False)
        if cached_data is not None:
            print(f"Loaded cached features for {subset_name}.")
            return cached_data.item()

    print(f"Generating features for {subset_name}...")

    # 2. Setup Models
    device = Config.DEVICE

    # Load Localizer
    localizer = SpineLocalizer(pretrained=True).to(device)
    loc_path = os.path.join(Config.CHECKPOINT_DIR, "spine_localizer.pth")
    if os.path.exists(loc_path):
        localizer.load_state_dict(torch.load(loc_path, map_location=device))
        print(f"Loaded Localizer from {loc_path}")
    else:
        print("Warning: Localizer checkpoint not found. Using ImageNet weights.")
    localizer.eval()

    # Load Encoder
    encoder = SliceEncoder(backbone_name=Config.ENCODER_BACKBONE, pretrained=True).to(
        device
    )
    enc_path = os.path.join(Config.CHECKPOINT_DIR, "slice_encoder.pth")
    if os.path.exists(enc_path):
        encoder.load_state_dict(torch.load(enc_path, map_location=device))
        print(f"Loaded Encoder from {enc_path}")
    else:
        print("Warning: Encoder checkpoint not found. Using ImageNet weights.")
    encoder.eval()

    # 3. Processing Loop
    features_dict = {}
    unique_studies = metadata_df["StudyInstanceUID"].unique()

    # Batch size for slice processing (fitting in GPU memory)
    SLICE_BATCH_SIZE = 32

    for study_uid in unique_studies:
        try:
            row = metadata_df[metadata_df["StudyInstanceUID"] == study_uid].iloc[0]
            img_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

            # Get all slices
            dcm_files = [
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.endswith(".dcm")
            ]
            # Sort by slice number
            try:
                dcm_files.sort(
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                )
            except:
                dcm_files.sort()

            if not dcm_files:
                continue

            # Load all images for this study
            images = [process_dicom(f) for f in dcm_files]
            num_slices = len(images)

            # --- Localization ---
            # Process in batches
            centers = []
            for i in range(0, num_slices, SLICE_BATCH_SIZE):
                batch_imgs = images[i : i + SLICE_BATCH_SIZE]
                batch_centers = get_spine_centers(batch_imgs, localizer, device)
                centers.extend(batch_centers)

            # --- Encoding ---
            study_features = []

            # Prepare 2.5D stacks
            # We need to process stacks in batches
            stack_batch = []

            for i in range(num_slices):
                # Determine neighbors
                img_c = images[i]
                img_p = images[i - 1] if i > 0 else img_c
                img_n = images[i + 1] if i < num_slices - 1 else img_c

                cy, cx = centers[i]
                crop_h, crop_w = Config.ENCODER_CROP_SIZE

                crop_c = crop_image(img_c, (cy, cx), (crop_h, crop_w))
                crop_p = crop_image(img_p, (cy, cx), (crop_h, crop_w))
                crop_n = crop_image(img_n, (cy, cx), (crop_h, crop_w))

                # Stack: (3, H, W)
                stack = np.stack([crop_p, crop_c, crop_n], axis=0)
                stack_batch.append(torch.from_numpy(stack).float())

                # When batch is full or last slice
                if len(stack_batch) == SLICE_BATCH_SIZE or i == num_slices - 1:
                    batch_tensor = torch.stack(stack_batch).to(device)

                    with torch.no_grad():
                        # (Batch, Feature_Dim)
                        feats = encoder(batch_tensor)
                        study_features.append(feats.cpu())

                    stack_batch = []

            # Concatenate all features for the study
            if study_features:
                full_study_features = torch.cat(study_features, dim=0)  # (Seq_Len, Dim)
                features_dict[study_uid] = full_study_features

        except Exception as e:
            print(f"Error processing study {study_uid}: {e}")
            continue

    # 4. Save Cache
    save_cache(features_dict, cache_filename, use_parquet=False)

    return features_dict


def collate_fn(batch):
    """
    Custom collate function to handle variable sequence lengths.
    Args:
        batch: List of tuples (features, labels)
    """
    # Separate features and labels
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]

    # Pad features
    # features_list is list of (Seq_Len, Dim) tensors
    # Result: (Batch, Max_Seq_Len, Dim)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    # Stack labels
    labels_stacked = torch.stack(labels_list)

    return features_padded, labels_stacked


def train_aggregator(debug=False):
    """
    Trains the Stage 3 Sequence Aggregator (RNN).
    """
    # 1. Setup
    Config.setup()
    device = Config.DEVICE
    print(f"Starting Aggregator Training on device: {device}")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        print("Debug mode: limiting metadata size.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Extract Features (Pre-compute)
    # This step uses the GPU heavily but only once per dataset
    train_features = extract_features(
        train_df, "train", load_cached_data=True, debug=debug
    )
    val_features = extract_features(val_df, "val", load_cached_data=True, debug=debug)

    # 4. Prepare Datasets and Loaders
    print("Initializing Datasets...")
    train_dataset = FeatureSequenceDataset(train_features, train_df)
    val_dataset = FeatureSequenceDataset(val_features, val_df)

    if len(train_dataset) == 0:
        print("No training data available.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.SEQ_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.SEQ_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 5. Initialize Model
    # Determine input dimension from features
    sample_uid = list(train_features.keys())[0]
    input_dim = train_features[sample_uid].shape[1]

    model = SequenceAggregator(
        input_dim=input_dim,
        hidden_dim=Config.RNN_HIDDEN_SIZE,
        num_layers=Config.RNN_NUM_LAYERS,
        dropout=Config.RNN_DROPOUT,
    )
    model.to(device)

    criterion = WeightedLogLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.SEQ_LR)

    # 6. Training Loop
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    epochs = Config.SEQ_EPOCHS if not debug else 2

    print("Starting training loop...")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                logits = model(features)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.8f} - Val Loss: {avg_val_loss:.8f}"
        )

        # --- Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            save_path = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Aggregator training completed.")
