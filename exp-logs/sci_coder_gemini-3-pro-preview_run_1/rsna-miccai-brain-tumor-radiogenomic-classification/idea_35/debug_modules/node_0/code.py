import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import (
    TRAIN_METADATA_PATH,
    INPUT_DIR,
    IMG_SIZE,
    WORKING_DIR,
    DEVICE,
    SEED,
)
from library.utils import set_seed, AverageMeter
from library.data_processing import process_subject
from library.dataset import SICAVDataset, get_transforms
from library.model import SICAVModel
from library.train_eval import (
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
)


def run_demo():
    print("=== Starting SICAV Pipeline Demonstration ===\n")

    # 1. Setup
    set_seed(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ==========================================
    # Module 1: Data Processing (Real Data Test)
    # ==========================================
    print("--- Testing Data Processing (Single Subject) ---")

    # Load metadata to find a valid subject
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    assert len(df_train) > 0, "Training metadata is empty."

    # Pick the first subject
    sample_row = df_train.iloc[0]
    subject_id = sample_row["BraTS21ID"]
    print(f"Processing subject ID: {subject_id}")

    # Run the processing logic
    # This reads DICOMs, handles ROI, and stacks channels
    processed_tensor = process_subject(sample_row, INPUT_DIR)

    # Validation
    print(f"Output Tensor Shape: {processed_tensor.shape}")
    print(f"Output Tensor Type: {processed_tensor.dtype}")

    assert processed_tensor.shape == (
        IMG_SIZE,
        IMG_SIZE,
        9,
    ), f"Expected shape ({IMG_SIZE}, {IMG_SIZE}, 9), got {processed_tensor.shape}"
    assert (
        processed_tensor.dtype == np.float32
    ), f"Expected float32, got {processed_tensor.dtype}"
    assert (
        0.0 <= processed_tensor.min() and processed_tensor.max() <= 1.0
    ), "Image data should be normalized to [0, 1]"

    print("Data processing logic verified.\n")

    # ==========================================
    # Module 2: Dataset & DataLoader (Synthetic)
    # ==========================================
    print("--- Testing Dataset and DataLoader ---")

    # Generate synthetic data for speed (Batch of 4)
    num_samples = 4
    syn_ids = np.arange(num_samples)
    syn_images = np.random.rand(num_samples, IMG_SIZE, IMG_SIZE, 9).astype(np.float32)
    syn_labels = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)

    # Initialize Dataset with training transforms
    train_transform = get_transforms("train")
    dataset = SICAVDataset(syn_ids, syn_images, syn_labels, transforms=train_transform)

    # Check single item retrieval
    img_t, lbl_t = dataset[0]
    print(f"Dataset Item Shape (C, H, W): {img_t.shape}")

    # Albumentations + ToTensorV2 converts (H, W, C) -> (C, H, W)
    assert img_t.shape == (
        9,
        IMG_SIZE,
        IMG_SIZE,
    ), f"Expected tensor shape (9, {IMG_SIZE}, {IMG_SIZE}), got {img_t.shape}"
    assert isinstance(img_t, torch.Tensor), "Output should be a torch Tensor"

    # Initialize DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Fetch one batch
    batch_imgs, batch_lbls = next(iter(loader))
    print(f"Batch Shape: {batch_imgs.shape}")
    assert batch_imgs.shape == (2, 9, IMG_SIZE, IMG_SIZE), "Incorrect batch shape"

    print("Dataset and DataLoader verified.\n")

    # ==========================================
    # Module 3: Model Architecture
    # ==========================================
    print("--- Testing SICAV Model Architecture ---")

    # Initialize model (pretrained=False for speed/offline safety)
    model = SICAVModel(pretrained=False).to(DEVICE)

    # Verify first layer modification (Input channels should be 9)
    # EfficientNet usually names the first layer 'conv_stem'
    first_layer = getattr(model.backbone, "conv_stem", None)
    if first_layer is None:
        first_layer = getattr(model.backbone, "conv1", None)  # Fallback check

    assert first_layer is not None, "Could not locate first convolutional layer."
    print(f"First Layer In_Channels: {first_layer.in_channels}")
    assert (
        first_layer.in_channels == 9
    ), "First layer was not modified to accept 9 channels."

    # Forward pass with synthetic batch
    batch_imgs = batch_imgs.to(DEVICE)
    logits = model(batch_imgs)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (2, 1), "Output shape should be (Batch_Size, 1)"

    print("Model architecture verified.\n")

    # ==========================================
    # Module 4: Training & Evaluation Loop
    # ==========================================
    print("--- Testing Training and Evaluation Loop ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run 1 Training Epoch
    print("Running training step...")
    train_loss = train_one_epoch(model, loader, criterion, optimizer, DEVICE)
    print(f"Train Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float"

    # Run Evaluation
    print("Running evaluation step...")
    val_loss, val_auc = evaluate(model, loader, criterion, DEVICE)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Checkpoint Logic
    ckpt_path = os.path.join(WORKING_DIR, "demo_checkpoint.pth")
    save_checkpoint(model.state_dict(), ckpt_path)
    assert os.path.exists(ckpt_path), "Checkpoint file was not created."

    # Load Checkpoint
    loaded = load_checkpoint(model, ckpt_path, DEVICE)
    assert loaded, "Failed to load checkpoint."

    print("Training loop and checkpointing verified.\n")

    print("=== Demonstration Complete: All components functioning correctly. ===")


if __name__ == "__main__":
    run_demo()
