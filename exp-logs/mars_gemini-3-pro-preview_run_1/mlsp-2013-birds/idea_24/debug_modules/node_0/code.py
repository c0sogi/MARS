import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_logger
from library.dataset import load_data, BirdDataset, get_transforms
from library.model import create_model
from library.training import train_model
from library.inference import predict_test_set, create_submission, save_pseudo_labels


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for demonstration...")

    # Override Config for speed and demonstration purposes
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Small subset to run fast
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.EPOCHS = 2

    # Set SWA to start at epoch 1 (0-indexed, so the second epoch) to demo the transition
    Config.TEACHER_SWA_START_EPOCH = 1

    # Define working directories for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"

    # Ensure clean slate for demo output
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set reproducible seed
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n>>> Loading Data...")

    # Load metadata (load_cached_data=False forces reading from CSVs and creating new subset)
    train_df, val_df, test_df = load_data(load_cached_data=False)

    # Verify DataFrames are not empty and respect DEBUG size
    assert (
        len(train_df) == Config.DEBUG_SUBSET_SIZE
    ), f"Train size mismatch: {len(train_df)}"
    assert len(val_df) == Config.DEBUG_SUBSET_SIZE, f"Val size mismatch: {len(val_df)}"
    assert (
        len(test_df) == Config.DEBUG_SUBSET_SIZE
    ), f"Test size mismatch: {len(test_df)}"

    print(
        f"Data Loaded successfully. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # -------------------------------------------------------------------------
    # 3. Dataset & DataLoader Instantiation
    # -------------------------------------------------------------------------
    print("\n>>> Creating Datasets and Loaders...")

    # Create Datasets
    train_ds = BirdDataset(train_df, transforms=get_transforms("train"), mode="train")
    val_ds = BirdDataset(val_df, transforms=get_transforms("val"), mode="val")
    test_ds = BirdDataset(test_df, transforms=get_transforms("test"), mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify Batch Shape
    images, targets = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Expected shape: (Batch, 3, 256, 640) and (Batch, 19)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), "Incorrect Image Shape"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect Target Shape"

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\n>>> Initializing Model...")

    model = create_model(pretrained=True)
    model = model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_out = model(images.to(device))
    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    print("Model initialized and forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Starting Training Loop (Demo: 2 Epochs)...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train model
    # We use a small mixup_alpha and define SWA parameters to trigger SWA logic in the 2nd epoch
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        mixup_alpha=0.2,
        swa_start_epoch=Config.TEACHER_SWA_START_EPOCH,
        swa_lr=Config.TEACHER_SWA_LR,
        epochs=Config.EPOCHS,
        save_dir=Config.WORKING_DIR,
        model_alias="demo_model",
        patience=5,
    )

    # Verify checkpoints were created
    expected_files = [
        "demo_model_base_best.pth",
        "demo_model_last.pth",
        "demo_model_swa.pth",
    ]
    for f in expected_files:
        path = os.path.join(Config.WORKING_DIR, f)
        assert os.path.exists(path), f"Checkpoint {f} was not created."
    print("Training complete. Checkpoints verified.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n>>> Performing Inference on Test Set...")

    # Run inference
    ids, probs = predict_test_set(trained_model, test_loader, device, tta=False)

    assert len(ids) == len(
        test_df
    ), "Number of predictions does not match test set size"
    assert probs.shape == (
        len(test_df),
        Config.NUM_CLASSES,
    ), "Probability matrix shape mismatch"

    # Generate Submission CSV
    sub_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    create_submission(ids, probs, sub_path)
    assert os.path.exists(sub_path), "Submission CSV not found"

    # Generate Pseudo-labels (Parquet)
    pseudo_path = os.path.join(Config.WORKING_DIR, "demo_pseudo_labels.parquet")
    save_pseudo_labels(ids, probs, pseudo_path)
    assert os.path.exists(pseudo_path), "Pseudo-label Parquet not found"

    print(f"Inference complete. Files saved to {Config.WORKING_DIR}")

    # Validate content of submission
    df_sub = pd.read_csv(sub_path)
    # Expected rows = num_samples * num_classes
    expected_rows = len(test_df) * Config.NUM_CLASSES
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    print("\n>>> Demonstration Completed Successfully!")


if __name__ == "__main__":
    main()
