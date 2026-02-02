import os
import sys
import warnings
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import ISICDataset, get_transforms, load_dataset_dataframe
from library.model import ISICModel
from library.engine import fit, predict_and_submit


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Override Config parameters for a quick demonstration
    print("Configuring for quick demonstration...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples for speed

    # Ensure device is correctly detected
    print(f"Device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Loading & Verification
    # ---------------------------------------------------------
    print("Loading and verifying datasets...")

    # Load subsets of dataframes
    df_train = load_dataset_dataframe(
        Config.TRAIN_CSV, debug_size=Config.DEBUG_SUBSET_SIZE
    )
    df_val = load_dataset_dataframe(Config.VAL_CSV, debug_size=Config.DEBUG_SUBSET_SIZE)

    # Assertions to verify data loading
    assert len(df_train) == Config.DEBUG_SUBSET_SIZE, "Train subset size mismatch"
    assert len(df_val) == Config.DEBUG_SUBSET_SIZE, "Val subset size mismatch"

    # Initialize Datasets
    train_dataset = ISICDataset(
        df_train, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = ISICDataset(df_val, transforms=get_transforms("val"), mode="val")

    # Verify dataset item structure
    sample = train_dataset[0]
    assert "image" in sample, "Dataset sample missing 'image' key"
    assert "target" in sample, "Dataset sample missing 'target' key"
    assert sample["image"].shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {sample['image'].shape}"

    # Initialize DataLoaders
    # num_workers=0 for simple, error-free execution in demo script
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("DataLoaders initialized successfully.")

    # ---------------------------------------------------------
    # 3. Model Initialization & Verification
    # ---------------------------------------------------------
    print("Initializing model...")
    model = ISICModel(pretrained=True)
    model.to(Config.DEVICE)

    # Verify forward pass dimensions
    dummy_input = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(Config.DEVICE)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 1), got {dummy_output.shape}"

    print("Model initialized and verified.")

    # ---------------------------------------------------------
    # 4. Training Loop Execution
    # ---------------------------------------------------------
    print("Starting training loop...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    best_auc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # Verify training outputs
    assert isinstance(best_auc, float), "fit() did not return a float AUC score"
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"
    print(f"Training finished. Best AUC: {best_auc}")

    # ---------------------------------------------------------
    # 5. Inference & Submission
    # ---------------------------------------------------------
    print("Starting inference...")

    # Load test data subset
    df_test = load_dataset_dataframe(
        Config.TEST_CSV, debug_size=Config.DEBUG_SUBSET_SIZE
    )
    test_dataset = ISICDataset(df_test, transforms=get_transforms("test"), mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Load the best model weights
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )

    # Generate submission
    df_sub = predict_and_submit(
        model=model,
        test_loader=test_loader,
        device=Config.DEVICE,
        submission_path=Config.SUBMISSION_PATH,
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), "Submission dataframe length mismatch"
    assert list(df_sub.columns) == [
        "image_name",
        "target",
    ], "Submission columns mismatch"

    print(f"Submission generated at {Config.SUBMISSION_PATH}")
    print("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
