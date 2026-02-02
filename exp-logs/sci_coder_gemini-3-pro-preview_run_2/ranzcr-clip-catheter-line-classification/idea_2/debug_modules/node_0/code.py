import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import CatheterDataset
from library.model import CatheterModel
from library.engine import train_model, generate_submission


def run_demo():
    # --- 1. Setup & Configuration ---
    print("Initializing Configuration...")

    # Modify Config for fast demonstration
    Config.DEBUG = True  # Use subset (100 samples) for training/val to save time
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Minimal workers for demo script

    # Ensure directories exist
    Config.setup()

    # Set reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # --- 2. Data Preparation (Train/Val) ---
    print("\n[Step 1] Loading Training Data...")

    # Instantiate Datasets
    # In DEBUG mode, these will take a random sample of Config.DEBUG_SAMPLE_SIZE
    train_dataset = CatheterDataset(Config.TRAIN_METADATA_PATH, mode="train")
    val_dataset = CatheterDataset(Config.VAL_METADATA_PATH, mode="val")

    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Val Dataset Length: {len(val_dataset)}")

    # Validate Data Shapes
    sample_img, sample_label = train_dataset[0]
    assert sample_img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Image shape mismatch"
    assert sample_label.shape == (len(Config.TARGET_COLS),), "Label shape mismatch"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # --- 3. Model Initialization ---
    print("\n[Step 2] Initializing Model...")
    # pretrained=False to avoid downloading large weights during demo execution
    model = CatheterModel(pretrained=False)
    model.to(device)

    # Verify model output shape with a dummy batch
    dummy_batch = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
    with torch.no_grad():
        dummy_out = model(dummy_batch)
    assert dummy_out.shape == (
        2,
        len(Config.TARGET_COLS),
    ), "Model output shape mismatch!"
    print("Model forward pass check passed.")

    # --- 4. Training ---
    print("\n[Step 3] Starting Training Loop...")

    # Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # Train
    best_auc = train_model(
        model, train_loader, val_loader, optimizer, scheduler, device, Config.EPOCHS
    )

    print(f"Training completed. Best AUC: {best_auc}")
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Best model was not saved!"

    # --- 5. Inference / Submission ---
    print("\n[Step 4] Generating Submission...")

    # Prepare a small test set for the demo.
    # We disable DEBUG mode to prevent random shuffling in the Dataset class (which uses .sample()),
    # ensuring that the DataLoader order matches the CSV order.
    Config.DEBUG = False

    # Create a temporary metadata file with just 10 rows for speed
    full_test_df = pd.read_csv("./metadata/test.csv")
    temp_test_path = os.path.join(Config.WORKING_DIR, "temp_test_demo.csv")
    full_test_df.head(10).to_csv(temp_test_path, index=False)

    # Point Config to this temporary file
    Config.TEST_METADATA_PATH = temp_test_path

    # Load Test Dataset (uses the temp path)
    test_dataset = CatheterDataset(Config.TEST_METADATA_PATH, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load the best model weights
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Generate Submission
    generate_submission(model, test_loader, device)

    # Verify output
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created!"
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Submission generated with shape: {submission_df.shape}")

    # Assertions
    assert len(submission_df) == 10, "Submission row count mismatch!"
    expected_cols = ["StudyInstanceUID"] + Config.TARGET_COLS
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch!"

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
