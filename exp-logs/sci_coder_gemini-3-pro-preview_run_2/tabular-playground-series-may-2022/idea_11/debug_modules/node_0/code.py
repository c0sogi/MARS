import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import ManufacturingDataset
from library.model import FiLMResFunnel
from library.engine import Trainer, predict


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for a fast demo run
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    Config.WORKING_DIR = demo_dir
    Config.CACHE_PATH = os.path.join(demo_dir, "processed_data_demo.npz")
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")

    # Reduce epochs and patience for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 128  # Smaller batch for the small subset

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # Ensure reproducibility
    seed_everything(Config.RANDOM_SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[Step 2] Loading Datasets...")

    # Load full datasets (this processes/caches the data once)
    # We force load_cached_data=False initially to demonstrate processing logic if cache doesn't exist
    # In a real run, we'd leave it True.
    train_ds_full = ManufacturingDataset(split="train", load_cached_data=True)
    val_ds_full = ManufacturingDataset(split="val", load_cached_data=True)
    test_ds_full = ManufacturingDataset(split="test", load_cached_data=True)

    # Create Subsets for Speed (Simulate a quick epoch)
    # We'll use 2000 samples for train, 500 for val, 500 for test
    subset_indices_train = list(range(2000))
    subset_indices_val = list(range(500))
    subset_indices_test = list(range(500))

    train_ds = Subset(train_ds_full, subset_indices_train)
    val_ds = Subset(val_ds_full, subset_indices_val)
    test_ds = Subset(test_ds_full, subset_indices_test)

    print(
        f"Subset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 workers often faster for tiny datasets/debugging
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 3. Model & Verification
    # --------------------------------------------------------------------------
    print("\n[Step 3] Initializing Model and Verifying Architecture...")

    model = FiLMResFunnel().to(Config.DEVICE)

    # Verification: Check Forward Pass dimensions
    # Fetch a batch
    dummy_x_cont, dummy_x_cat, dummy_y = next(iter(train_loader))
    dummy_x_cont = dummy_x_cont.to(Config.DEVICE)
    dummy_x_cat = dummy_x_cat.to(Config.DEVICE)

    # Check Input Shapes
    assert (
        dummy_x_cont.shape[1] == Config.NUM_CONTINUOUS_FEATURES
    ), f"Expected {Config.NUM_CONTINUOUS_FEATURES} continuous features, got {dummy_x_cont.shape[1]}"
    assert (
        dummy_x_cat.shape[1] == Config.SEQUENCE_LENGTH
    ), f"Expected sequence length {Config.SEQUENCE_LENGTH}, got {dummy_x_cat.shape[1]}"

    # Run Forward Pass
    with torch.no_grad():
        output = model(dummy_x_cont, dummy_x_cat)

    # Check Output Shape
    assert output.shape == (
        dummy_x_cont.shape[0],
        1,
    ), f"Expected output shape {(dummy_x_cont.shape[0], 1)}, got {output.shape}"

    print(
        "Verification Successful: Model accepts inputs and produces correct output shape."
    )

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    print("\n[Step 4] Starting Training...")

    # Setup Training Components
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=Config.DEVICE,
        config=Config,
    )

    # Run Fit (1 Epoch as configured)
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS, patience=1)

    # Verify Checkpoint Creation
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")
    print(f"Checkpoint saved successfully at {Config.MODEL_PATH}")

    # --------------------------------------------------------------------------
    # 5. Inference & Submission
    # --------------------------------------------------------------------------
    print("\n[Step 5] Running Inference and Generating Submission...")

    # Load Best Model
    checkpoint = load_checkpoint(Config.MODEL_PATH, model, device=Config.DEVICE)
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with AUC {checkpoint['best_auc']:.4f}"
    )

    # Predict on Test Subset
    preds = predict(model, test_loader, Config.DEVICE)

    # Verify Predictions
    assert len(preds) == len(
        test_ds
    ), f"Prediction count {len(preds)} does not match test subset size {len(test_ds)}"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions contain values outside [0, 1] range (Sigmoid failure?)"

    # Create Submission DataFrame
    # Note: Since we used a subset, we need the corresponding IDs.
    # We load the test metadata to get IDs, then slice it to match our subset indices.
    test_meta = pd.read_csv(Config.TEST_META)
    subset_ids = test_meta.iloc[subset_indices_test]["id"].values

    submission_df = pd.DataFrame({"id": subset_ids, "target": preds})

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify File
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        print("First 5 rows:")
        print(submission_df.head())
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
