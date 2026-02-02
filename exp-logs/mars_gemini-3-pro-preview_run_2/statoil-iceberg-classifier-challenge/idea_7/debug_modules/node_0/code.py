import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library files
from library.config import Config, set_deterministic
from library.utils import seed_everything
from library.dataset import process_data, get_fold_loaders, get_test_loader
from library.model import SEAHN
from library.engine import Trainer, predict


def run_demo():
    print("Initializing Demo Script...")

    # 1. Setup and Configuration Override for Speed
    # We modify the Config class attributes directly to limit the runtime.
    print("Configuring environment for rapid demonstration...")
    Config.MAX_SAMPLES = 64  # Use only 64 samples for training/validation
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_FOLDS = 5  # Keep default, but we only run one fold
    Config.WORKING_DIR = "./working/demo_execution"  # Separate dir for demo
    Config.PROCESSED_DATA_CACHE = os.path.join(Config.WORKING_DIR, "processed_data.npz")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # 2. Data Processing Verification
    print("\n--- Step 1: Data Processing ---")
    # Force processing from scratch to verify logic (load_cached_data=False)
    # Note: process_data loads the FULL dataset first, then get_fold_loaders slices it based on MAX_SAMPLES
    X_train_img, X_train_stats, y_train, X_test_img, X_test_stats, test_ids = (
        process_data(load_cached_data=False)
    )

    # Verify Data Shapes (Full Dataset)
    print(f"Full Train Image Shape: {X_train_img.shape}")
    print(f"Full Train Stats Shape: {X_train_stats.shape}")

    # Assertions to check data integrity
    assert (
        len(X_train_img) == len(X_train_stats) == len(y_train)
    ), "Mismatch in training data lengths"
    assert X_train_img.shape[1:] == (
        75,
        75,
        3,
    ), f"Unexpected image shape: {X_train_img.shape[1:]}"
    assert (
        X_train_stats.shape[1] == Config.NUM_STAT_FEATURES
    ), f"Unexpected stats features: {X_train_stats.shape[1]}"
    assert not np.isnan(X_train_stats).any(), "NaN values found in statistical features"

    print("Data processing logic verified.")

    # 3. Data Loading Verification
    print("\n--- Step 2: Data Loading (Fold 0) ---")
    # Get loaders for the first fold. This triggers the MAX_SAMPLES slicing.
    train_loader, val_loader = get_fold_loaders(fold_idx=0, load_cached_data=True)

    # Fetch one batch to verify tensor shapes
    sample_imgs, sample_stats, sample_labels = next(iter(train_loader))

    print(f"Batch Image Tensor Shape: {sample_imgs.shape}")
    print(f"Batch Stats Tensor Shape: {sample_stats.shape}")
    print(f"Batch Labels Shape: {sample_labels.shape}")

    # Assertions for Loader
    assert sample_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Incorrect batch image shape"
    assert sample_stats.shape == (
        Config.BATCH_SIZE,
        Config.NUM_STAT_FEATURES,
    ), "Incorrect batch stats shape"
    assert sample_labels.shape == (Config.BATCH_SIZE,), "Incorrect batch label shape"

    print("Data loader verification successful.")

    # 4. Model Initialization
    print("\n--- Step 3: Model Initialization ---")
    device = torch.device(Config.DEVICE)
    model = SEAHN().to(device)

    # Verify Forward Pass
    # Move sample batch to device
    sample_imgs = sample_imgs.to(device)
    sample_stats = sample_stats.to(device)

    with torch.no_grad():
        output = model(sample_imgs, sample_stats)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    print("Model initialized and forward pass verified.")

    # 5. Training Loop
    print("\n--- Step 4: Training Loop ---")
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(),
    )

    # Run Training
    # We expect this to run quickly due to MAX_SAMPLES=64 and NUM_EPOCHS=2
    best_loss = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
    )

    print(f"Training finished. Best Validation Loss: {best_loss:.4f}")
    assert isinstance(best_loss, float), "Trainer did not return a float loss"
    assert best_loss < float("inf"), "Training loss is infinite"

    # 6. Inference and Submission
    print("\n--- Step 5: Inference and Submission ---")
    test_loader, test_ids_loader = get_test_loader(load_cached_data=True)

    # Verify test IDs match
    assert np.array_equal(
        test_ids, test_ids_loader
    ), "Test IDs mismatch between processing and loader"

    # Generate Predictions
    preds = predict(model, test_loader, device)

    print(f"Predictions Shape: {preds.shape}")
    assert len(preds) == len(
        test_ids
    ), "Number of predictions does not match number of test IDs"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions contain values outside [0, 1]"

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": preds})

    # Save Submission
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Head of submission:")
    print(submission_df.head())

    # Final check
    assert os.path.exists(submission_path), "Submission file was not created"

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
