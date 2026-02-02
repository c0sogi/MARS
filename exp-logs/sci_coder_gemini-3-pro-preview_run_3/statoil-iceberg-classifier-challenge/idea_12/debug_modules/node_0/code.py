import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, load_checkpoint, save_submission
from library.data_loader import process_and_cache_data, get_cv_loaders, get_test_loader
from library.model import IcebergSEResNet
from library.train_eval import fit_fold

if __name__ == "__main__":
    print("Initializing Demonstration...")

    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config class directly to limit the scope of the run
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = Config.WORKING_DIR  # Cache in the demo directory
    Config.MAX_SAMPLES = 100  # Limit to 100 samples for speed
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_FOLDS = 3  # Standard CV setup
    Config.PATIENCE = 2  # Short patience

    # Ensure the new working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(
        f"Configuration set: Max Samples={Config.MAX_SAMPLES}, Epochs={Config.NUM_EPOCHS}"
    )

    # 2. Data Processing and Caching
    print("\n--- Testing Data Processing ---")
    # Force reload to ensure we use the limited MAX_SAMPLES
    # We delete potential existing cache files in this demo dir to be safe,
    # though process_and_cache_data logic handles cache hits.
    # Here we just run it; if cache exists from previous demo run, it loads it.
    X_train, y_train, angles_train, X_test, ids_test, angles_test = (
        process_and_cache_data(load_cached_data=False)
    )

    # Validation: Check shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Assertions to verify data loading logic
    assert (
        len(X_train) == Config.MAX_SAMPLES
    ), f"Expected {Config.MAX_SAMPLES} training samples, got {len(X_train)}"
    assert X_train.shape[1:] == (
        3,
        75,
        75,
    ), f"Expected image shape (3, 75, 75), got {X_train.shape[1:]}"
    assert len(y_train) == len(X_train), "Mismatch between X_train and y_train length"
    assert (
        len(X_test) == Config.MAX_SAMPLES
    ), f"Expected {Config.MAX_SAMPLES} test samples, got {len(X_test)}"

    print("Data processing validation passed.")

    # 3. Data Loaders
    print("\n--- Testing Data Loaders ---")
    fold_idx = 0
    train_loader, val_loader = get_cv_loaders(fold_idx, X_train, y_train, angles_train)

    # Fetch one batch to verify
    images, angles, labels = next(iter(train_loader))

    print(f"Batch images shape: {images.shape}")
    print(f"Batch angles shape: {angles.shape}")
    print(f"Batch labels shape: {labels.shape}")

    # Assertions
    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75), "Incorrect batch image shape"
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect batch angle shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect batch label shape"

    print("Data loader validation passed.")

    # 4. Model Initialization and Forward Pass
    print("\n--- Testing Model Architecture ---")
    device = torch.device(Config.DEVICE)
    model = IcebergSEResNet().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    outputs = model(images, angles)

    print(f"Model output shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Model output should be (Batch_Size, 1)"
    # Check if output is not NaN
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"

    print("Model architecture validation passed.")

    # 5. Training Loop (Fit Fold)
    print("\n--- Testing Training Loop (Fold 0) ---")
    # This runs training and validation for the specified epochs
    best_loss = fit_fold(fold_idx, train_loader, val_loader)

    print(f"Training finished. Best Val Loss: {best_loss}")

    # Verify checkpoints exist
    checkpoint_path = os.path.join(
        Config.WORKING_DIR, f"checkpoint_fold_{fold_idx}.pth"
    )
    best_model_path = os.path.join(
        Config.WORKING_DIR, f"model_best_fold_{fold_idx}.pth"
    )

    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
    assert os.path.exists(best_model_path), "Best model file was not created"

    print("Training loop and checkpointing validation passed.")

    # 6. Inference and Submission
    print("\n--- Testing Inference and Submission ---")

    # Load the best model
    model_inference = IcebergSEResNet().to(device)
    start_epoch, best_score = load_checkpoint(best_model_path, model_inference)
    model_inference.eval()

    print(f"Loaded model from epoch {start_epoch} with score {best_score}")

    # Get test loader
    test_loader = get_test_loader(X_test, angles_test, ids_test)

    # Run Inference
    predictions = []
    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            # Forward pass
            logits = model_inference(images, angles)
            probs = torch.sigmoid(logits)
            predictions.extend(probs.cpu().numpy().flatten())

    predictions = np.array(predictions)

    # Assertions on predictions
    assert len(predictions) == len(
        ids_test
    ), "Number of predictions does not match number of test IDs"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions are not valid probabilities (0-1)"

    # Save Submission
    submission_file = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    save_submission(predictions, ids_test, filename=submission_file)

    # Verify Submission File
    assert os.path.exists(submission_file), "Submission file was not created"

    df_sub = pd.read_csv(submission_file)
    print("Submission file head:")
    print(df_sub.head())

    assert list(df_sub.columns) == [
        "id",
        "is_iceberg",
    ], "Submission columns are incorrect"
    assert (
        len(df_sub) == Config.MAX_SAMPLES
    ), f"Submission length mismatch. Expected {Config.MAX_SAMPLES}, got {len(df_sub)}"

    print("Inference and submission validation passed.")
    print("\nDemonstration completed successfully.")
