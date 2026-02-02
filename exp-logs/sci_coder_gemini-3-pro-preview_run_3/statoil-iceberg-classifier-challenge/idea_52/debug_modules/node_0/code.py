import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_loader import process_and_cache_data, get_dataloaders
from library.model import MS_IDPH_CNN
from library.trainer import Trainer


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("1. Setting up Configuration...")

    # Override Config for a fast demo run
    Config.EXPERIMENT_NAME = "demo_usage"
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Use only 100 samples
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_FOLDS = 2  # Setup for 2 folds (we will run only fold 0)

    # Re-run setup to create directories based on new EXPERIMENT_NAME
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"   Experiment Name: {Config.EXPERIMENT_NAME}")
    print(f"   Cache Dir: {Config.CACHE_DIR}")
    print(f"   Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Processing
    # -------------------------------------------------------------------------
    print("\n2. Processing and Caching Data...")

    # This function loads JSONs, processes images to (N, 2, 75, 75), handles angles, and saves .npy files
    data = process_and_cache_data(load_cached_data=True)

    # Verify data dictionary structure
    required_keys = [
        "X_train",
        "y_train",
        "angle_train",
        "X_test",
        "angle_test",
        "ids_test",
    ]
    for key in required_keys:
        assert key in data, f"Missing key {key} in processed data."

    # Verify Shapes (based on DEBUG flag, these might be full size until get_dataloaders slices them,
    # but process_and_cache_data processes everything first, slicing happens in loader/trainer usually.
    # However, process_and_cache_data in library.data_loader processes the FULL dataset.)
    assert data["X_train"].ndim == 4, "X_train should be 4-dimensional (N, C, H, W)"
    assert (
        data["X_train"].shape[1] == 2
    ), "X_train should have 2 channels (HH, HV) initially"
    assert data["X_train"].shape[2:] == (75, 75), "Image size should be 75x75"

    print("   Data processing verified.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n3. Initializing DataLoaders (Fold 0)...")

    # Get dataloaders for the first fold
    # This handles the DEBUG slicing internally
    train_loader, val_loader = get_dataloaders(data, fold_idx=0)

    # Verify Batch Structure
    # Fetch one batch to check shapes and channel expansion (2 -> 4 channels)
    imgs, angs, labels = next(iter(train_loader))

    print(f"   Batch Image Shape: {imgs.shape}")
    print(f"   Batch Angle Shape: {angs.shape}")
    print(f"   Batch Label Shape: {labels.shape}")

    # Assertions
    # IcebergDataset expands 2 channels to 4: HH, HV, Avg, Ratio
    assert imgs.shape[1] == 4, f"Expected 4 input channels, got {imgs.shape[1]}"
    assert imgs.shape[2:] == (75, 75), "Image dimensions mismatch"
    assert angs.shape[1] == 1, "Angle should be (B, 1)"
    assert labels.shape[1] == 1, "Label should be (B, 1)"

    print("   DataLoaders verified.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\n4. Instantiating Model...")

    model = MS_IDPH_CNN().to(Config.DEVICE)

    # Verify Forward Pass
    imgs = imgs.to(Config.DEVICE)
    angs = angs.to(Config.DEVICE)

    with torch.no_grad():
        out = model(imgs, angs)

    print(f"   Model Output Shape: {out.shape}")
    assert out.shape == (imgs.shape[0], 1), "Model output shape mismatch"

    print("   Model architecture verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\n5. Starting Training Loop (Fold 0)...")

    trainer = Trainer(
        model=model,
        device=Config.DEVICE,
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=0,
    )

    # Run training
    best_val_loss = trainer.fit()

    print(f"   Training completed. Best Val Loss: {best_val_loss:.4f}")

    # Verify Checkpoints
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_best_fold_0.pth")
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file not found at {checkpoint_path}"
    print(f"   Checkpoint verified at: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n6. Running Inference on Test Set...")

    # Load best model
    best_model = MS_IDPH_CNN().to(Config.DEVICE)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    best_model.eval()

    # Prepare Test Data
    # In a real scenario, we would use a DataLoader for the test set.
    # For this demo, we'll manually process a small batch from the cached test data.
    X_test_all = data["X_test"]
    angle_test_all = data["angle_test"]
    ids_test_all = data["ids_test"]

    # Select a small subset for quick inference verification
    n_test_samples = 20
    X_test_sub = X_test_all[:n_test_samples]
    angle_test_sub = angle_test_all[:n_test_samples]
    ids_test_sub = ids_test_all[:n_test_samples]

    # Manually apply the 4-channel expansion logic from IcebergDataset
    # (Since we aren't using the Dataset class directly here for the raw numpy arrays)
    # Note: IcebergDataset logic: HH, HV, Avg, Ratio
    hh = X_test_sub[:, 0]
    hv = X_test_sub[:, 1]
    avg = (hh + hv) / 2.0
    ratio = hh - hv
    X_test_4ch = np.stack([hh, hv, avg, ratio], axis=1)  # (N, 4, 75, 75)

    # Convert to tensor
    test_imgs = torch.from_numpy(X_test_4ch).float().to(Config.DEVICE)
    test_angs = torch.from_numpy(angle_test_sub).float().view(-1, 1).to(Config.DEVICE)

    # Predict
    with torch.no_grad():
        logits = best_model(test_imgs, test_angs)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # Verify Predictions
    assert len(probs) == n_test_samples
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be between 0 and 1"
    print(f"   Inference successful on {n_test_samples} samples.")
    print(f"   Sample probabilities: {probs[:5]}")

    # Create Submission File
    # We will create a submission file for the FULL test set using dummy values
    # for the non-calculated parts to ensure file format correctness,
    # or just save the subset for demonstration.
    # The prompt asks for "Submission Format" compliance.
    # Let's generate a valid submission file with 0.5 for unprocessed, and actuals for processed.

    print("\n7. Generating Submission File...")
    full_probs = np.full(len(ids_test_all), 0.5)
    full_probs[:n_test_samples] = probs

    submission_df = pd.DataFrame({"id": ids_test_all, "is_iceberg": full_probs})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file was not created."
    print(f"   Submission saved to {submission_path}")
    print(f"   Submission head:\n{submission_df.head()}")


if __name__ == "__main__":
    run_demo()
