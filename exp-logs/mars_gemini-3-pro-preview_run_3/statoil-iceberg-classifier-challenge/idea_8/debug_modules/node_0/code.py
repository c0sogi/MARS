import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_train_val_loaders, get_test_loader
from library.model import ACResNet
from library.train import run_fold
from library.inference import create_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Overrides for Speed
    # We modify the Config class attributes directly to create a "Demo Mode"
    print("Configuring for fast demonstration...")
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_FILE = "./working/demo_submission.csv"
    Config.MAX_SAMPLES = 100  # Limit training data to 100 samples
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.N_FOLDS = 2  # Run only 2 folds instead of 5
    Config.BATCH_SIZE = 8  # Small batch size for the small dataset
    Config.PATIENCE = 2  # Short patience

    # Re-run setup to create the new working directories
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Validate Data Loading Logic
    print("\n=== Validating Data Loader ===")
    # We perform this check to ensure the data shapes and types are correct
    # load_cached_data=False forces the processing of raw JSON files
    train_loader, val_loader = get_train_val_loaders(
        fold_index=0, load_cached_data=False
    )

    # Fetch a single batch
    images, angles, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Angle Shape: {angles.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    # Expected: (Batch_Size, 3, 75, 75)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Incorrect image tensor shape"
    # Expected: (Batch_Size,)
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect angle tensor shape"
    # Expected: (Batch_Size,)
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"
    # Check data types
    assert images.dtype == torch.float32, "Images should be float32"
    assert angles.dtype == torch.float32, "Angles should be float32"

    print("Data Loader validation passed.")

    # 3. Validate Model Architecture
    print("\n=== Validating Model Architecture ===")
    device = torch.device("cpu")  # Use CPU for simple shape check
    model = ACResNet().to(device)

    # Create dummy inputs
    dummy_img = torch.randn(Config.BATCH_SIZE, 3, 75, 75).to(device)
    dummy_angle = torch.randn(Config.BATCH_SIZE).to(device)

    # Forward pass
    output = model(dummy_img, dummy_angle)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    # Expected output: (Batch_Size, 1) - raw logits
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    print("Model architecture validation passed.")

    # 4. Run Training Loop
    print("\n=== Running Training Loop (Demo) ===")
    # We run the first two folds to demonstrate the cross-validation loop
    for fold in range(Config.N_FOLDS):
        print(f"\n--- Fold {fold} ---")
        # We use load_cached_data=True now since get_train_val_loaders in step 2
        # would have cached the processed numpy arrays in the working dir.
        # However, since we changed WORKING_DIR to a new path, we need to regenerate
        # or just pass False. Passing False is safer for a standalone demo.
        run_fold(fold_index=fold, load_cached_data=False)

        # Verify checkpoint creation
        fold_dir = os.path.join(Config.WORKING_DIR, f"fold_{fold}")
        best_model_path = os.path.join(fold_dir, "model_best.pth")
        assert os.path.exists(best_model_path), f"Best model not saved for fold {fold}"
        print(f"Checkpoint verified for Fold {fold}")

    # 5. Run Inference and Submission Generation
    print("\n=== Running Inference and Submission ===")
    # This function loads the models trained in the previous step and generates a CSV
    create_submission(load_cached_data=False)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    # Assertions on Submission
    # The test set has 321 samples (based on metadata info provided in prompt)
    # Note: get_test_loader does NOT filter by MAX_SAMPLES, so it processes the full test set.
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "is_iceberg" in df_sub.columns, "Submission missing 'is_iceberg' column"
    assert len(df_sub) == 321, f"Expected 321 predictions, found {len(df_sub)}"

    # Check probabilities range
    probs = df_sub["is_iceberg"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of [0, 1] range"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
