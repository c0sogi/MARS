import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import components from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model_factory import create_model
from library.trainer import train_model
from library.inference import run_inference

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo Speed
    # -------------------------------------------------------------------------
    print("Setting up demo configuration...")

    # Modify Config attributes to ensure the script runs quickly (within 1 hour)
    Config.SEED = 42
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 64  # Small subset for demonstration
    Config.BATCH_SIZE = 16

    # Limit to a single model backbone to save time
    Config.MODEL_BACKBONES = ["resnet50.a1_in1k"]

    # Redirect outputs to a demo directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set reproducibility
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying Data Loading ---")

    # Load dataloaders with debug settings
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force reload to ensure logic works
        debug=Config.DEBUG,
        debug_subset_size=Config.DEBUG_SUBSET_SIZE,
    )

    # Verify Train Loader
    try:
        batch = next(iter(train_loader))
        images, labels = batch
        print(f"Train Batch Shape: {images.shape}, Labels Shape: {labels.shape}")

        # Assertions
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), f"Image batch shape mismatch. Expected {(Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)}, got {images.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
        ), f"Label batch shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

        # Verify dataset size (approximate due to drop_last=True in train_loader)
        assert (
            len(train_loader.dataset) <= Config.DEBUG_SUBSET_SIZE
        ), "Train dataset size exceeds debug subset size."

        print("Data Loading verification passed.")
    except StopIteration:
        raise AssertionError("Train loader is empty.")

    # -------------------------------------------------------------------------
    # 3. Model Factory Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying Model Creation ---")
    model_name = Config.MODEL_BACKBONES[0]

    # Create model
    model = create_model(model_name, pretrained=True, num_classes=1)
    model.to(device)
    model.eval()

    # Test forward pass with dummy input
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("Model creation verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Running Training Demo ---")

    # Train the model (Config.EPOCHS is set to 1)
    best_model_path = train_model(model_name, train_loader, val_loader, patience=1)

    # Verify checkpoint existence
    assert os.path.exists(
        best_model_path
    ), f"Best model file not found at {best_model_path}"
    print(f"Training successful. Model saved to {best_model_path}")

    # -------------------------------------------------------------------------
    # 5. Inference Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Running Inference Demo ---")

    # Run inference (uses Config.MODEL_BACKBONES and Config.WORKING_DIR)
    run_inference(
        load_cached_data=False,
        debug=Config.DEBUG,
        debug_subset_size=Config.DEBUG_SUBSET_SIZE,
    )

    # -------------------------------------------------------------------------
    # 6. Submission Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying Submission ---")

    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print("Submission Head:")
        print(sub_df.head())

        # Structural Checks
        assert "id" in sub_df.columns, "Submission missing 'id' column"
        assert "label" in sub_df.columns, "Submission missing 'label' column"
        assert len(sub_df) > 0, "Submission file is empty"

        # Length Check: Should match the test set size (or subset size in debug)
        # Note: Test loader does not drop last, so length should be exactly the subset size
        # (or total test size if subset > total)
        metadata_test_len = len(pd.read_csv(Config.TEST_METADATA_PATH))
        expected_len = min(metadata_test_len, Config.DEBUG_SUBSET_SIZE)

        assert (
            len(sub_df) == expected_len
        ), f"Submission length mismatch. Expected {expected_len}, got {len(sub_df)}"

        # Value Checks
        assert sub_df["label"].min() >= 0.0, "Found probabilities < 0.0"
        assert sub_df["label"].max() <= 1.0, "Found probabilities > 1.0"
        assert (
            sub_df["id"].dtype == int or sub_df["id"].dtype == np.int64
        ), "ID column is not integer"

        print("Submission verification passed.")
    else:
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_PATH}"
        )

    print("\nAll demonstrations and verifications completed successfully.")
