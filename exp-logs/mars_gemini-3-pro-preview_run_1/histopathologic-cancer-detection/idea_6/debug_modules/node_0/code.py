import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.dataset import PathologyDataset, get_transforms, prepare_folds
from library.models import get_model
from library.engine import train_fold, predict_submission


def run_demo():
    print("Starting Library Usage Demonstration...")

    # ==========================================
    # 1. Configuration Setup for Speed
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config defaults to run a tiny, fast experiment
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 images
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.N_FOLDS = 2  # Setup for 2 folds (we will only run one)
    Config.WORK_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-run setup to create new directories
    Config.setup()

    # Set seed for reproducibility
    set_seed(42)
    print("Configuration updated: DEBUG=True, EPOCHS=1, SAMPLE_SIZE=50")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n[2] Verifying Data Loading and Processing...")

    # Load metadata manually to test Dataset class in isolation
    train_df = pd.read_csv(Config.TRAIN_METADATA).iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Initialize Dataset
    dataset = PathologyDataset(
        df=train_df, transform=get_transforms("train"), data_dir=Config.INPUT_DIR
    )

    # Test __getitem__
    img, label = dataset[0]

    # Assertions
    print(f"  Image Shape: {img.shape}")
    print(f"  Label: {label}")

    assert img.shape == (3, 48, 48), f"Expected (3, 48, 48), got {img.shape}"
    assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"
    assert label.shape == (1,), f"Expected label shape (1,), got {label.shape}"

    # Verify Folds Generation
    print("  Verifying fold generation...")
    folds_df = prepare_folds(load_cached_data=False)  # Force regeneration
    assert "fold" in folds_df.columns, "Folds dataframe missing 'fold' column"
    assert os.path.exists(
        os.path.join(Config.WORK_DIR, "folds.parquet")
    ), "Folds cache file not created"
    print("  Data loading verification passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    model_name = "densenet121"
    model = get_model(
        model_name, pretrained=False
    )  # No need to download weights for shape check
    model.eval()

    # Create dummy input batch
    dummy_input = torch.randn(Config.BATCH_SIZE, 3, 48, 48)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input Shape: {dummy_input.shape}")
    print(f"  Output Shape: {output.shape}")

    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 1), got {output.shape}"
    print("  Model architecture verification passed.")

    # ==========================================
    # 4. Training Engine Verification
    # ==========================================
    print("\n[4] Verifying Training Pipeline (Train Fold)...")

    # Train Fold 0 using densenet121
    # This tests: DataLoader creation, Optimizer, Loss, Train Loop, Val Loop, Checkpointing
    best_model_path, best_auc = train_fold(fold_idx=0, model_name=model_name)

    print(f"  Training completed. Best AUC: {best_auc}")
    print(f"  Checkpoint saved at: {best_model_path}")

    assert os.path.exists(best_model_path), "Best model checkpoint file was not saved."
    assert isinstance(best_auc, float), "AUC should be a float."
    print("  Training pipeline verification passed.")

    # ==========================================
    # 5. Inference Pipeline Verification
    # ==========================================
    print("\n[5] Verifying Inference Pipeline...")

    # Define the model config for inference (using the model we just trained)
    models_config = [(model_name, best_model_path)]

    # Run prediction
    # This tests: Test DataLoader, TTA, Ensemble Logic, Submission Generation
    predict_submission(models_config)

    print(f"  Submission saved to: {Config.SUBMISSION_PATH}")

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission shape: {sub_df.shape}")
    print(f"  Columns: {sub_df.columns.tolist()}")

    # Check rows count (should match DEBUG_SAMPLE_SIZE because we are in debug mode)
    # Note: get_test_dataloader uses Config.DEBUG_SAMPLE_SIZE when DEBUG is True
    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows in submission, got {len(sub_df)}"

    assert (
        "id" in sub_df.columns and "label" in sub_df.columns
    ), "Submission missing required columns 'id' and 'label'"

    # Check probability range
    assert (
        sub_df["label"].min() >= 0.0 and sub_df["label"].max() <= 1.0
    ), "Predictions are outside valid probability range [0, 1]"

    print("  Inference pipeline verification passed.")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    run_demo()
