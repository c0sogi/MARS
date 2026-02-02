import os
import sys
import pandas as pd
import torch
import warnings
import numpy as np

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data import get_data_loaders
from library.model import AsymmetricDCNResNet
from library.train import train_model


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set random seed for reproducibility
    print("Setting random seed...")
    seed_everything(42)

    # Define parameters for the demonstration
    DEBUG_SIZE = 2000  # Small subset for speed
    BATCH_SIZE = 64  # Appropriate batch size for small data
    EPOCHS = 2  # Minimal epochs to verify loop logic
    LEARNING_RATE = 1e-3

    print(f"\nConfiguration:")
    print(f"  Debug Sample Size: {DEBUG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")

    # ---------------------------------------------------------
    # 2. Data Pipeline Demonstration
    # ---------------------------------------------------------
    print("\n=== 1. Demonstrating Data Loading & Feature Engineering ===")

    # We call get_data_loaders with load_cached_data=False to force the
    # feature engineering pipeline to run on the raw parquet files.
    # This also generates the cache files in ./working/idea_32/
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=BATCH_SIZE, load_cached_data=False, debug_sample_size=DEBUG_SIZE
    )

    # Verify DataLoaders are not empty
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Validation loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"

    # Fetch a single batch to inspect structure
    X_batch, y_batch = next(iter(train_loader))

    print(f"  Batch X Shape: {X_batch.shape}")
    print(f"  Batch y Shape: {y_batch.shape}")

    # Assertions
    assert (
        X_batch.shape[0] == BATCH_SIZE
    ), f"Expected batch size {BATCH_SIZE}, got {X_batch.shape[0]}"
    assert X_batch.ndim == 2, "Input tensor should be 2D (Batch, Features)"
    assert y_batch.ndim == 1, "Target tensor should be 1D"

    # Determine input dimension dynamically
    input_dim = X_batch.shape[1]
    print(f"  Detected Input Dimension: {input_dim}")

    # ---------------------------------------------------------
    # 3. Model Architecture Demonstration
    # ---------------------------------------------------------
    print("\n=== 2. Demonstrating Model Architecture ===")

    # Instantiate the model
    # We use the config defaults for DCN layers and ResNet blocks
    model = AsymmetricDCNResNet(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        dcn_layers=Config.DCN_LAYERS,
        resnet_blocks=Config.RESNET_BLOCKS,
        hidden_dim=Config.HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # Move to CPU for this quick check (or GPU if available, handled by device placement later)
    model.eval()

    # Run a dummy forward pass
    with torch.no_grad():
        logits = model(X_batch)

    print(f"  Logits Shape: {logits.shape}")

    # Assertions
    # Output should be (Batch_Size, Num_Classes)
    assert logits.shape == (
        BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected output shape {(BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    # ---------------------------------------------------------
    # 4. Training Loop Integration
    # ---------------------------------------------------------
    print("\n=== 3. Demonstrating Full Training Pipeline ===")

    # train_model integrates loading, training, validation, and submission generation.
    # It will pick up the cached data generated in step 2 because we use the same debug_sample_size.
    trained_model = train_model(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        debug_sample_size=DEBUG_SIZE,
        create_submission=True,
    )

    assert isinstance(
        trained_model, torch.nn.Module
    ), "train_model did not return a model instance"

    # ---------------------------------------------------------
    # 5. Artifact Verification
    # ---------------------------------------------------------
    print("\n=== 4. Verifying Artifacts ===")

    # 5.1 Check Submission File
    sub_path = Config.SUBMISSION_PATH
    print(f"  Checking submission file at: {sub_path}")

    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Submission file not found at {sub_path}")

    df_sub = pd.read_csv(sub_path)
    print(f"  Submission Shape: {df_sub.shape}")
    print(f"  Submission Columns: {df_sub.columns.tolist()}")

    # Assertions
    assert list(df_sub.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], f"Submission columns mismatch. Expected {[Config.ID_COL, Config.TARGET_COL]}"

    # Since we used a debug sample size, the submission should reflect that size
    assert (
        len(df_sub) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} predictions, found {len(df_sub)}"

    # 5.2 Check Model Checkpoint
    ckpt_path = Config.MODEL_SAVE_PATH
    print(f"  Checking model checkpoint at: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path)
    assert "model_state_dict" in checkpoint, "Checkpoint missing model_state_dict"
    assert "best_acc" in checkpoint, "Checkpoint missing best_acc metric"

    print(f"  Best Validation Accuracy recorded: {checkpoint['best_acc']:.4f}")

    print("\nSUCCESS: All demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    main()
