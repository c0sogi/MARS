import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
import shutil

# Import provided library components
from library.config import Config
from library.utils import seed_everything, calculate_auc, get_device
from library.data import (
    prepare_folds,
    get_dataloaders,
    get_test_dataloader,
    load_test_metadata,
)
from library.models import get_model
from library.training import run_fold_training
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_demo_data():
    """
    Creates a small subset of the metadata for demonstration purposes
    to ensure the script runs quickly.
    """
    print("\n[Demo] Creating data subsets...")

    # Create working directory for demo
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # Load original metadata
    train_full = pd.read_csv("./metadata/train.csv")
    val_full = pd.read_csv("./metadata/val.csv")
    test_full = pd.read_csv("./metadata/test.csv")

    # Sample subsets (enough for a few batches)
    # Ensure we have both classes in train/val for AUC calculation
    train_subset = (
        pd.concat(
            [
                train_full[train_full["label"] == 0].head(20),
                train_full[train_full["label"] == 1].head(20),
            ]
        )
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    val_subset = (
        pd.concat(
            [
                val_full[val_full["label"] == 0].head(10),
                val_full[val_full["label"] == 1].head(10),
            ]
        )
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    test_subset = test_full.head(20).reset_index(drop=True)

    # Save subsets
    train_path = os.path.join(demo_dir, "train_subset.csv")
    val_path = os.path.join(demo_dir, "val_subset.csv")
    test_path = os.path.join(demo_dir, "test_subset.csv")

    train_subset.to_csv(train_path, index=False)
    val_subset.to_csv(val_path, index=False)
    test_subset.to_csv(test_path, index=False)

    print(f"  Train subset: {len(train_subset)} samples")
    print(f"  Val subset:   {len(val_subset)} samples")
    print(f"  Test subset:  {len(test_subset)} samples")

    return demo_dir, train_path, val_path, test_path


def override_config(demo_dir, train_path, val_path, test_path):
    """
    Overrides Config attributes at runtime to use the demo data and settings.
    """
    print("\n[Demo] Overriding Configuration...")

    # Path Overrides
    Config.WORK_DIR = demo_dir
    Config.TRAIN_META_PATH = train_path
    Config.VAL_META_PATH = val_path
    Config.TEST_META_PATH = test_path
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Training Overrides for Speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_FOLDS = 2  # Minimum for CV logic
    Config.NUM_WORKERS = 2

    # Model Override (Use a lightweight model for demo)
    # 'resnet18' is standard in timm and much faster than convnext_tiny
    Config.MODEL_ARCHS = ["resnet18"]

    print(f"  Work Dir: {Config.WORK_DIR}")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Model: {Config.MODEL_ARCHS}")


def test_utils():
    print("\n[Test] Verifying Utility Functions...")

    # Test AUC Calculation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    auc = calculate_auc(y_true, y_pred)
    print(f"  Calculated AUC: {auc:.4f}")

    # Simple assertion for AUC logic
    assert 0.0 <= auc <= 1.0, "AUC must be between 0 and 1"

    # Test Device
    device = get_device()
    print(f"  Current Device: {device}")
    assert isinstance(device, torch.device)


def test_data_pipeline():
    print("\n[Test] Verifying Data Pipeline...")

    # 1. Prepare Folds
    # This should generate folds.parquet in the demo directory
    df_folds = prepare_folds(load_cached_data=False)

    assert "fold" in df_folds.columns, "Folds DataFrame missing 'fold' column"
    assert os.path.exists(
        os.path.join(Config.WORK_DIR, "folds.parquet")
    ), "folds.parquet not saved"
    print("  Folds prepared successfully.")

    # 2. Get DataLoaders for Fold 0
    train_loader, val_loader = get_dataloaders(fold_id=0, load_cached_data=True)

    # Check batch structure
    images, labels = next(iter(train_loader))

    print(f"  Batch Image Shape: {images.shape}")
    print(f"  Batch Label Shape: {labels.shape}")

    # Assertions
    expected_size = Config.CROP_SIZE  # 64
    assert (
        images.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Got {images.shape[0]}"
    assert (
        images.shape[2] == expected_size and images.shape[3] == expected_size
    ), f"Image dimensions mismatch. Expected {expected_size}x{expected_size}"
    assert labels.ndim == 1, "Labels should be 1D tensor"


def test_model_instantiation():
    print("\n[Test] Verifying Model Architecture...")

    model_name = Config.MODEL_ARCHS[0]
    model = get_model(model_name, pretrained=False)

    # Move to CPU for shape check
    model.eval()

    # Create dummy input: (Batch, 3, 64, 64)
    dummy_input = torch.randn(2, 3, Config.CROP_SIZE, Config.CROP_SIZE)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Model: {model_name}")
    print(f"  Input Shape: {dummy_input.shape}")
    print(f"  Output Shape: {output.shape}")

    # timm num_classes=1 typically outputs (B, 1)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"


def test_training_loop():
    print("\n[Test] Running Training Loop (Fold 0)...")

    # Run training for Fold 0 using the lightweight ResNet18
    # This uses the run_fold_training function from library.training
    run_fold_training(
        fold_id=0,
        model_name=Config.MODEL_ARCHS[0],
        num_epochs=Config.NUM_EPOCHS,
        patience=1,
        load_cached_data=True,
    )

    # Verify model artifact creation
    expected_model_path = os.path.join(
        Config.WORK_DIR, f"{Config.MODEL_ARCHS[0]}_fold_0.pth"
    )
    if os.path.exists(expected_model_path):
        print(f"  [Success] Model saved at {expected_model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {expected_model_path}")


def test_inference_pipeline():
    print("\n[Test] Running Inference Pipeline...")

    # Generate submission using the trained model from the previous step
    # This uses generate_submission from library.inference
    generate_submission(load_cached_data=False)

    # Verify submission file
    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"  Submission loaded. Shape: {df_sub.shape}")

    # Verify content
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)
    assert len(df_sub) == len(df_test_meta), "Submission length mismatch"
    assert (
        "id" in df_sub.columns and "label" in df_sub.columns
    ), "Missing columns in submission"
    assert df_sub["label"].dtype == float, "Label column should be float probabilities"

    print("  [Success] Submission generated successfully.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Prepare Environment
    demo_dir, train_path, val_path, test_path = create_demo_data()
    override_config(demo_dir, train_path, val_path, test_path)

    # 3. Validation Steps
    try:
        test_utils()
        test_data_pipeline()
        test_model_instantiation()
        test_training_loop()
        test_inference_pipeline()

        print("\nAll demonstration steps completed successfully!")

    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] Runtime Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
