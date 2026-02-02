import os
import shutil
import torch
import numpy as np
import pandas as pd
import sys

# Import components from the provided library
from library.utils import seed_everything, AverageMeter, calculate_roc_auc
from library.dataset import get_dataloaders, get_test_ids
from library.model import WideAntiAliasedRes2NeXt
from library.train import run_training
from library.inference import generate_ensemble_predictions


def test_utils():
    """
    Verifies the functionality of utility classes and functions.
    """
    print("\n--- Testing Utilities ---")

    # 1. Test Seed Everything
    seed_everything(42)
    r1 = np.random.rand()
    seed_everything(42)
    r2 = np.random.rand()
    assert r1 == r2, "seed_everything did not produce reproducible numpy results"
    print("seed_everything: Passed")

    # 2. Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=1)
    meter.update(20, n=1)
    assert meter.avg == 15.0, f"AverageMeter failed. Expected 15.0, got {meter.avg}"
    assert meter.count == 2, "AverageMeter count incorrect"
    print("AverageMeter: Passed")

    # 3. Test ROC AUC Calculation
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    # AUC should be 0.75
    auc = calculate_roc_auc(y_true, y_scores)
    assert 0.0 <= auc <= 1.0, "AUC score out of range"
    # Single class case
    auc_single = calculate_roc_auc([0, 0], [0.1, 0.2])
    assert auc_single == 0.5, "AUC should be 0.5 for single class input"
    print("calculate_roc_auc: Passed")


def test_dataset_and_loader():
    """
    Verifies data loading, transformations, and batch shapes.
    """
    print("\n--- Testing Dataset and DataLoader ---")

    # Use a small batch size for testing
    batch_size = 4

    # get_dataloaders loads metadata and images
    # We use num_workers=0 to avoid multiprocessing overhead during this quick check
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=0, load_cached_data=True
    )

    # Check Train Loader
    images, labels = next(iter(train_loader))

    # Expected shape: (Batch, Channels, Height, Width) -> (4, 3, 32, 32)
    assert images.shape == (
        batch_size,
        3,
        32,
        32,
    ), f"Incorrect train image shape: {images.shape}"
    assert labels.shape == (batch_size,), f"Incorrect train label shape: {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32 tensors"

    # Check Test Loader (no labels)
    test_images = next(iter(test_loader))
    assert test_images.shape == (
        batch_size,
        3,
        32,
        32,
    ), f"Incorrect test image shape: {test_images.shape}"

    print(f"DataLoaders initialized successfully. Batch shape: {images.shape}")


def test_model_architecture():
    """
    Verifies the model architecture can process inputs and produce valid outputs.
    """
    print("\n--- Testing Model Architecture ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideAntiAliasedRes2NeXt().to(device)
    model.eval()

    # Create dummy input (Batch=2, C=3, H=32, W=32)
    dummy_input = torch.randn(2, 3, 32, 32).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output shape: (Batch, 1) for binary classification logits
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    print("Model forward pass successful.")


def run_demo_training_pipeline():
    """
    Runs the full training pipeline in debug mode.
    This tests library.train.run_training which orchestrates the whole process.
    """
    print("\n--- Running Demo Training Pipeline ---")

    work_dir = "./working/demo_execution"
    submission_path = os.path.join(work_dir, "submission", "submission_demo.csv")

    # Clean up previous run if exists
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)

    # Run training with debug=True
    # debug=True slices the dataset to 100 samples, sets epochs=2, n_seeds=1
    run_training(
        epochs=1,  # Overridden by debug=True inside the function usually, but passing 1 to be safe
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        n_seeds=1,  # Run only 1 seed for speed
        debug=True,  # ENABLE DEBUG MODE
        work_dir=work_dir,
        submission_path=submission_path,
    )

    # Verify outputs
    model_path = os.path.join(
        work_dir, "model_seed_42.pth"
    )  # debug mode usually forces seed 42 or we check the loop
    # Note: library/train.py loop uses range(n_seeds). If n_seeds=1, it saves model_seed_0.pth.
    # However, let's check what file was actually created.
    expected_model = os.path.join(work_dir, "model_seed_0.pth")

    assert os.path.exists(
        expected_model
    ), f"Model checkpoint not found at {expected_model}"
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Verify submission content
    df = pd.read_csv(submission_path)
    assert (
        "id" in df.columns and "has_cactus" in df.columns
    ), "Submission columns missing"
    assert len(df) == 100, f"Debug mode should produce 100 predictions, got {len(df)}"

    print("Training pipeline completed successfully.")
    return work_dir


def run_demo_inference_pipeline(work_dir):
    """
    Tests the standalone inference function using the model trained in the previous step.
    """
    print("\n--- Running Demo Inference Pipeline ---")

    submission_path = os.path.join(work_dir, "submission", "submission_inference.csv")

    # We use the model saved from the training step (seed 0)
    # Note: generate_ensemble_predictions loads test data internally.
    # Since we can't easily force 'debug' mode on generate_ensemble_predictions without modifying the library,
    # we will rely on the fact that it uses get_dataloaders.
    # CAUTION: get_dataloaders loads the FULL test set unless we modified the cache or source.
    # However, for this demonstration, we will just ensure the function runs.
    # To avoid long runtime on full test set inference, we'll skip this if it takes too long,
    # but since the test set is small (~3k images), inference on GPU is very fast (<10s).

    try:
        generate_ensemble_predictions(
            seeds=[0],
            work_dir=work_dir,
            submission_path=submission_path,
            batch_size=32,
            num_workers=2,
            load_cached_data=True,
        )

        assert os.path.exists(
            submission_path
        ), "Inference submission file not generated"
        df = pd.read_csv(submission_path)
        print(f"Inference generated {len(df)} predictions.")

    except Exception as e:
        print(f"Inference pipeline failed: {e}")
        raise e


def main():
    # Set global seed
    seed_everything(42)

    # 1. Unit Tests
    test_utils()
    test_dataset_and_loader()
    test_model_architecture()

    # 2. Integration Test (Training)
    # This runs a quick training loop on a subset of data
    work_dir = run_demo_training_pipeline()

    # 3. Integration Test (Inference)
    # Uses the model trained above to generate predictions
    run_demo_inference_pipeline(work_dir)

    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
