import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_overall_lwlrap, mixup_data
from library.dataset import get_dataloaders, get_label_map
from library.model import ClassWiseEfficientNet
from library.trainer import Trainer, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Audio Tagging Demo ===")

    # 1. Setup and Configuration Override
    # We override Config parameters to run a fast demo instead of a full training session.
    set_seed(42)

    demo_dir = "./working/demo_run_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Setting up working directory: {demo_dir}")

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2

    # Re-initialize directories based on new paths
    Config.setup()

    # 2. Create Data Subsets
    # Instead of processing 20k files, we create temporary metadata files pointing to a tiny subset.
    print("Creating dataset subsets for rapid execution...")

    train_full = pd.read_csv(Config.TRAIN_META)
    val_full = pd.read_csv(Config.VAL_META)
    test_full = pd.read_csv(Config.TEST_META)

    # Select small subsets (ensure enough for at least one batch)
    train_sub = train_full.head(32)
    val_sub = val_full.head(16)
    test_sub = test_full.head(16)

    # Save subset metadata
    train_sub_path = os.path.join(demo_dir, "train_subset.csv")
    val_sub_path = os.path.join(demo_dir, "val_subset.csv")
    test_sub_path = os.path.join(demo_dir, "test_subset.csv")

    train_sub.to_csv(train_sub_path, index=False)
    val_sub.to_csv(val_sub_path, index=False)
    test_sub.to_csv(test_sub_path, index=False)

    # Point Config to these new metadata files
    Config.TRAIN_META = train_sub_path
    Config.VAL_META = val_sub_path
    Config.TEST_META = test_sub_path

    # 3. Logic Verification: LWLRAP Metric
    print("Verifying LWLRAP metric calculation...")
    # Test Case: Perfect prediction
    truth_perfect = np.array([[1, 0, 0], [0, 1, 1]])
    scores_perfect = np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.8]])
    score = calculate_overall_lwlrap(truth_perfect, scores_perfect)
    assert np.isclose(
        score, 1.0
    ), f"LWLRAP Verification Failed: Expected 1.0, got {score}"

    # Test Case: Known imperfect prediction
    # Sample 0: Truth=[1, 0], Scores=[0.2, 0.8] -> Rank 2 correct -> Prec=1/2
    # Sample 1: Truth=[0, 1], Scores=[0.9, 0.1] -> Rank 2 correct -> Prec=1/2
    # Overall = 0.5
    truth_imperfect = np.array([[1, 0], [0, 1]])
    scores_imperfect = np.array([[0.2, 0.8], [0.9, 0.1]])
    score = calculate_overall_lwlrap(truth_imperfect, scores_imperfect)
    assert np.isclose(
        score, 0.5
    ), f"LWLRAP Verification Failed: Expected 0.5, got {score}"
    print("LWLRAP metric verified.")

    # 4. Logic Verification: Model Architecture
    print("Verifying Model Architecture...")
    model = ClassWiseEfficientNet(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.eval()
    # Create dummy input: (Batch=2, Channel=1, Freq=128, Time=200)
    dummy_input = torch.randn(2, 1, 128, 200)
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch, Num_Classes)
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model Output Shape Mismatch: Expected {(2, Config.NUM_CLASSES)}, got {output.shape}"
    print("Model architecture verified.")

    # 5. Logic Verification: Mixup Augmentation
    print("Verifying Mixup Augmentation...")
    x = torch.randn(4, 1, 128, 100)
    y = torch.randn(4, 80)
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)

    assert mixed_x.shape == x.shape, "Mixup output shape mismatch"
    assert y_a.shape == y.shape, "Mixup target_a shape mismatch"
    # With alpha=1.0, lambda is likely not 0 or 1, so input should change
    if 0.0 < lam < 1.0:
        assert not torch.allclose(mixed_x, x), "Mixup failed to modify input"
    print("Mixup verified.")

    # 6. Pipeline Execution
    print("\n--- Executing Training Pipeline ---")

    # Step A: Data Loading (Spectrogram computation happens here)
    # load_cached_data=False forces processing of our new subsets
    print("Loading and processing data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    assert len(train_loader) > 0, "Train loader is empty!"

    # Step B: Model Training
    print("Initializing Trainer...")
    trainer = Trainer(train_loader, val_loader, test_loader)

    print("Starting training loop (2 epochs)...")
    trainer.fit()

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Checkpoint found at {best_model_path}")
    else:
        print(
            "Notice: No checkpoint saved (Validation score might not have improved in 2 epochs)."
        )

    # Step C: Inference
    print("Running inference on test subset...")
    fnames, preds = trainer.predict_test()

    # Verify predictions
    assert len(fnames) == len(
        test_sub
    ), f"Prediction count mismatch: Expected {len(test_sub)}, got {len(fnames)}"
    assert preds.shape == (
        len(test_sub),
        Config.NUM_CLASSES,
    ), f"Prediction shape mismatch: {preds.shape}"

    # Step D: Submission Generation
    print("Generating submission file...")
    generate_submission(fnames, preds)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    print(f"Submission successfully saved to {Config.SUBMISSION_PATH}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
