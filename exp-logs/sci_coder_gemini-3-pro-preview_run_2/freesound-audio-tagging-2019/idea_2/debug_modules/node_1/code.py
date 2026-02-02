import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, AverageMeter, calculate_lwlrap
from library.dataset import get_dataloaders, mixup_data
from library.model import AudioClassifier
from library.trainer import Trainer


def run_demo():
    print("==== Starting Library Demonstration ====")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Set paths to working directory to avoid messing with real submission paths
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    Config.WORKING_DIR = demo_working_dir
    Config.SAVE_DIR = demo_working_dir
    Config.SUBMISSION_PATH = os.path.join(demo_working_dir, "submission.csv")

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use 50 samples for train/val/test

    # Reduce training parameters for speed
    Config.BATCH_SIZE = 8
    Config.MAX_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.PRETRAINED = False  # Disable downloading weights for speed/offline safety

    # Ensure reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, Epochs=1, Batch=8")

    # ---------------------------------------------------------
    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Utility Functions...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(val=10, n=2)
    meter.update(val=20, n=2)
    # Total sum = 10*2 + 20*2 = 60, Total count = 4, Avg = 15
    assert meter.avg == 15.0, f"AverageMeter failed. Expected 15.0, got {meter.avg}"
    print("AverageMeter: OK")

    # Test LWLRAP Metric
    # Case: 2 samples, 3 classes.
    # Sample 1: True=[1, 0, 0], Pred=[0.9, 0.1, 0.1] -> Rank 1 correct -> Precision 1.0
    # Sample 2: True=[0, 1, 1], Pred=[0.1, 0.8, 0.9] -> Rank 1 (0.9) correct, Rank 2 (0.8) correct.
    y_true = np.array([[1, 0, 0], [0, 1, 1]])
    y_score = np.array([[0.9, 0.1, 0.1], [0.1, 0.8, 0.9]])

    score = calculate_lwlrap(y_true, y_score)
    # Expected: Perfect ranking implies score should be 1.0
    assert np.isclose(
        score, 1.0
    ), f"LWLRAP calculation failed. Expected 1.0, got {score}"
    print("calculate_lwlrap: OK")

    # ---------------------------------------------------------
    # 3. Verify Data Pipeline
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying Data Pipeline...")

    train_loader, val_loader, test_loader = get_dataloaders()

    # Check loader lengths
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch to verify shapes and collate_fn
    specs, labels, fnames = next(iter(train_loader))

    print(f"Batch Spectrogram Shape: {specs.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    # Assertions
    # Shape: (Batch, Channels, Freq, Time)
    assert specs.ndim == 4, "Spectrogram should be 4D (B, C, F, T)"
    assert (
        specs.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}"
    assert specs.shape[1] == 1, "Expected 1 channel (mono)"
    assert specs.shape[2] == Config.N_MELS, f"Expected {Config.N_MELS} mel bins"
    # Labels: (Batch, Num_Classes)
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Label shape mismatch"

    # Verify Mixup
    mixed_x, y_a, y_b, lam = mixup_data(specs, labels, alpha=1.0, use_cuda=False)
    assert mixed_x.shape == specs.shape, "Mixup output shape mismatch"
    print("Data Loading & Processing: OK")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[Step 4] Verifying Model Architecture...")

    model = AudioClassifier(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.eval()

    # Create dummy input: (Batch, 1, Freq=128, Time=313) ~10 seconds
    dummy_input = torch.randn(2, 1, Config.N_MELS, 313)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"
    print("Model Architecture: OK")

    # ---------------------------------------------------------
    # 5. Verify Training Loop (Trainer)
    # ---------------------------------------------------------
    print("\n[Step 5] Verifying Training Loop...")

    trainer = Trainer(train_loader, val_loader, test_loader)

    # Run fit (1 epoch as configured)
    trainer.fit()

    # Check if model checkpoint was saved
    expected_model_path = os.path.join(Config.SAVE_DIR, "best_model.pth")
    if os.path.exists(expected_model_path):
        print(f"Checkpoint saved successfully at {expected_model_path}")
    else:
        # If validation didn't improve (unlikely with random init but possible),
        # the trainer might not save 'best_model.pth'.
        # However, for this demo, we just want to ensure code ran without error.
        print(
            "Training finished (no checkpoint saved, likely due to random init/short run)."
        )

    # ---------------------------------------------------------
    # 6. Verify Inference
    # ---------------------------------------------------------
    print("\n[Step 6] Verifying Inference...")

    # Force save a dummy model if one wasn't saved, to allow predict to run
    if not os.path.exists(expected_model_path):
        torch.save(trainer.model.state_dict(), expected_model_path)
        trainer.best_model_path = expected_model_path

    trainer.predict()

    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated at {Config.SUBMISSION_PATH}")
        print(f"Submission Shape: {sub_df.shape}")

        # Verify columns
        # fname + 80 classes = 81 columns
        assert (
            len(sub_df.columns) == Config.NUM_CLASSES + 1
        ), "Incorrect number of columns in submission"
        assert "fname" in sub_df.columns, "fname column missing"

        # Verify rows (should match debug subset size or total test size depending on loader logic)
        # Since we set DEBUG=True, dataset length is capped.
        expected_rows = min(len(pd.read_csv(Config.TEST_CSV)), Config.DEBUG_SUBSET_SIZE)
        # Note: DataLoader drop_last=False for test, so we get all samples
        assert (
            len(sub_df) == expected_rows
        ), f"Expected {expected_rows} rows, got {len(sub_df)}"
        print("Inference: OK")
    else:
        raise AssertionError("Submission file was not created.")

    print("\n==== Demonstration Complete: All checks passed ====")


if __name__ == "__main__":
    run_demo()
