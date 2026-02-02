import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, calculate_lwlrap
from library.dataset import AudioDataset
from library.model import AudioClassifier
from library.engine import train_model, predict


def main():
    print("==================================================")
    print("   Audio Tagging Pipeline Demonstration Script    ")
    print("==================================================")

    # ---------------------------------------------------------
    # 1. Configuration Override for Fast Execution
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Modify Config for speed (Debug Mode)
    Config.debug = True
    Config.debug_sample_size = 50  # Use only 50 samples for demo
    Config.epochs = 1  # Train for only 1 epoch
    Config.batch_size = 8
    Config.num_workers = 0  # Disable multiprocessing for simple script execution
    Config.pretrained = False  # Skip downloading weights for speed

    # Ensure output directories exist
    Config.setup()

    # Set reproducibility seeds
    set_seed(Config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Batch Size: {Config.batch_size}")

    # ---------------------------------------------------------
    # 2. Verify Metric Logic (LWLRAP)
    # ---------------------------------------------------------
    print("\n[2] Verifying LWLRAP metric calculation...")

    # Test Case 1: Perfect Predictions
    y_true_perfect = np.array([[1, 0, 0], [0, 1, 0]])
    y_score_perfect = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
    score_perfect = calculate_lwlrap(y_true_perfect, y_score_perfect)
    assert np.isclose(score_perfect, 1.0), f"Expected 1.0, got {score_perfect}"
    print("    Perfect score test passed (1.0).")

    # Test Case 2: Known Imperfect Predictions
    # Sample 1 (True Class 0): Pred Rank 2 (Score 0.1 vs 0.9). Precision@2 = 1/2 = 0.5
    # Sample 2 (True Class 1): Pred Rank 1 (Score 0.9). Precision@1 = 1/1 = 1.0
    # Average = (0.5 + 1.0) / 2 = 0.75
    y_true_imperfect = np.array([[1, 0, 0], [0, 1, 0]])
    y_score_imperfect = np.array([[0.1, 0.9, 0.0], [0.1, 0.9, 0.0]])
    score_imperfect = calculate_lwlrap(y_true_imperfect, y_score_imperfect)
    assert np.isclose(score_imperfect, 0.75), f"Expected 0.75, got {score_imperfect}"
    print("    Imperfect score test passed (0.75).")

    # ---------------------------------------------------------
    # 3. Dataset Initialization & Verification
    # ---------------------------------------------------------
    print("\n[3] Loading Datasets...")

    train_dataset = AudioDataset(Config.train_metadata, mode="train")
    val_dataset = AudioDataset(Config.val_metadata, mode="val")
    test_dataset = AudioDataset(Config.test_metadata, mode="test")

    print(f"    Train Samples: {len(train_dataset)}")
    print(f"    Val Samples:   {len(val_dataset)}")

    # Verify Data Shapes
    spec, target, fname = train_dataset[0]
    print(f"    Sample Spectrogram Shape: {spec.shape}")
    print(f"    Sample Target Shape: {target.shape}")

    # Assertions for shape correctness
    # Spectrogram: (1, n_mels, time)
    assert spec.dim() == 3 and spec.shape[0] == 1, "Spectrogram must be (1, F, T)"
    assert spec.shape[1] == Config.n_mels, f"Expected {Config.n_mels} mel bins"
    assert (
        target.shape[0] == Config.num_classes
    ), f"Expected {Config.num_classes} classes"

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("\n[4] Initializing AudioClassifier...")

    model = AudioClassifier(
        num_classes=Config.num_classes, pretrained=Config.pretrained
    )
    model = model.to(device)

    # Verify Forward Pass
    dummy_batch = torch.randn(2, 1, Config.n_mels, spec.shape[2]).to(device)
    with torch.no_grad():
        dummy_out = model(dummy_batch)

    print(f"    Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (2, Config.num_classes), "Model output shape mismatch"

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("\n[5] Starting Training...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=Config.epochs,
        pct_start=Config.pct_start,
    )

    # Train model (returns model with best weights loaded)
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=Config.epochs,
    )

    # ---------------------------------------------------------
    # 6. Inference & Submission
    # ---------------------------------------------------------
    print("\n[6] Generating Predictions...")

    predict(trained_model, test_loader, device)

    # Verify Submission File
    if os.path.exists(Config.submission_path):
        sub_df = pd.read_csv(Config.submission_path)
        print(f"    Submission saved to: {Config.submission_path}")
        print(f"    Submission shape: {sub_df.shape}")

        # Verify submission format
        assert "fname" in sub_df.columns, "Submission missing 'fname' column"
        assert (
            len(sub_df.columns) == Config.num_classes + 1
        ), "Incorrect number of columns in submission"

        # In debug mode, we expect 'debug_sample_size' rows
        expected_rows = Config.debug_sample_size
        assert (
            len(sub_df) == expected_rows
        ), f"Expected {expected_rows} rows, found {len(sub_df)}"
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\n==================================================")
    print("       Pipeline Completed Successfully            ")
    print("==================================================")


if __name__ == "__main__":
    main()
