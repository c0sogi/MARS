import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_global_stats
from library.data import process_data, IcebergDataset
from library.model import DMWBNet
from library.train import run_kfold_training


def run_demo():
    print("=== Starting Demonstration of Iceberg Classifier Library ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # ---------------------------------------------------------
    print("[1] Configuring environment for demo...")

    # Override Config parameters to ensure fast execution
    Config.EPOCHS = 1  # Run only 1 epoch per fold
    Config.N_FOLDS = 2  # Run only 2 folds instead of 5
    Config.BATCH_SIZE = 16  # Small batch size
    Config.PATIENCE = 1  # Aggressive early stopping
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Redirect outputs to a demo directory
    Config.WORK_DIR = "./working/demo_run"
    Config.CACHE_FILE = os.path.join(Config.WORK_DIR, "cache", "processed_data.npz")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(os.path.dirname(Config.CACHE_FILE), exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("    Configuration updated for speed and isolation.")

    # ---------------------------------------------------------
    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("\n[2] Verifying library.utils.get_global_stats...")
    stats = get_global_stats(Config.TRAIN_JSON)

    # Assertions
    assert "band_1" in stats and "band_2" in stats
    assert "min" in stats["band_1"] and "max" in stats["band_1"]
    assert stats["band_1"]["min"] < stats["band_1"]["max"]
    print(
        f"    Global Stats computed successfully: Band 1 Min={stats['band_1']['min']:.2f}, Max={stats['band_1']['max']:.2f}"
    )

    # ---------------------------------------------------------
    # 3. Verify Data Processing
    # ---------------------------------------------------------
    print("\n[3] Verifying library.data.process_data...")
    # Force processing from scratch to test logic
    X, y, inc, X_test, inc_test, test_ids = process_data(load_cached_data=False)

    # Assertions
    print(f"    Training Data Shape: {X.shape}")
    print(f"    Test Data Shape: {X_test.shape}")

    # Check dimensions: (N, 3, 75, 75)
    assert X.ndim == 4
    assert X.shape[1] == 3  # Channels: Band1, Band2, Mean
    assert X.shape[2] == 75 and X.shape[3] == 75

    # Check Incidence Angles (should be float and no NaNs after processing)
    assert not np.isnan(inc).any(), "Found NaNs in training incidence angles"
    assert not np.isnan(inc_test).any(), "Found NaNs in test incidence angles"

    # Check Labels
    assert set(np.unique(y)).issubset({0.0, 1.0})
    print("    Data processing logic verified.")

    # ---------------------------------------------------------
    # 4. Verify Dataset Class
    # ---------------------------------------------------------
    print("\n[4] Verifying library.data.IcebergDataset...")
    # Create a small dataset instance
    ds = IcebergDataset(X[:10], inc[:10], y[:10], transform=True)

    # Test __getitem__
    img, angle, label = ds[0]

    # Assertions
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 75, 75)
    assert isinstance(angle, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    print("    Dataset __getitem__ verified.")

    # ---------------------------------------------------------
    # 5. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[5] Verifying library.model.DMWBNet...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMWBNet().to(device)

    # Create dummy input
    dummy_img = torch.randn(4, 3, 75, 75).to(device)
    dummy_inc = torch.randn(4, 1).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_img, dummy_inc)

    # Assertions
    assert output.shape == (4, 1), f"Expected output shape (4, 1), got {output.shape}"
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Output values out of probability range [0, 1]"
    print("    Model forward pass verified.")

    # ---------------------------------------------------------
    # 6. Verify Training Pipeline
    # ---------------------------------------------------------
    print("\n[6] Verifying library.train.run_kfold_training...")
    print("    Starting training loop (this may take a minute)...")

    # Run the training pipeline with reduced config
    run_kfold_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)

    # Check if artifacts were created
    fold_0_path = os.path.join(Config.WORK_DIR, "model_fold_0.pth")
    assert os.path.exists(fold_0_path), "Model checkpoint for fold 0 not found"
    print("    Training pipeline executed successfully.")

    # ---------------------------------------------------------
    # 7. Verify Submission
    # ---------------------------------------------------------
    print("\n[7] Verifying Submission File...")

    if os.path.exists(Config.SUBMISSION_FILE):
        df_sub = pd.read_csv(Config.SUBMISSION_FILE)
        print(f"    Submission loaded. Shape: {df_sub.shape}")

        # Assertions
        assert list(df_sub.columns) == ["id", "is_iceberg"]
        assert len(df_sub) == len(test_ids)
        assert df_sub["is_iceberg"].min() >= 0.0
        assert df_sub["is_iceberg"].max() <= 1.0

        # Check against sample submission format
        sample_sub = pd.read_csv(
            os.path.join(Config.INPUT_DIR, "sample_submission.csv")
        )
        assert len(df_sub) == len(sample_sub), "Submission length mismatch with sample"

        print("    Submission format verified.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
