import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.data import get_loaders, RNADataset
from library.model import ADSRN
from library.loss import AnchoredMCRMSELoss
from library.engine import Engine


def create_subset_data():
    """
    Creates a small subset of the training, validation, and test data
    to allow for rapid demonstration and testing.
    """
    print("Creating data subsets...")

    # Ensure demo directories exist
    demo_dir = "./working/demo_task"
    os.makedirs(demo_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Create subsets (ensure enough for at least one batch)
    # Batch size will be set to 4 for demo
    subset_size = 12

    mini_train = train_df.head(subset_size).copy()
    mini_val = val_df.head(subset_size).copy()
    mini_test = test_df.head(subset_size).copy()

    # Save subsets
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def configure_demo(train_path, val_path, test_path):
    """
    Overrides the global Config class to use demo paths and parameters.
    """
    print("Configuring demo parameters...")

    # Paths
    Config.WORKING_DIR = "./working/demo_task"
    Config.TRAIN_CSV = train_path
    Config.VAL_CSV = val_path
    Config.TEST_CSV = test_path

    # Cache files (use new names to force processing of subsets)
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "cache_train.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "cache_val.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "cache_test.npz")

    # Output
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 2
    Config.PATIENCE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)


def verify_components():
    """
    Verifies the logic of Data Loading, Model, Loss, and Metric components.
    """
    print("\n--- Verifying Components ---")

    seed_everything(Config.SEED)
    device = torch.device("cpu")  # Use CPU for simple verification logic

    # 1. Verify Data Loading
    print("1. Verifying Data Loading...")
    # Force reload from CSVs by setting load_cached_data=False initially or relying on new cache paths
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    batch = next(iter(train_loader))
    inputs, targets, p_idx = batch

    # Check Shapes
    # Inputs: (B, 107, 18)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_CHANNELS,
    ), f"Input shape mismatch: {inputs.shape}"
    # Targets: (B, 107, 5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Target shape mismatch: {targets.shape}"
    # Partner Indices: (B, 107)
    assert p_idx.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Partner indices shape mismatch: {p_idx.shape}"

    print("   Data Loading Verified.")

    # 2. Verify Model Architecture
    print("2. Verifying Model Architecture...")
    model = ADSRN().to(device)
    inputs = inputs.to(device)
    p_idx = p_idx.to(device)

    # Forward pass
    y_2, y_1 = model(inputs, p_idx)

    # Check Output Shapes: (B, 107, 5)
    assert y_2.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Model output y_2 shape mismatch: {y_2.shape}"
    assert y_1.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Model output y_1 shape mismatch: {y_1.shape}"

    print("   Model Architecture Verified.")

    # 3. Verify Loss Function
    print("3. Verifying Loss Function...")
    criterion = AnchoredMCRMSELoss()
    targets = targets.to(device)

    loss = criterion(y_2, targets)

    # Check Loss is scalar and has grad
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.requires_grad, "Loss should track gradients"
    assert loss.item() >= 0, "Loss should be non-negative"

    print(f"   Loss Function Verified (Value: {loss.item():.4f}).")

    # 4. Verify Metric
    print("4. Verifying Metric...")
    metric = MCRMSE()

    # Create synthetic perfect predictions
    metric.update(targets, targets)
    score = metric.compute()
    assert score == 0.0, "Perfect predictions should yield 0.0 error"

    metric.reset()
    # Create synthetic error (add 1.0 to everything)
    metric.update(targets + 1.0, targets)
    score = metric.compute()
    # RMSE of diff 1.0 is 1.0
    assert np.isclose(score, 1.0), f"Expected score 1.0, got {score}"

    print("   Metric Verified.")


def run_pipeline():
    """
    Runs the training and inference pipeline using the Engine class.
    """
    print("\n--- Running Training Pipeline ---")

    # 1. Train
    best_model_path = Engine.run_training()

    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"Training complete. Model saved to {best_model_path}")

    # 2. Inference
    print("\n--- Running Inference Pipeline ---")
    Engine.generate_submission(best_model_path)

    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not generated."

    # Verify Submission Format
    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {df_sub.shape}")

    # Expected rows: N_test_samples * 107
    # We used subset_size=12 for test
    expected_rows = 12 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    print("Pipeline execution successful.")


if __name__ == "__main__":
    # 1. Create subsets
    train_csv, val_csv, test_csv = create_subset_data()

    # 2. Configure Config
    configure_demo(train_csv, val_csv, test_csv)

    # 3. Verify Components
    verify_components()

    # 4. Run Pipeline
    run_pipeline()
