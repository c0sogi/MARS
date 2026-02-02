import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import provided library modules
from library.config import Config
from library.utils import set_seed, MCRMSEMetric
from library.loss import MaskedMCRMSELoss
from library.data import get_loaders
from library.model import RNANet
from library.train import train_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demo by creating subsets of data
    and overriding Config parameters to point to them.
    """
    print(">>> Setting up demo environment...")

    # Define paths
    demo_work_dir = "./working/demo_execution"
    demo_meta_dir = "./working/demo_metadata"

    os.makedirs(demo_work_dir, exist_ok=True)
    os.makedirs(demo_meta_dir, exist_ok=True)

    # 1. Create Data Subsets (Top 21 rows to ensure at least a few batches)
    # We read from the original metadata and save small versions to working dir
    subset_size = 21

    # Train
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_train_sub = df_train.head(subset_size)
    train_sub_path = os.path.join(demo_meta_dir, "train_subset.csv")
    df_train_sub.to_csv(train_sub_path, index=False)

    # Val
    df_val = pd.read_csv(Config.VAL_CSV)
    df_val_sub = df_val.head(subset_size)
    val_sub_path = os.path.join(demo_meta_dir, "val_subset.csv")
    df_val_sub.to_csv(val_sub_path, index=False)

    # Test
    df_test = pd.read_csv(Config.TEST_CSV)
    df_test_sub = df_test.head(subset_size)
    test_sub_path = os.path.join(demo_meta_dir, "test_subset.csv")
    df_test_sub.to_csv(test_sub_path, index=False)

    print(f"Created data subsets with {subset_size} samples each.")

    # 2. Override Config
    # We modify the static attributes of the Config class directly.
    Config.WORK_DIR = demo_work_dir
    Config.TRAIN_CSV = train_sub_path
    Config.VAL_CSV = val_sub_path
    Config.TEST_CSV = test_sub_path

    # Update cache paths to avoid conflicts with real training
    Config.TRAIN_CACHE = os.path.join(demo_work_dir, "train_demo.npz")
    Config.VAL_CACHE = os.path.join(demo_work_dir, "val_demo.npz")
    Config.TEST_CACHE = os.path.join(demo_work_dir, "test_demo.npz")

    Config.MODEL_PATH = os.path.join(demo_work_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_work_dir, "submission.csv")

    # Reduce training parameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.PATIENCE = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    Config.print_config()


def verify_data_pipeline():
    """
    Verifies the data loading pipeline and the shape of the tensors.
    """
    print("\n>>> Verifying Data Pipeline...")

    # Force reload to ensure we use the subsets
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Fetch one batch
    inputs, partner_indices, partner_mask, targets = next(iter(train_loader))

    print(f"Input Batch Shape: {inputs.shape}")
    print(f"Targets Batch Shape: {targets.shape}")

    # Assertions
    # Inputs: (Batch, Seq_Len, Input_Dim=19)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)}, got {inputs.shape}"

    # Targets: (Batch, Seq_Len, Num_Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)}, got {targets.shape}"

    # Partner Mask: (Batch, Seq_Len)
    assert partner_mask.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Partner mask shape mismatch"

    print("Data Pipeline verification passed.")
    return inputs, partner_indices, partner_mask, targets


def verify_model_and_loss(inputs, partner_indices, partner_mask, targets):
    """
    Verifies the model forward pass and loss calculation.
    """
    print("\n>>> Verifying Model and Loss...")

    device = torch.device("cpu")  # Use CPU for simple verification
    model = RNANet().to(device)
    criterion = MaskedMCRMSELoss().to(device)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        preds = model(inputs, partner_indices, partner_mask)

    print(f"Prediction Shape: {preds.shape}")
    assert preds.shape == targets.shape, "Prediction and Target shapes must match."

    # Loss Calculation
    loss = criterion(preds, targets)
    print(f"Computed Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("Model and Loss verification passed.")


def verify_metric_logic():
    """
    Manually verifies the MCRMSEMetric logic with synthetic data.
    """
    print("\n>>> Verifying MCRMSEMetric Logic...")

    metric = MCRMSEMetric()

    # Create synthetic data
    # Config.SCORED_COLS are ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Config.TARGET_COLS has 5 columns. Indices 0, 1, 3 are scored.

    batch_size = 2
    seq_len = 107
    num_targets = 5

    # Case: Preds = 0, Targets = 1.
    # Error = (0-1)^2 = 1. RMSE = 1. MCRMSE = 1.
    preds = np.zeros((batch_size, seq_len, num_targets))
    targets = np.ones((batch_size, seq_len, num_targets))

    metric.update(preds, targets)
    score = metric.compute()

    print(f"Synthetic Score (Expected ~1.0): {score}")

    # Allow small float error
    assert abs(score - 1.0) < 1e-5, f"Metric logic failed. Expected 1.0, got {score}"

    print("Metric logic verification passed.")


def run_full_training_demo():
    """
    Runs the end-to-end training loop using the library's train_model function.
    """
    print("\n>>> Running End-to-End Training Demo...")

    # This will use the overridden Config parameters
    train_model()

    # Verify outputs
    if os.path.exists(Config.MODEL_PATH):
        print(f"Success: Model saved at {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created.")

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Success: Submission file saved at {Config.SUBMISSION_PATH}")
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission shape: {df_sub.shape}")
        # 21 samples * 107 seq_len = 2247 rows
        expected_rows = 21 * Config.SEQ_LEN
        assert (
            len(df_sub) == expected_rows
        ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Setup
    setup_demo_environment()

    # 2. Verify Components
    inputs, p_idx, p_mask, targets = verify_data_pipeline()
    verify_model_and_loss(inputs, p_idx, p_mask, targets)
    verify_metric_logic()

    # 3. Run Training
    run_full_training_demo()

    print("\n>>> Demo Completed Successfully.")
