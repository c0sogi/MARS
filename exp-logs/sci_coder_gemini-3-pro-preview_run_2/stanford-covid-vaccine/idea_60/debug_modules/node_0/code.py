import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.data import RNAProcessor, get_loaders
from library.model import AHIRN
from library.loss import MCRMSELoss
from library.engine import run_engine


def setup_demo_config():
    """
    Overrides Config parameters for a quick demo run.
    This ensures the pipeline runs on a small subset of data with reduced model size.
    """
    print("Setting up demo configuration...")

    # Define working directories
    Config.WORKING_DIR = "./working"
    Config.IDEA_DIR = os.path.join(Config.WORKING_DIR, "demo_idea")
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Point to mini datasets (created later)
    Config.TRAIN_METADATA_PATH = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    Config.VAL_METADATA_PATH = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    Config.TEST_METADATA_PATH = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Define cache paths to avoid conflicts with full experiments
    Config.TRAIN_CACHE_PATH = os.path.join(Config.IDEA_DIR, "mini_train_cache.npz")
    Config.VAL_CACHE_PATH = os.path.join(Config.IDEA_DIR, "mini_val_cache.npz")
    Config.TEST_CACHE_PATH = os.path.join(Config.IDEA_DIR, "mini_test_cache.npz")

    # Model and submission outputs
    Config.MODEL_SAVE_PATH = os.path.join(Config.IDEA_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Reduce hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.HIDDEN_DIM = 32  # Smaller model size
    Config.LATENT_DIM = 32
    Config.FEEDBACK_EMBED_DIM = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Ensure reproducibility
    Config.SEED = 123
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)


def create_mini_datasets():
    """
    Creates small subsets of the original metadata files for demonstration purposes.
    Reads from ./metadata and writes to ./working.
    """
    print("Creating mini datasets...")

    # Paths to original metadata
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    # Sample sizes
    n_train = 32
    n_val = 16
    n_test = 16

    # Create mini train set
    if os.path.exists(orig_train_path):
        df_train = pd.read_csv(orig_train_path)
        df_train.head(n_train).to_csv(Config.TRAIN_METADATA_PATH, index=False)
    else:
        raise FileNotFoundError(
            f"Original train metadata not found at {orig_train_path}"
        )

    # Create mini validation set
    if os.path.exists(orig_val_path):
        df_val = pd.read_csv(orig_val_path)
        df_val.head(n_val).to_csv(Config.VAL_METADATA_PATH, index=False)
    else:
        raise FileNotFoundError(f"Original val metadata not found at {orig_val_path}")

    # Create mini test set
    if os.path.exists(orig_test_path):
        df_test = pd.read_csv(orig_test_path)
        df_test.head(n_test).to_csv(Config.TEST_METADATA_PATH, index=False)
    else:
        raise FileNotFoundError(f"Original test metadata not found at {orig_test_path}")

    print(f"Mini datasets created at {Config.WORKING_DIR}")


def verify_data_processing():
    """
    Verifies that RNAProcessor correctly processes data, produces correct shapes,
    and handles caching.
    """
    print("Verifying data processing...")
    processor = RNAProcessor()

    # Ensure we start fresh by removing any existing cache for this test
    if os.path.exists(Config.TRAIN_CACHE_PATH):
        os.remove(Config.TRAIN_CACHE_PATH)

    # Process mini train dataset
    inputs, partners, targets, ids = processor.process(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_CACHE_PATH,
        mode="train",
        load_cached_data=False,
    )

    # --- Assertions ---
    # 1. Check Input Shapes
    # Expected channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (Partner) = 18
    expected_channels = 18
    assert inputs.ndim == 3, f"Inputs should be 3D, got {inputs.ndim}"
    assert inputs.shape[1] == Config.SEQ_LENGTH, f"Seq len mismatch: {inputs.shape[1]}"
    assert (
        inputs.shape[2] == expected_channels
    ), f"Channel mismatch: {inputs.shape[2]} vs {expected_channels}"

    # 2. Check Partner Indices Shape
    assert partners.shape == (
        inputs.shape[0],
        Config.SEQ_LENGTH,
    ), "Partner indices shape mismatch"

    # 3. Check Targets Shape
    # Expected targets: 5 (reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C)
    assert targets.shape == (
        inputs.shape[0],
        Config.SEQ_LENGTH,
        5,
    ), "Targets shape mismatch"

    # 4. Check IDs
    assert len(ids) == inputs.shape[0], "IDs length mismatch"

    print("Data processing verification passed.")


def verify_model_logic():
    """
    Verifies the AHIRN model architecture, forward pass, and loss calculation.
    """
    print("Verifying model logic...")

    device = torch.device("cpu")  # Use CPU for simple logic verification
    model = AHIRN().to(device)
    model.eval()

    # Create dummy batch
    B, L = 2, Config.SEQ_LENGTH
    in_channels = 18

    # Random input tensor
    dummy_input = torch.randn(B, L, in_channels).to(device)
    # Random partner indices (valid range -1 to L-1)
    dummy_partners = torch.randint(-1, L, (B, L)).to(device)

    # Run Forward Pass
    with torch.no_grad():
        y1, y2 = model(dummy_input, dummy_partners)

    # --- Assertions ---
    # 1. Check Output Shapes
    assert y1.shape == (B, L, 5), f"Output y1 shape mismatch: {y1.shape}"
    assert y2.shape == (B, L, 5), f"Output y2 shape mismatch: {y2.shape}"

    # 2. Check Loss Calculation
    criterion = MCRMSELoss()
    dummy_targets = torch.randn(B, L, 5).to(device)

    loss = criterion(y2, dummy_targets)

    # Loss should be a scalar and non-negative
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("Model logic verification passed.")


def run_full_pipeline():
    """
    Runs the full training and inference pipeline using the Engine class.
    This simulates the actual competition workflow.
    """
    print("\nRunning full training pipeline (Engine)...")

    # run_engine() initializes the Engine, which uses Config parameters.
    # Since we updated Config in setup_demo_config(), it will use the mini datasets.
    run_engine()

    # --- Verify Submission ---
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with shape: {df_sub.shape}")

    # Expected rows: n_test_samples * seq_length
    # We used 16 test samples in create_mini_datasets
    expected_rows = 16 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.ALL_TARGETS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    print("Pipeline execution successful.")


if __name__ == "__main__":
    # 1. Setup Configuration
    setup_demo_config()

    # 2. Prepare Data
    create_mini_datasets()

    # 3. Verify Components
    verify_data_processing()
    verify_model_logic()

    # 4. Run End-to-End Pipeline
    run_full_pipeline()
