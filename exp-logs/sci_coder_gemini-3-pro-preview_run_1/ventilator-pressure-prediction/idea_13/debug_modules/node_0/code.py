import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.model import VentilatorModel
from library.data_utils import prepare_datasets, VentilatorDataset
from library.train_utils import train_model, predict_and_submit


def setup_demo_environment(demo_dir):
    """Creates a clean demo directory."""
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)


def create_subset_csv(source_path, dest_path, num_breaths=50):
    """
    Reads a subset of breaths from the source CSV and saves it to dest_path.
    Ensures that full breaths (80 steps) are preserved.
    """
    # Each breath is 80 steps. Read a buffer to ensure we get enough.
    # We need num_breaths * 80 rows.
    rows_to_read = (num_breaths + 5) * 80

    df = pd.read_csv(source_path, nrows=rows_to_read)

    # Get the first num_breaths unique breath_ids
    breath_ids = df["breath_id"].unique()[:num_breaths]

    # Filter
    subset_df = df[df["breath_id"].isin(breath_ids)].copy()

    # Verify shape
    expected_len = num_breaths * 80
    assert (
        len(subset_df) == expected_len
    ), f"Subset length {len(subset_df)} does not match expected {expected_len}"

    subset_df.to_csv(dest_path, index=False)
    print(
        f"Created subset {os.path.basename(dest_path)}: {len(subset_df)} rows ({num_breaths} breaths)"
    )
    return len(subset_df)


class DemoConfig(Config):
    """
    Configuration overrides for the demo execution.
    Reduces model size and training duration for speed.
    """

    # Paths
    WORKING_DIR = "./working/demo_execution"
    CACHE_DIR = WORKING_DIR

    TRAIN_PATH = os.path.join(WORKING_DIR, "train_subset.csv")
    VAL_PATH = os.path.join(WORKING_DIR, "val_subset.csv")
    TEST_PATH = os.path.join(WORKING_DIR, "test_subset.csv")

    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Hyperparameters for Speed
    EPOCHS = 2
    BATCH_SIZE = 16  # Small batch for small data

    # Model Architecture (Scaled down)
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2

    # Data Loading
    NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo


def run_demo():
    print("=== Starting Ventilator Prediction Pipeline Demo ===")

    # 1. Setup
    demo_dir = DemoConfig.WORKING_DIR
    setup_demo_environment(demo_dir)

    # Set seeds
    torch.manual_seed(DemoConfig.SEED)
    np.random.seed(DemoConfig.SEED)

    # 2. Create Data Subsets
    print("\n[Step 1] Creating Data Subsets...")
    # We use the metadata files which are guaranteed to exist per instructions
    train_len = create_subset_csv(
        "./metadata/train.csv", DemoConfig.TRAIN_PATH, num_breaths=50
    )
    val_len = create_subset_csv(
        "./metadata/val.csv", DemoConfig.VAL_PATH, num_breaths=20
    )
    test_len = create_subset_csv(
        "./metadata/test.csv", DemoConfig.TEST_PATH, num_breaths=20
    )

    # 3. Data Pipeline
    print("\n[Step 2] Running Data Pipeline (Feature Engineering & Scaling)...")
    # Force reload to ensure we use the subsets, not cached full data if it existed
    train_ds, val_ds, test_ds = prepare_datasets(DemoConfig, load_cached_data=False)

    # Verify Dataset Logic
    print("Verifying dataset integrity...")
    assert (
        len(train_ds) == train_len // DemoConfig.SEQ_LEN
    ), "Train dataset size mismatch"
    assert len(val_ds) == val_len // DemoConfig.SEQ_LEN, "Val dataset size mismatch"
    assert len(test_ds) == test_len // DemoConfig.SEQ_LEN, "Test dataset size mismatch"

    # Check item shape: (80, num_features)
    sample_x, sample_y = train_ds[0]
    assert sample_x.shape == (DemoConfig.SEQ_LEN, len(DemoConfig.INPUT_FEATURES))
    assert sample_y.shape == (DemoConfig.SEQ_LEN,)
    print("Dataset verification passed.")

    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=DemoConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=DemoConfig.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=DemoConfig.BATCH_SIZE, shuffle=False)

    # 4. Model Initialization & Forward Pass Check
    print("\n[Step 3] Initializing Model & Checking Forward Pass...")
    device = torch.device(DemoConfig.DEVICE)
    model = VentilatorModel(DemoConfig).to(device)

    # Create a dummy batch
    dummy_input = torch.randn(2, DemoConfig.SEQ_LEN, DemoConfig.INPUT_DIM).to(device)
    with torch.no_grad():
        final_pred, aux_pred = model(dummy_input)

    # Verify Output Shapes
    assert final_pred.shape == (
        2,
        DemoConfig.SEQ_LEN,
        1,
    ), f"Unexpected final output shape: {final_pred.shape}"
    assert aux_pred.shape == (
        2,
        DemoConfig.SEQ_LEN,
        1,
    ), f"Unexpected aux output shape: {aux_pred.shape}"
    print("Model forward pass successful.")

    # 5. Training Loop
    print("\n[Step 4] Running Training Loop...")
    best_mae = train_model(train_loader, val_loader, config=DemoConfig)

    assert os.path.exists(DemoConfig.MODEL_PATH), "Model checkpoint was not saved."
    print(f"Training finished. Best MAE: {best_mae:.4f}")

    # 6. Inference & Submission
    print("\n[Step 5] Running Inference & Submission...")
    predict_and_submit(test_loader, config=DemoConfig)

    # Verify Submission
    assert os.path.exists(DemoConfig.SUBMISSION_PATH), "Submission file not found."
    sub_df = pd.read_csv(DemoConfig.SUBMISSION_PATH)

    print(f"Submission generated with shape: {sub_df.shape}")
    assert list(sub_df.columns) == [
        "id",
        "pressure",
    ], "Submission columns are incorrect."
    assert (
        len(sub_df) == test_len
    ), f"Submission length {len(sub_df)} != Test length {test_len}"
    assert not sub_df.isnull().values.any(), "Submission contains NaN values."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
