import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import shutil

# Import library components
from library.config import Config
from library.utils import set_seed, get_pair_map
from library.data import get_dataloaders, one_hot_encode, process_data
from library.model import SpatiallyAugmentedBiGRU
from library.train import Trainer, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def create_demo_datasets(config):
    """
    Creates small subsets of the original metadata to allow for rapid execution.
    """
    print("Creating demo datasets...")

    # Define source paths (original metadata)
    orig_train_path = "./metadata/train.parquet"
    orig_val_path = "./metadata/val.parquet"
    orig_test_path = "./metadata/test.parquet"

    # Load subsets (e.g., 20 samples for train, 10 for val/test)
    df_train = pd.read_parquet(orig_train_path).head(20)
    df_val = pd.read_parquet(orig_val_path).head(10)
    df_test = pd.read_parquet(orig_test_path).head(10)

    # Save to the paths defined in the modified config
    df_train.to_parquet(config.TRAIN_PATH, index=False)
    df_val.to_parquet(config.VAL_PATH, index=False)
    df_test.to_parquet(config.TEST_PATH, index=False)

    print(f"Demo datasets created in {config.WORKING_DIR}")


def verify_utilities():
    """
    Verifies the logic of helper functions.
    """
    print("\nVerifying utilities...")

    # 1. Test get_pair_map
    # Structure: ((..)) -> Indices: 0 pairs with 5, 1 pairs with 4. 2,3 unpaired.
    structure = "((..))"
    expected_map = np.array([5, 4, -1, -1, 1, 0])
    pair_map = get_pair_map(structure)

    assert np.array_equal(
        pair_map, expected_map
    ), f"get_pair_map failed. Expected {expected_map}, got {pair_map}"
    print("  get_pair_map: OK")

    # 2. Test one_hot_encode
    seq = "AGUC"
    token_dict = {"A": 0, "G": 1, "U": 2, "C": 3}
    length = 4
    encoded = one_hot_encode(seq, token_dict, length)

    expected_encoded = np.eye(4)  # Identity matrix for this specific case
    assert np.array_equal(encoded, expected_encoded), "one_hot_encode failed."
    print("  one_hot_encode: OK")


def main():
    # 1. Setup
    set_seed(42)

    # Initialize Config
    # We use a small batch size and few epochs for the demo
    config = Config(debug=True, epochs=2, batch_size=4)

    # Override paths to use a separate demo directory
    config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Update data paths to point to our demo subset files
    config.TRAIN_PATH = os.path.join(config.WORKING_DIR, "train_subset.parquet")
    config.VAL_PATH = os.path.join(config.WORKING_DIR, "val_subset.parquet")
    config.TEST_PATH = os.path.join(config.WORKING_DIR, "test_subset.parquet")

    # Update cache and output paths
    config.TRAIN_CACHE = os.path.join(config.WORKING_DIR, "train_cache.pt")
    config.VAL_CACHE = os.path.join(config.WORKING_DIR, "val_cache.pt")
    config.TEST_CACHE = os.path.join(config.WORKING_DIR, "test_cache.pt")
    config.MODEL_PATH = os.path.join(config.WORKING_DIR, "best_model.pth")
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "submission.csv")

    # 2. Create Data
    create_demo_datasets(config)

    # 3. Verify Utils
    verify_utilities()

    # 4. Data Processing
    print("\nInitializing DataLoaders...")
    # This will trigger process_data and caching
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # Verify DataLoader output shapes
    sample_X, sample_y = next(iter(train_loader))
    print(f"  Train Batch X shape: {sample_X.shape}")  # Should be (Batch, 107, 28)
    print(f"  Train Batch y shape: {sample_y.shape}")  # Should be (Batch, 68, 5)

    assert sample_X.shape[1] == config.SEQ_LEN, "Incorrect sequence length in features"
    assert sample_X.shape[2] == config.INPUT_DIM, "Incorrect feature dimension"
    assert (
        sample_y.shape[1] == config.PRED_LEN
    ), "Incorrect prediction length in targets"
    assert sample_y.shape[2] == config.NUM_CLASSES, "Incorrect number of target classes"

    # 5. Model Initialization
    print("\nInitializing Model...")
    model = SpatiallyAugmentedBiGRU(config)

    # Verify Forward Pass
    model.eval()
    with torch.no_grad():
        output = model(sample_X)
    print(f"  Model Output shape: {output.shape}")  # Should be (Batch, 107, 5)

    assert output.shape == (
        sample_X.shape[0],
        config.SEQ_LEN,
        config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # 6. Training Loop
    print("\nStarting Training (Demo)...")
    trainer = Trainer(config, model, train_loader, val_loader)
    trainer.fit()

    # Verify Model Checkpoint exists
    assert os.path.exists(config.MODEL_PATH), "Model checkpoint was not saved."
    print("  Training finished and model saved.")

    # 7. Inference and Submission
    print("\nGenerating Submission...")
    # Load best model state
    best_state = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(best_state)

    generate_submission(config, model, test_loader)

    # Verify Submission File
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"  Submission shape: {df_sub.shape}")
    print(f"  Submission columns: {list(df_sub.columns)}")

    # Expected rows: 10 test samples * 107 positions = 1070 rows
    expected_rows = 10 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check for required columns
    expected_cols = ["id_seqpos"] + config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
