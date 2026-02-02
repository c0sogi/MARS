import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, parse_list_column, calculate_global_mcrmse
from library.data import Preloader, RNADataset, get_dataloaders
from library.model import BridgedHybridNet
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_demo_subset():
    """
    Creates a small subset of the metadata CSVs to ensure the demo runs quickly.
    """
    print("Creating demo data subsets...")

    # Define temporary directory for demo data
    demo_data_dir = "./working/demo_metadata"
    os.makedirs(demo_data_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Sample a small fraction (e.g., 20 samples)
    subset_size = 20
    train_subset = train_df.head(subset_size)
    val_subset = val_df.head(subset_size)
    test_subset = test_df.head(subset_size)

    # Save subsets
    train_subset_path = os.path.join(demo_data_dir, "train_subset.csv")
    val_subset_path = os.path.join(demo_data_dir, "val_subset.csv")
    test_subset_path = os.path.join(demo_data_dir, "test_subset.csv")

    train_subset.to_csv(train_subset_path, index=False)
    val_subset.to_csv(val_subset_path, index=False)
    test_subset.to_csv(test_subset_path, index=False)

    return train_subset_path, val_subset_path, test_subset_path


def verify_utils():
    """
    Verifies utility functions.
    """
    print("Verifying utils...")

    # Test parse_list_column
    s = "[0.1, 0.2, 0.3]"
    arr = parse_list_column(s)
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    # Test calculate_global_mcrmse
    # Shape: (N=2, Seq=3, Cols=2)
    preds = np.array(
        [[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]]
    )
    targets = np.array(
        [[[1.1, 1.9], [1.1, 1.9], [1.1, 1.9]], [[1.1, 1.9], [1.1, 1.9], [1.1, 1.9]]]
    )

    # RMSE for col 0: sqrt((0.1^2 + ...)/N) = 0.1
    # RMSE for col 1: sqrt((-0.1^2 + ...)/N) = 0.1
    # MCRMSE = (0.1 + 0.1) / 2 = 0.1
    score = calculate_global_mcrmse(preds, targets)
    assert np.isclose(score, 0.1), f"Expected 0.1, got {score}"
    print("Utils verification passed.")


def verify_model_architecture():
    """
    Verifies the model forward pass and output shape.
    """
    print("Verifying model architecture...")

    batch_size = 4
    seq_len = Config.SEQ_LEN  # 107
    in_channels = Config.IN_CHANNELS  # 14

    model = BridgedHybridNet()
    model.eval()

    # Create dummy inputs
    inputs = torch.randn(batch_size, in_channels, seq_len)
    # Partner indices: just map to self for simplicity
    partner_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    with torch.no_grad():
        output = model(inputs, partner_indices)

    # Expected output: (Batch, Seq_Len, 5)
    expected_shape = (batch_size, seq_len, 5)
    assert (
        output.shape == expected_shape
    ), f"Shape mismatch: {output.shape} vs {expected_shape}"
    print("Model architecture verification passed.")


def run_demo_pipeline(train_path, val_path, test_path):
    """
    Runs the training and submission pipeline using the Trainer class.
    Overrides Config to use demo data and fast hyperparameters.
    """
    print("Running demo pipeline...")

    # 1. Override Config
    Config.TRAIN_CSV = train_path
    Config.VAL_CSV = val_path
    Config.TEST_CSV = test_path

    # Use a separate working directory for the demo
    Config.WORKING_DIR = "./working/demo_execution/"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Speed up training
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Initialize Trainer
    trainer = Trainer()

    # 3. Run Training
    print("  > Starting training...")
    trainer.run_training()

    # Verify model file was created
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "best_model.pth was not created."
    print("  > Training finished. Model saved.")

    # 4. Generate Submission
    print("  > Generating submission...")
    trainer.generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "submission.csv was not created."

    # Check submission content format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch."
    assert len(sub_df) > 0, "Submission file is empty."

    print(f"  > Submission generated at {Config.SUBMISSION_PATH}")
    print("Demo pipeline completed successfully.")


if __name__ == "__main__":
    # 1. Set Seed
    set_seed(42)

    # 2. Verify Components
    verify_utils()
    verify_model_architecture()

    # 3. Create Demo Data
    train_csv, val_csv, test_csv = create_demo_subset()

    # 4. Run Pipeline
    run_demo_pipeline(train_csv, val_csv, test_csv)
