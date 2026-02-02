import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.utils import seed_everything, get_pair_distance_vector, calculate_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.train import train, generate_submission


def create_mini_datasets(config: Config):
    """
    Creates small subsets of the original data to speed up the demonstration.
    """
    print("Creating mini datasets for rapid demonstration...")

    # Define paths for mini datasets
    mini_train_path = os.path.join(config.working_dir, "mini_train.parquet")
    mini_val_path = os.path.join(config.working_dir, "mini_val.parquet")
    mini_test_path = os.path.join(config.working_dir, "mini_test.parquet")

    # Load original metadata
    # Note: We use the paths defined in the default config to find the source
    default_config = Config()

    df_train = pd.read_parquet(default_config.train_file)
    df_val = pd.read_parquet(default_config.val_file)
    df_test = pd.read_parquet(default_config.test_file)

    # Sample subsets (e.g., 10 samples each)
    df_train_mini = df_train.head(10)
    df_val_mini = df_val.head(10)
    df_test_mini = df_test.head(10)

    # Save to working directory
    df_train_mini.to_parquet(mini_train_path, index=False)
    df_val_mini.to_parquet(mini_val_path, index=False)
    df_test_mini.to_parquet(mini_test_path, index=False)

    print(f"Mini datasets saved to {config.working_dir}")

    # Return updated paths
    return mini_train_path, mini_val_path, mini_test_path


def verify_utils():
    """
    Verifies the logic of utility functions.
    """
    print("\nVerifying Utils...")

    # Test Case: ((..))
    # Indices: 012345
    # Pairs: (0, 5), (1, 4)
    # Distances:
    # 0: 5 - 0 = 5
    # 1: 4 - 1 = 3
    # 2: Unpaired = 0
    # 3: Unpaired = 0
    # 4: 1 - 4 = -3
    # 5: 0 - 5 = -5
    structure = "((..))"
    expected = np.array([5, 3, 0, 0, -3, -5], dtype=np.int32)
    result = get_pair_distance_vector(structure)

    assert np.array_equal(
        result, expected
    ), f"Utils Verification Failed: Expected {expected}, got {result}"
    print("Utils verification passed.")


def verify_model(config: Config):
    """
    Verifies the model architecture and forward pass.
    """
    print("\nVerifying Model...")

    device = torch.device("cpu")  # Use CPU for simple verification
    model = RNAModel(config).to(device)
    model.eval()

    batch_size = 2
    seq_len = config.seq_len  # 107

    # Create dummy inputs
    # Seqs: Integers [0, 3]
    seqs = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    # Loops: Integers [0, 6]
    loops = torch.randint(0, config.loop_types_size, (batch_size, seq_len)).to(device)
    # Dists: Signed integers (approx range -107 to 107)
    dists = torch.randint(-seq_len, seq_len, (batch_size, seq_len)).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(seqs, loops, dists)

    # Check Output Shape: (Batch, SeqLen, NumTargets)
    # NumTargets is 3 by default in Config (reactivity, deg_Mg_pH10, deg_Mg_50C)
    expected_shape = (batch_size, seq_len, len(config.target_cols))

    assert (
        outputs.shape == expected_shape
    ), f"Model Verification Failed: Expected shape {expected_shape}, got {outputs.shape}"

    print(f"Model verification passed. Output shape: {outputs.shape}")


def verify_loss():
    """
    Verifies the Masked MSE Loss logic.
    """
    print("\nVerifying Loss...")
    criterion = MaskedMSELoss()

    # Setup: Batch=1, Seq=4, Targets=1
    # Preds:  [1.0, 2.0, 3.0, 4.0]
    # Truth:  [1.0, 2.0, 5.0, 4.0]
    # Mask:   [1.0, 1.0, 1.0, 0.0] (Last position ignored)

    preds = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # (1, 4, 1)
    targets = torch.tensor([[[1.0], [2.0], [5.0], [4.0]]])  # (1, 4, 1)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # (1, 4)

    # Calculation:
    # Pos 0: (1-1)^2 = 0
    # Pos 1: (2-2)^2 = 0
    # Pos 2: (3-5)^2 = 4
    # Pos 3: Ignored (Mask 0)
    # Sum Errors = 4
    # Valid Elements = 3 (Mask sum) * 1 (Channels) = 3
    # MSE = 4 / 3 = 1.3333...

    loss = criterion(preds, targets, mask)
    expected_loss = 4.0 / 3.0

    assert torch.isclose(
        loss, torch.tensor(expected_loss), atol=1e-5
    ), f"Loss Verification Failed: Expected {expected_loss}, got {loss.item()}"

    print(f"Loss verification passed. Loss: {loss.item():.4f}")


def run_demo():
    # 1. Setup Configuration
    # We create a specific directory for this demo execution
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    config = Config()
    config.working_dir = demo_working_dir
    config.num_epochs = 1  # Run only 1 epoch for speed
    config.batch_size = 4  # Small batch size
    config.num_workers = 0  # Avoid multiprocessing overhead for demo
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Prepare Data
    # Create mini parquet files and point config to them
    mini_train, mini_val, mini_test = create_mini_datasets(config)
    config.train_file = mini_train
    config.val_file = mini_val
    config.test_file = mini_test

    # 3. Verify Components
    verify_utils()
    verify_model(config)
    verify_loss()

    # 4. Run Training Pipeline
    print("\nStarting Training Pipeline...")
    # This calls the library train function which handles data loading, model init, loop, and saving
    best_model_path = train(config)

    assert os.path.exists(best_model_path), "Training failed: best_model.pth not found."
    print(f"Training completed. Best model saved to {best_model_path}")

    # 5. Generate Submission
    print("\nGenerating Submission...")
    # This calls the library submission function
    generate_submission(config)

    assert os.path.exists(
        config.submission_file
    ), "Submission generation failed: submission.csv not found."

    # Validate submission format briefly
    sub_df = pd.read_csv(config.submission_file)
    print(f"Submission generated with {len(sub_df)} rows.")
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)
    run_demo()
