import os
import shutil
import numpy as np
import torch
import pandas as pd

# Import provided library components
from library.config import Config
from library.data_utils import one_hot_encode, get_adjacency_info, load_or_process_data
from library.model_components import RNAModel
from library.loss_metric import MCRMSELoss, calculate_mcrmse
from library.model import train_model, generate_submission
from library.train_eval import train_epoch, validate


def test_data_utils():
    print("\n=== Testing Data Utils ===")

    # 1. Test One-Hot Encoding
    seq = "ACGU"
    alphabet = "ACGU"
    encoded = one_hot_encode(seq, alphabet)

    expected = np.eye(4, dtype=np.float32)
    assert np.array_equal(
        encoded, expected
    ), "One-hot encoding failed for simple sequence."
    print("One-hot encoding verified.")

    # 2. Test Adjacency Info
    # Structure: . ( . )
    # Indices: 0 1 2 3
    # Pairs: 1-3
    structure = ".(.)"
    indices, mask = get_adjacency_info(structure)

    expected_indices = np.array([0, 3, 2, 1])  # 0->0, 1->3, 2->2, 3->1
    expected_mask = np.array([0, 1, 0, 1], dtype=np.float32)

    assert np.array_equal(
        indices, expected_indices
    ), f"Adjacency indices incorrect. Got {indices}"
    assert np.array_equal(mask, expected_mask), f"Adjacency mask incorrect. Got {mask}"
    print("Adjacency info parsing verified.")


def test_loss_metric():
    print("\n=== Testing Loss & Metric ===")

    config = Config()
    # Setup dummy data
    # Batch=1, Seq=107, Targets=5
    # Scored length is 68
    preds = torch.ones((1, 107, 5), dtype=torch.float32) * 2.0
    targets = torch.ones((1, 68, 5), dtype=torch.float32) * 1.0

    # Expected behavior:
    # 1. Slice preds to 68 -> all 2.0
    # 2. Diff = 2.0 - 1.0 = 1.0
    # 3. MSE = 1.0, RMSE = 1.0
    # 4. Mean across columns = 1.0

    criterion = MCRMSELoss()
    loss = criterion(preds, targets)

    assert torch.isclose(
        loss, torch.tensor(1.0), atol=1e-4
    ), f"Loss calculation incorrect. Got {loss.item()}"
    print("MCRMSELoss verified.")

    # Test Metric Calculation (Scored columns only)
    # Config.scored_cols are indices [0, 1, 3] usually (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # If all values are identical errors, metric should still be 1.0
    metric = calculate_mcrmse(preds, targets)
    assert np.isclose(
        metric, 1.0, atol=1e-4
    ), f"Metric calculation incorrect. Got {metric}"
    print("Metric calculation verified.")


def test_model_forward(config):
    print("\n=== Testing Model Forward Pass ===")

    device = torch.device("cpu")  # Test on CPU for simplicity
    model = RNAModel(config).to(device)
    model.eval()

    batch_size = 2
    seq_len = config.seq_len  # 107
    num_feat = config.num_features  # 14

    # Create dummy inputs
    inputs = torch.randn(batch_size, seq_len, num_feat).to(device)
    bpp_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    bpp_mask = torch.zeros(batch_size, seq_len, 1).to(device)

    with torch.no_grad():
        output = model(inputs, bpp_indices, bpp_mask)

    expected_shape = (batch_size, seq_len, config.num_targets)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
    print(f"Model forward pass verified. Output shape: {output.shape}")


def run_demo_pipeline():
    print("\n=== Running Demo Pipeline ===")

    # 1. Setup Config for Demo
    config = Config()
    config.debug = True
    config.debug_subset_size = 50  # Small subset for speed
    config.epochs = 1
    config.batch_size = 4
    config.working_dir = "./working/demo_execution"
    config.model_save_path = os.path.join(config.working_dir, "demo_model.pth")
    config.submission_path = os.path.join(config.working_dir, "demo_submission.csv")

    # Separate cache files for demo to avoid overwriting production caches
    config.train_cache_path = os.path.join(config.working_dir, "train_data.npz")
    config.val_cache_path = os.path.join(config.working_dir, "val_data.npz")
    config.test_cache_path = os.path.join(config.working_dir, "test_data.npz")

    os.makedirs(config.working_dir, exist_ok=True)

    # 2. Verify Data Loading
    print("Loading/Processing Data...")
    train_ds = load_or_process_data("train", config)
    val_ds = load_or_process_data("val", config)

    assert (
        len(train_ds) == config.debug_subset_size
    ), f"Train dataset size mismatch. Expected {config.debug_subset_size}, got {len(train_ds)}"
    sample = train_ds[0]
    assert "sequence" in sample and "targets" in sample, "Dataset sample missing keys."
    print("Data loading verified.")

    # 3. Run Training Loop (via library function)
    print("Executing Training Loop...")
    saved_model_path = train_model(config)

    assert os.path.exists(saved_model_path), "Model file was not saved."
    print(f"Training verified. Model saved to {saved_model_path}")

    # 4. Run Submission Generation (via library function)
    print("Generating Submission...")
    # Ensure test cache is also handled
    generate_submission(config)

    assert os.path.exists(config.submission_path), "Submission file was not created."

    # Check submission format
    sub_df = pd.read_csv(config.submission_path)
    expected_rows = config.debug_subset_size * config.seq_len
    # Note: In debug mode, test set is also truncated to debug_subset_size
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    print(f"Submission verified. Rows: {len(sub_df)}")
    print("Demo pipeline completed successfully.")


if __name__ == "__main__":
    # Set fixed seeds
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Run Unit Tests
        test_data_utils()
        test_loss_metric()

        # Run Model Test
        config = Config()
        test_model_forward(config)

        # Run Full Pipeline Integration
        run_demo_pipeline()

        print("\nAll tests and demonstrations passed!")

    except AssertionError as e:
        print(f"\nAssertion Failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
