import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import provided library modules
from library.config import Config
from library.data import process_dataframe, RNADataset, get_dataloaders
from library.model import DenseContextNet
from library.loss import MaskedMCRMSELoss
from library.utils import set_seed, MetricTracker
from library.train import train_model


def create_dummy_data(num_samples=10):
    """
    Generates a dummy dataframe with random sequences and structures
    matching the competition format (length 107).
    """
    seq_len = 107
    scored_len = 68

    ids = [f"id_{i:04d}" for i in range(num_samples)]
    sequences = []
    structures = []
    loops = []

    # Randomly generate sequence, structure, loop strings
    bases = ["A", "G", "C", "U"]
    structs = [".", "(", ")"]
    loop_types = ["S", "M", "I", "B", "H", "E", "X"]

    for _ in range(num_samples):
        s = "".join(np.random.choice(bases, seq_len))
        st = "".join(np.random.choice(structs, seq_len))
        l = "".join(np.random.choice(loop_types, seq_len))
        sequences.append(s)
        structures.append(st)
        loops.append(l)

    df = pd.DataFrame(
        {
            "id": ids,
            "sequence": sequences,
            "structure": structures,
            "predicted_loop_type": loops,
        }
    )

    # Add target columns (stored as stringified lists)
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for col in target_cols:
        # Create random float lists of length scored_len
        values = []
        for _ in range(num_samples):
            # Random floats between 0 and 1
            arr = np.random.rand(scored_len).tolist()
            # Convert to string representation
            values.append(str(arr))
        df[col] = values

    return df


def run_demo():
    # 1. Setup
    print("=== Setting up Demo Environment ===")
    set_seed(42)
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    # Define paths for dummy data
    train_csv_path = os.path.join(demo_dir, "train.csv")
    val_csv_path = os.path.join(demo_dir, "val.csv")
    test_csv_path = os.path.join(demo_dir, "test.csv")

    # Create and save dummy data
    print("Generating dummy datasets...")
    df_train = create_dummy_data(num_samples=10)
    df_val = create_dummy_data(num_samples=4)
    df_test = create_dummy_data(num_samples=4)
    # Test set doesn't strictly need targets in the file, but process_dataframe handles it if missing.
    # However, create_dummy_data adds them. We can drop them for realism if we want,
    # but for this demo, keeping them is fine as the test loader ignores them.

    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    # 2. Verify Data Processing
    print("\n=== Verifying Data Processing Logic ===")
    # Test process_dataframe
    inputs, partner_indices, targets, ids = process_dataframe(df_train, is_test=False)

    # Assertions
    # Input shape: (N, 107, 19)
    assert inputs.shape == (
        10,
        107,
        19,
    ), f"Expected inputs shape (10, 107, 19), got {inputs.shape}"
    # Partner indices shape: (N, 107)
    assert partner_indices.shape == (
        10,
        107,
    ), f"Expected partner_indices shape (10, 107), got {partner_indices.shape}"
    # Targets shape: (N, 107, 5) - Note: padded from 68 to 107
    assert targets.shape == (
        10,
        107,
        5,
    ), f"Expected targets shape (10, 107, 5), got {targets.shape}"
    assert len(ids) == 10

    print("Data processing shapes verified.")

    # Test Dataset
    dataset = RNADataset(inputs, partner_indices, targets)
    item_in, item_idx, item_tgt = dataset[0]
    assert isinstance(item_in, torch.Tensor)
    assert isinstance(item_idx, torch.Tensor)
    assert isinstance(item_tgt, torch.Tensor)
    print("RNADataset instantiation and retrieval verified.")

    # 3. Verify Model Architecture
    print("\n=== Verifying Model Architecture ===")
    device = "cpu"  # Force CPU for simple verification
    model = DenseContextNet().to(device)
    model.eval()

    # Create a batch
    batch_size = 2
    dummy_input = torch.from_numpy(inputs[:batch_size]).to(device)
    dummy_pidx = torch.from_numpy(partner_indices[:batch_size]).long().to(device)

    # Handle the partner index mapping (unpaired -1 -> self) usually done in Dataset
    # But here we just manually fix it for the raw model call
    for b in range(batch_size):
        mask = dummy_pidx[b] == -1
        dummy_pidx[b][mask] = torch.arange(107)[mask]

    with torch.no_grad():
        output = model(dummy_input, dummy_pidx)

    # Expected output: (Batch, Length, Num_Targets) = (2, 107, 5)
    assert output.shape == (
        2,
        107,
        5,
    ), f"Expected model output (2, 107, 5), got {output.shape}"
    print("Model forward pass successful. Output shape verified.")

    # 4. Verify Loss Function
    print("\n=== Verifying Masked MCRMSE Loss ===")
    criterion = MaskedMCRMSELoss()

    # Scored columns are indices [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Unscored are [2, 4]
    # Scored length is 68.

    # Case 1: Perfect prediction
    preds_perfect = torch.zeros((2, 107, 5))
    targets_perfect = torch.zeros((2, 107, 5))
    loss_zero = criterion(preds_perfect, targets_perfect)
    assert torch.isclose(
        loss_zero, torch.tensor(0.0)
    ), f"Expected 0 loss, got {loss_zero}"

    # Case 2: Error only in unscored columns (indices 2, 4) or unscored positions (>67)
    preds_unscored_err = torch.zeros((2, 107, 5))
    targets_unscored_err = torch.zeros((2, 107, 5))

    # Add error to index 2 (deg_pH10) which is NOT scored
    targets_unscored_err[:, :68, 2] = 100.0
    # Add error to position 70 (which is > 68)
    targets_unscored_err[:, 70, 0] = 100.0

    loss_masked = criterion(preds_unscored_err, targets_unscored_err)
    assert torch.isclose(
        loss_masked, torch.tensor(0.0)
    ), f"Expected 0 loss (masked), got {loss_masked}"

    # Case 3: Error in scored column [0] at valid position [0]
    preds_err = torch.zeros((1, 107, 5))
    targets_err = torch.zeros((1, 107, 5))
    targets_err[0, 0, 0] = 1.0  # Error of 1.0 in one cell

    # Calculation:
    # Col 0 MSE: (1^2 + 0...)/68 = 1/68
    # Col 1 MSE: 0
    # Col 3 MSE: 0
    # RMSEs: sqrt(1/68), 0, 0
    # MCRMSE: (sqrt(1/68) + 0 + 0) / 3
    expected_val = (np.sqrt(1.0 / 68.0)) / 3.0

    loss_calc = criterion(preds_err, targets_err)
    assert torch.isclose(
        loss_calc, torch.tensor(expected_val, dtype=torch.float32), atol=1e-5
    ), f"Expected {expected_val}, got {loss_calc.item()}"

    print("Loss function masking and calculation verified.")

    # 5. Verify Metric Tracker
    print("\n=== Verifying Metric Tracker ===")
    tracker = MetricTracker()
    tracker.reset()

    # Use the same data as Case 3 above
    tracker.update(preds_err, targets_err)
    result = tracker.result()

    # MetricTracker logic:
    # SSE per column. Col 0 SSE = 1.0. Col 1,3 SSE = 0.
    # Count per column = 1 * 68 = 68.
    # MSE: [1/68, 0, 0]
    # RMSE: [sqrt(1/68), 0, 0]
    # Mean RMSE: same as above
    assert np.isclose(
        result, expected_val, atol=1e-5
    ), f"Expected tracker result {expected_val}, got {result}"
    print("MetricTracker verified.")

    # 6. Integration Test: Full Training Loop
    print("\n=== Verifying Full Training Integration ===")

    # Override Config paths to use our dummy data
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_CSV = train_csv_path
    Config.VAL_CSV = val_csv_path
    Config.TEST_CSV = test_csv_path

    # Override Cache paths to avoid conflicts and ensure we use the new dummy data
    Config.TRAIN_CACHE = os.path.join(demo_dir, "data_cache", "train_data.npz")
    Config.VAL_CACHE = os.path.join(demo_dir, "data_cache", "val_data.npz")
    Config.TEST_CACHE = os.path.join(demo_dir, "data_cache", "test_data.npz")

    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Run training in debug mode
    # debug=True sets epochs=2, batch_size=4
    # We will override epochs to 1 for even faster execution
    print("Starting train_model()...")
    train_model(debug=True, epochs=1)

    # Verify output file exists
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    print("\nIntegration test passed. Model saved successfully.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
