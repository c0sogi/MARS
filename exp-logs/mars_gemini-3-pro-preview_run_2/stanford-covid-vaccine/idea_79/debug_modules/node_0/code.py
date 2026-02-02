import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader
import torch.optim as optim

# Import from the provided library files
from library.utils import set_seed, parse_dot_bracket, parse_list_column, mcrmse
from library.data import preprocess_data, RNADataset
from library.model import RIS_DRN, loss_fn
from library.config import SCORED_SEQ_LENGTH, SCORED_INDICES, SEQ_LENGTH


def main():
    # 1. Setup and Reproducibility
    print(">>> Setting up environment...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # Define paths for temporary demonstration files
    working_dir = "./working"
    mini_train_csv = os.path.join(working_dir, "demo_mini_train.csv")
    mini_cache_path = os.path.join(working_dir, "demo_cache.npz")

    # 2. Demonstrate Library Utils
    print("\n>>> Demonstrating library.utils...")

    # Test parse_dot_bracket
    structure = "((..))"
    partner_map = parse_dot_bracket(structure)
    expected_map = np.array([5, 4, -1, -1, 1, 0], dtype=np.int32)
    np.testing.assert_array_equal(
        partner_map, expected_map, err_msg="Partner map parsing failed"
    )
    print("    parse_dot_bracket: Verified.")

    # Test parse_list_column
    list_str = "[0.1, 0.2, 0.3]"
    parsed_arr = parse_list_column(list_str)
    expected_arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    np.testing.assert_allclose(
        parsed_arr, expected_arr, err_msg="List column parsing failed"
    )
    print("    parse_list_column: Verified.")

    # Test mcrmse metric
    # Create dummy ground truth and predictions
    # Shape: (N=2, Seq=3, Channels=2)
    y_true = np.array(
        [[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]]
    )
    y_pred = np.array(
        [[[1.1, 1.9], [1.1, 1.9], [1.1, 1.9]], [[1.1, 1.9], [1.1, 1.9], [1.1, 1.9]]]
    )

    # Error is 0.1 for all elements. RMSE per column should be 0.1. Mean of RMSEs = 0.1.
    score = mcrmse(y_true, y_pred)
    np.testing.assert_almost_equal(
        score, 0.1, decimal=5, err_msg="MCRMSE calculation failed"
    )
    print("    mcrmse: Verified.")

    # 3. Demonstrate Data Processing
    print("\n>>> Demonstrating library.data...")

    # Create a mini dataset from the metadata
    # We read the first 20 rows to ensure we have enough for a small batch
    full_train_df = pd.read_csv("./metadata/train.csv")
    mini_df = full_train_df.head(20).copy()
    mini_df.to_csv(mini_train_csv, index=False)
    print(f"    Created mini dataset with {len(mini_df)} samples at {mini_train_csv}")

    # Run preprocessing
    # We disable loading from cache to force processing logic to run
    data_dict = preprocess_data(
        csv_path=mini_train_csv,
        cache_path=mini_cache_path,
        is_test=False,
        load_cached_data=False,
    )

    # Verify data dictionary structure and shapes
    inputs = data_dict["inputs"]
    targets = data_dict["targets"]
    p_map = data_dict["partner_map"]

    assert inputs.shape == (20, SEQ_LENGTH, 18), f"Input shape mismatch: {inputs.shape}"
    assert targets.shape == (
        20,
        SEQ_LENGTH,
        5,
    ), f"Target shape mismatch: {targets.shape}"
    assert p_map.shape == (20, SEQ_LENGTH), f"Partner map shape mismatch: {p_map.shape}"
    print("    preprocess_data: Shapes verified.")

    # Create Dataset and DataLoader
    dataset = RNADataset(data_dict, is_test=False)
    # Use a small batch size for demonstration
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Fetch one batch
    batch = next(iter(dataloader))
    print(f"    DataLoader: Fetched batch with keys {list(batch.keys())}")
    assert batch["inputs"].shape == (batch_size, SEQ_LENGTH, 18)
    assert batch["targets"].shape == (batch_size, SEQ_LENGTH, 5)

    # 4. Demonstrate Model
    print("\n>>> Demonstrating library.model...")

    model = RIS_DRN().to(device)
    print("    Model instantiated successfully.")

    # Move batch to device
    b_inputs = batch["inputs"].to(device)
    b_partner_map = batch["partner_map"].to(device)
    b_targets = batch["targets"].to(device)

    # Forward pass
    logits_1, logits_2 = model(b_inputs, b_partner_map)

    # Verify output shapes: (Batch, SeqLen, 5)
    expected_shape = (batch_size, SEQ_LENGTH, 5)
    assert (
        logits_1.shape == expected_shape
    ), f"Logits 1 shape mismatch: {logits_1.shape}"
    assert (
        logits_2.shape == expected_shape
    ), f"Logits 2 shape mismatch: {logits_2.shape}"
    print("    Forward pass: Output shapes verified.")

    # Calculate Loss
    loss = loss_fn(logits_1, logits_2, b_targets)
    print(f"    Loss calculated: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # 5. Demonstrate Training Step
    print("\n>>> Demonstrating Training Step...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    # Capture initial weights of a specific layer to verify update
    initial_weight = model.interaction.head.weight.clone()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if weights updated
    updated_weight = model.interaction.head.weight
    assert not torch.equal(
        initial_weight, updated_weight
    ), "Weights did not update after optimizer step"
    print("    Optimizer step: Weights updated successfully.")

    # 6. Demonstrate Validation Logic
    print("\n>>> Demonstrating Validation Logic...")
    model.eval()
    with torch.no_grad():
        _, val_logits = model(b_inputs, b_partner_map)

        # Slice to scored length as done in library.train.validate
        preds_sliced = val_logits[:, :SCORED_SEQ_LENGTH, :]
        targets_sliced = b_targets[:, :SCORED_SEQ_LENGTH, :]

        # Filter for scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # SCORED_INDICES are [0, 1, 3]
        preds_filtered = preds_sliced[:, :, SCORED_INDICES]
        targets_filtered = targets_sliced[:, :, SCORED_INDICES]

        val_score = mcrmse(targets_filtered, preds_filtered)
        print(f"    Validation MCRMSE on batch: {val_score:.6f}")

    # 7. Cleanup
    print("\n>>> Cleaning up...")
    if os.path.exists(mini_train_csv):
        os.remove(mini_train_csv)
    if os.path.exists(mini_cache_path):
        os.remove(mini_cache_path)
    print("    Temporary files removed.")

    print("\n>>> Demonstration completed successfully.")


if __name__ == "__main__":
    main()
