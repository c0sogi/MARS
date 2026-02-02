import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import (
    SEQ_LEN,
    SEQ_SCORED,
    TARGET_COLS,
    VOCAB_SIZE_SEQ,
    VOCAB_SIZE_STRUCT,
    VOCAB_SIZE_LOOP,
    INPUT_CHANNELS,
)
from library.utils import set_seed, parse_structure_to_indices, MCRMSE
from library.data import preprocess_dataframe, RNADataset
from library.model import DeepPostNormBiGRU, StructuralInteractionLayer


def demo_utils_logic():
    print("--- 1. Testing Utility Functions ---")

    # Test parse_structure_to_indices
    # Structure: ((..)) -> indices: 0-5, 1-4. 2,3 are unpaired (-1)
    structure = "((..))"
    expected_indices = np.array([5, 4, -1, -1, 1, 0], dtype=np.int32)
    computed_indices = parse_structure_to_indices(structure)

    np.testing.assert_array_equal(
        computed_indices,
        expected_indices,
        err_msg="Structure parsing failed for '((..))'",
    )
    print("✓ parse_structure_to_indices logic verified.")

    # Test MCRMSE Metric
    # Logic: It slices inputs to SEQ_SCORED (68).
    # We create tensors of shape (1, 107, 1) for simplicity.
    # Case: Error is 1.0 for first 68 positions, 100.0 for the rest.
    # Result should be 1.0, ignoring the tail.

    y_true = torch.zeros((1, SEQ_LEN, 1))
    y_pred = torch.zeros((1, SEQ_LEN, 1))

    # Set error in scored region
    y_pred[:, :SEQ_SCORED, :] = 1.0

    # Set huge error in unscored region (should be ignored)
    y_pred[:, SEQ_SCORED:, :] = 100.0

    score = MCRMSE(y_true, y_pred)

    # Expected: RMSE of 1.0 is 1.0. Mean of single column is 1.0.
    assert (
        abs(score.item() - 1.0) < 1e-5
    ), f"MCRMSE failed. Expected 1.0, got {score.item()}"
    print("✓ MCRMSE metric logic (including slicing) verified.")


def demo_data_pipeline():
    print("\n--- 2. Testing Data Pipeline ---")

    # Create synthetic dataframe
    num_samples = 4
    dummy_seq = "A" * SEQ_LEN
    dummy_struct = "." * SEQ_LEN
    dummy_loop = "E" * SEQ_LEN

    # Create dummy targets (list of floats)
    # Targets in the raw data are lists of length SEQ_SCORED
    dummy_target_vals = [0.5] * SEQ_SCORED

    data = {
        "id": [f"id_{i}" for i in range(num_samples)],
        "sequence": [dummy_seq] * num_samples,
        "structure": [dummy_struct] * num_samples,
        "predicted_loop_type": [dummy_loop] * num_samples,
    }

    # Add target columns
    for col in TARGET_COLS:
        data[col] = [dummy_target_vals] * num_samples

    df = pd.DataFrame(data)

    # Run preprocessing
    # is_test=False means it will process targets
    processed_data = preprocess_dataframe(df, is_test=False)

    # Verify shapes
    inputs = processed_data["inputs"]
    pair_indices = processed_data["pair_indices"]
    targets = processed_data["targets"]

    # Input shape: (N, SEQ_LEN, CHANNELS)
    assert inputs.shape == (
        num_samples,
        SEQ_LEN,
        INPUT_CHANNELS,
    ), f"Input shape mismatch. Got {inputs.shape}"

    # Pair indices shape: (N, SEQ_LEN)
    assert pair_indices.shape == (
        num_samples,
        SEQ_LEN,
    ), f"Pair indices shape mismatch. Got {pair_indices.shape}"

    # Target shape: (N, SEQ_LEN, 5) -> Note: padded to SEQ_LEN
    assert targets.shape == (
        num_samples,
        SEQ_LEN,
        len(TARGET_COLS),
    ), f"Target shape mismatch. Got {targets.shape}"

    # Verify target padding
    # The first SEQ_SCORED should be 0.5, the rest 0.0
    assert np.all(
        targets[0, :SEQ_SCORED, 0] == 0.5
    ), "Target values incorrect in scored region."
    assert np.all(
        targets[0, SEQ_SCORED:, 0] == 0.0
    ), "Target values incorrect in padded region."

    print("✓ Data preprocessing shapes and values verified.")

    # Create Dataset and DataLoader
    dataset = RNADataset(processed_data)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    print(f"✓ DataLoader produced batch with keys: {list(batch.keys())}")

    return loader


def demo_model_execution(loader):
    print("\n--- 3. Testing Model Architecture ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Instantiate Model
    model = DeepPostNormBiGRU().to(device)

    # Instantiate Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    # Get a batch
    batch = next(iter(loader))
    inputs = batch["inputs"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    pair_mask = batch["pair_mask"].to(device)
    targets = batch["targets"].to(device)

    print(f"Input batch shape: {inputs.shape}")

    # 1. Forward Pass
    preds = model(inputs, pair_indices, pair_mask)

    # Output shape should be (Batch, Seq_Len, Num_Targets)
    expected_shape = (inputs.shape[0], SEQ_LEN, len(TARGET_COLS))
    assert (
        preds.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {preds.shape}"

    print("✓ Forward pass successful.")

    # 2. Loss Calculation
    loss = MCRMSE(targets, preds)
    print(f"Initial Loss: {loss.item():.4f}")

    # 3. Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("✓ Backward pass and optimizer step successful.")

    # 4. Check StructuralInteractionLayer specifically
    # This layer is critical for the graph-like logic
    print("\n--- 4. Testing Structural Interaction Layer ---")
    hidden_dim = 32
    layer = StructuralInteractionLayer(hidden_dim=hidden_dim).to(device)

    # Create dummy inputs for this layer
    batch_size = 2
    x = torch.randn(batch_size, SEQ_LEN, hidden_dim).to(device)
    # Indices must be valid (0 to SEQ_LEN-1) or 0 with mask 0
    p_idx = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long).to(device)
    p_mask = torch.zeros(batch_size, SEQ_LEN).to(device)

    out = layer(x, p_idx, p_mask)
    assert out.shape == x.shape, "Interaction layer output shape mismatch."
    print("✓ StructuralInteractionLayer forward pass successful.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    try:
        # 1. Utils
        demo_utils_logic()

        # 2. Data
        loader = demo_data_pipeline()

        # 3. Model
        demo_model_execution(loader)

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\n[FAIL] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
