import os
import shutil
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.features import (
    parse_structure,
    get_distance_encoding,
    get_paired_index_map,
)
from library.dataset import RNADataset
from library.model import RNAMultiTaskBiGRU
from library.loss import JointAlignedLoss
from library.train import train_model, predict_and_submit
from library.utils import set_seed


def run_demo():
    # ==============================================================================
    # 1. Configuration Setup
    # ==============================================================================
    print(">>> Setting up Demo Configuration...")

    # Define a specific directory for this demo to avoid conflicts
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Create a config optimized for speed (fewer epochs)
    demo_config = Config(
        WORKING_DIR=demo_dir,
        SUBMISSION_PATH=os.path.join(demo_dir, "submission.csv"),
        EPOCHS=2,  # Reduced for speed
        BATCH_SIZE=16,  # Smaller batch size
        NUM_LAYERS=2,  # Reduced model complexity for speed
        HIDDEN_DIM=64,  # Reduced model size
        EMBED_DIM=32,  # Reduced embedding size
        SEED=42,
    )

    set_seed(demo_config.SEED)
    print(f"    Working Directory: {demo_config.WORKING_DIR}")
    print(f"    Epochs: {demo_config.EPOCHS}")

    # ==============================================================================
    # 2. Verify Feature Engineering Logic
    # ==============================================================================
    print("\n>>> Verifying Feature Engineering (library.features)...")

    # Test Case: Simple hairpin "((..))"
    # Indices: 0 paired with 5, 1 paired with 4. 2,3 are unpaired.
    structure = "((..))"
    seq_len = 6

    # Test parse_structure
    pairs = parse_structure(structure)
    expected_pairs = [
        (1, 4),
        (0, 5),
    ]  # Stack based parsing usually returns inner first or outer first depending on implementation order
    # The provided implementation:
    # 0 '(': stack=[0]
    # 1 '(': stack=[0, 1]
    # 2 '.': skip
    # 3 '.': skip
    # 4 ')': pop 1 -> pair (1, 4)
    # 5 ')': pop 0 -> pair (0, 5)
    # So pairs should be [(1, 4), (0, 5)]
    assert pairs == [(1, 4), (0, 5)], f"Structure parsing failed. Got {pairs}"
    print("    parse_structure: OK")

    # Test get_paired_index_map
    # Index 0 -> 5, 1 -> 4, 2 -> -1, 3 -> -1, 4 -> 1, 5 -> 0
    p_idx = get_paired_index_map(structure, seq_len)
    expected_idx = np.array([5, 4, -1, -1, 1, 0], dtype=np.int32)
    np.testing.assert_array_equal(
        p_idx, expected_idx, err_msg="Paired index map mismatch"
    )
    print("    get_paired_index_map: OK")

    # Test get_distance_encoding
    # 0: (5-0) = 5
    # 1: (4-1) = 3
    # 2: 0
    # 3: 0
    # 4: -(4-1) = -3
    # 5: -(5-0) = -5
    p_dist = get_distance_encoding(structure, seq_len)
    expected_dist = np.array([5, 3, 0, 0, -3, -5], dtype=np.float32)
    np.testing.assert_array_equal(
        p_dist, expected_dist, err_msg="Distance encoding mismatch"
    )
    print("    get_distance_encoding: OK")

    # ==============================================================================
    # 3. Verify Dataset Loading
    # ==============================================================================
    print("\n>>> Verifying Dataset Loading (library.dataset)...")

    # Initialize Train Dataset
    # This will trigger processing from metadata parquet files and caching to .npz
    train_dataset = RNADataset("train", config=demo_config)

    print(f"    Train Dataset Size: {len(train_dataset)}")
    assert len(train_dataset) > 0, "Dataset should not be empty"

    # Fetch one item
    item = train_dataset[0]

    # Verify Keys
    required_keys = ["seq", "loop", "pair_idx", "pair_dist", "mask_labels", "targets"]
    for k in required_keys:
        assert k in item, f"Missing key {k} in dataset item"

    # Verify Shapes
    # Seq length is 107
    L = demo_config.SEQ_LEN
    assert item["seq"].shape == (L,), f"Seq shape mismatch: {item['seq'].shape}"
    assert item["pair_idx"].shape == (L,), f"Pair idx shape mismatch"
    assert item["pair_dist"].shape == (L,), f"Pair dist shape mismatch"

    # Targets should be (68, 5) as per dataset definition (before slicing in model/loss)
    # 68 is seq_scored, 5 is number of provided ground truth columns
    assert item["targets"].shape == (
        68,
        5,
    ), f"Targets shape mismatch: {item['targets'].shape}"

    print("    Dataset shapes verification: OK")

    # ==============================================================================
    # 4. Verify Model and Forward Pass
    # ==============================================================================
    print("\n>>> Verifying Model Architecture (library.model)...")

    device = torch.device("cpu")  # Keep on CPU for simple verification
    model = RNAMultiTaskBiGRU(demo_config).to(device)

    # Create a dummy batch (Batch Size = 2)
    batch_seq = item["seq"].unsqueeze(0).repeat(2, 1).to(device)
    batch_loop = item["loop"].unsqueeze(0).repeat(2, 1).to(device)
    batch_pair_idx = item["pair_idx"].unsqueeze(0).repeat(2, 1).to(device)
    batch_pair_dist = item["pair_dist"].unsqueeze(0).repeat(2, 1).to(device)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        reg_out, recon_out = model(
            batch_seq, batch_loop, batch_pair_idx, batch_pair_dist
        )

    # Verify Output Shapes
    # Regression: (B, L, 3) -> Predicts 3 scored targets for all positions
    assert reg_out.shape == (
        2,
        L,
        3,
    ), f"Regression output shape mismatch: {reg_out.shape}"

    # Reconstruction: (B, L, 4) -> Predicts 4 bases (A, G, C, U)
    assert recon_out.shape == (
        2,
        L,
        4,
    ), f"Reconstruction output shape mismatch: {recon_out.shape}"

    print("    Model forward pass: OK")

    # ==============================================================================
    # 5. Verify Loss Calculation
    # ==============================================================================
    print("\n>>> Verifying Loss Function (library.loss)...")

    criterion = JointAlignedLoss(demo_config)

    # Create dummy targets and mask labels
    batch_targets = (
        item["targets"].unsqueeze(0).repeat(2, 1, 1).to(device)
    )  # (2, 68, 5)
    batch_mask_labels = (
        item["mask_labels"].unsqueeze(0).repeat(2, 1).to(device)
    )  # (2, 107)

    # Calculate Loss
    loss, mse, ce = criterion(reg_out, recon_out, batch_targets, batch_mask_labels)

    print(f"    Total Loss: {loss.item():.4f}")
    print(f"    MSE Loss: {mse.item():.4f}")
    print(f"    CE Loss: {ce.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("    Loss calculation: OK")

    # ==============================================================================
    # 6. Run Training Loop
    # ==============================================================================
    print("\n>>> Running Demo Training (library.train.train_model)...")

    # This runs the training loop using the config we created
    # It will save 'best_model.pth' in demo_dir
    best_score = train_model(config=demo_config)

    print(f"    Training finished. Best MCRMSE: {best_score:.4f}")
    assert os.path.exists(
        os.path.join(demo_dir, "best_model.pth")
    ), "Model checkpoint not saved"

    # ==============================================================================
    # 7. Run Inference and Submission
    # ==============================================================================
    print("\n>>> Running Inference (library.train.predict_and_submit)...")

    predict_and_submit(config=demo_config)

    # Verify Submission File
    sub_path = demo_config.SUBMISSION_PATH
    assert os.path.exists(sub_path), "Submission file not created"

    df_sub = pd.read_csv(sub_path)
    print(f"    Submission file loaded. Shape: {df_sub.shape}")

    # Expected rows: 240 test samples * 107 positions = 25680
    # Note: The test.json provided in the prompt description has 240 lines.
    # The sample_submission has 25680 rows.
    expected_rows = 240 * 107
    # If the test set in ./input/test.json matches the description:
    if len(df_sub) == expected_rows:
        print("    Row count matches expected (25680).")
    else:
        print(f"    Row count {len(df_sub)} (Expected approx {expected_rows}).")

    # Check Columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(df_sub.columns)}"

    # Check if unscored columns are 0
    assert (df_sub["deg_pH10"] == 0).all(), "deg_pH10 should be 0"
    assert (df_sub["deg_50C"] == 0).all(), "deg_50C should be 0"

    print("    Submission verification: OK")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
