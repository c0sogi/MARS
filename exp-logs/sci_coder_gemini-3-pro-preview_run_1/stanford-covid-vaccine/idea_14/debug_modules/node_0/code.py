import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import set_seed, mcrmse, save_submission, create_submission_dataframe
from library.dataset import process_dataframe, RNADataset, parse_structure_to_distance
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.train import train_one_epoch, validate


def main():
    print("Starting Library Usage Demonstration...")

    # =========================================================================
    # 1. Configuration Override for Demo
    # =========================================================================
    print("\n[1] Configuring environment for rapid demo...")

    # Modify Config to run a lightweight version
    Config.WORKING_DIR = "./working/demo_run"
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_data_demo.pt")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_data_demo.pt")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Reduce Model Size for Speed
    Config.HIDDEN_DIM = 64
    Config.EMBED_DIM = 32
    Config.N_LAYERS = 2
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1

    # Ensure demo directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set Seed
    set_seed(Config.SEED)
    print("    Configuration updated and seed set.")

    # =========================================================================
    # 2. Verify Utility Functions
    # =========================================================================
    print("\n[2] Verifying Utility Functions...")

    # Test MCRMSE
    # Create dummy ground truth and predictions
    # Shape: (N=2, L=3, C=3)
    y_true = np.array(
        [
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ]
    )
    # Predict exactly same -> error should be 0
    score_perfect = mcrmse(y_true, y_true)
    assert (
        score_perfect == 0.0
    ), f"MCRMSE should be 0.0 for perfect preds, got {score_perfect}"

    # Predict with offset of 1.0 -> RMSE should be 1.0
    y_pred_off = y_true + 1.0
    score_off = mcrmse(y_true, y_pred_off)
    assert np.isclose(score_off, 1.0), f"MCRMSE should be 1.0, got {score_off}"

    print("    MCRMSE metric verified.")

    # =========================================================================
    # 3. Data Pipeline Demonstration
    # =========================================================================
    print("\n[3] Verifying Data Pipeline...")

    # Test Structure Parsing Logic
    # Structure: "(..)" -> 0 pairs with 3 (dist +3), 3 pairs with 0 (dist -3)
    # Indices:    0123
    struct_test = "(..)"
    dists = parse_structure_to_distance(struct_test)
    expected_dists = np.array([3.0, 0.0, 0.0, -3.0], dtype=np.float32)
    np.testing.assert_array_equal(
        dists, expected_dists, err_msg="Structure parsing failed"
    )
    print("    Structure to distance parsing verified.")

    # Load Subset of Data
    if not os.path.exists(Config.TRAIN_FILE):
        raise FileNotFoundError(f"Metadata file not found at {Config.TRAIN_FILE}")

    print(f"    Loading subset from {Config.TRAIN_FILE}...")
    df_full = pd.read_parquet(Config.TRAIN_FILE)
    df_subset = df_full.head(20).copy()  # Take 20 samples

    # Process Dataframe
    print("    Processing dataframe into tensors...")
    data_dict = process_dataframe(df_subset, mode="train")

    # Verify Dictionary Keys
    expected_keys = {"sequence", "loop_type", "pair_dist", "ids", "targets"}
    assert set(data_dict.keys()) == expected_keys, "Processed data missing keys."

    # Verify Tensor Shapes
    # Sequence: (N, 107)
    assert data_dict["sequence"].shape == (20, 107)
    # Targets: (N, 107, 3) - Note: process_dataframe pads targets to seq_len
    assert data_dict["targets"].shape == (20, 107, 3)

    # Create Dataset and Loader
    dataset = RNADataset(data_dict, mode="train")
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Fetch one batch
    batch = next(iter(loader))
    print(f"    DataLoader operational. Batch keys: {list(batch.keys())}")

    # =========================================================================
    # 4. Model & Loss Demonstration
    # =========================================================================
    print("\n[4] Verifying Model and Loss...")

    device = torch.device("cpu")  # Use CPU for demo to avoid CUDA init overhead if any
    model = RNAModel().to(device)

    # Forward Pass
    seq = batch["sequence"].to(device)
    loop = batch["loop_type"].to(device)
    dist = batch["pair_dist"].to(device)
    targets = batch["targets"].to(device)

    preds = model(seq, loop, dist)

    # Check Output Shape: (Batch, Seq_Len, Num_Targets)
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Model output shape mismatch. Got {preds.shape}"
    print("    Model forward pass successful.")

    # Loss Calculation
    criterion = MaskedMSELoss()
    loss = criterion(preds, targets)

    # Loss should be a scalar tensor
    assert loss.ndim == 0, "Loss should be a scalar."
    assert loss.item() >= 0, "Loss should be non-negative."
    print(f"    Loss calculation successful: {loss.item():.6f}")

    # =========================================================================
    # 5. Training Loop Simulation
    # =========================================================================
    print("\n[5] Simulating Training Loop (1 Epoch)...")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run train_one_epoch
    avg_train_loss = train_one_epoch(
        model, loader, criterion, optimizer, device, epoch=1
    )
    print(f"    Training step complete. Avg Loss: {avg_train_loss:.6f}")

    # Run validate
    # Use the same loader for validation to save time
    avg_val_loss, val_score = validate(model, loader, criterion, device)
    print(
        f"    Validation step complete. Val Loss: {avg_val_loss:.6f}, MCRMSE: {val_score:.6f}"
    )

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    print("\n[6] Verifying Submission Generation...")

    # Generate dummy IDs and predictions
    # 5 samples, 107 length, 3 channels
    test_ids = [f"id_{i:05d}" for i in range(5)]
    test_preds = np.random.rand(5, 107, 3).astype(np.float32)

    # Create DataFrame
    sub_df = create_submission_dataframe(test_ids, test_preds)

    # Check Columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch."

    # Check Rows: 5 samples * 107 positions = 535 rows
    assert (
        len(sub_df) == 5 * 107
    ), f"Submission row count mismatch. Expected 535, got {len(sub_df)}"

    # Save Submission
    save_submission(test_ids, test_preds, Config.SUBMISSION_PATH)
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
