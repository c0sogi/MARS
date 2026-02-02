import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import provided library modules
from library import config
from library import utils
from library import data
from library import model


def main():
    print("==== Starting Demonstration Script ====")

    # 1. Setup and Seeding
    utils.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Create a Mini Dataset for Speed
    # We extract a small subset from the provided metadata to avoid processing the full dataset.
    mini_train_path = os.path.join(config.WORKING_DIR, "mini_train.csv")
    mini_cache_path = os.path.join(config.WORKING_DIR, "mini_train_cache.npz")

    print(f"\nCreating mini dataset at {mini_train_path}...")
    # Load original train metadata
    df_full = pd.read_csv(config.TRAIN_CSV)
    # Take 16 samples (matches config.BATCH_SIZE)
    df_mini = df_full.head(16)
    df_mini.to_csv(mini_train_path, index=False)

    # 3. Data Preprocessing
    print("Testing data preprocessing pipeline...")
    # Force re-computation by setting load_cached_data=False
    features, p_idx, p_mask, targets, ids = data.preprocess_data(
        csv_path=mini_train_path,
        cache_path=mini_cache_path,
        load_cached_data=False,
        is_test=False,
    )

    # Verify Data Shapes
    # Expected: (NumSamples, SeqLen, EmbedDim)
    expected_feat_shape = (16, config.SEQ_LENGTH, config.EMBED_DIM)
    assert (
        features.shape == expected_feat_shape
    ), f"Feature shape mismatch. Expected {expected_feat_shape}, got {features.shape}"

    # Expected: (NumSamples, SeqLen, NumTargets)
    expected_target_shape = (16, config.SEQ_LENGTH, len(config.TARGET_COLS))
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

    print("Data preprocessing verified successfully.")

    # 4. DataLoader Setup
    dataset = data.RNADataset(features, p_idx, p_mask, targets, ids)
    # Use a small batch size for the demo
    batch_size = 4
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    # 5. Model Initialization
    print("\nInitializing SCR_DN model...")
    net = model.SCR_DN().to(device)
    net.train()

    # 6. Forward Pass
    print("Executing forward pass...")
    batch = next(iter(dataloader))
    x, partner_indices, pairing_mask, y = [t.to(device) for t in batch]

    # The model returns a list of outputs (one for each recycling pass)
    outputs = net(x, partner_indices, pairing_mask)

    # Verify Model Outputs
    assert isinstance(outputs, list), "Model output should be a list of predictions."
    # Config defines NUM_PASSES=2, so loop runs range(2)
    assert len(outputs) == 2, f"Expected 2 output passes, got {len(outputs)}"

    final_pred = outputs[-1]
    expected_out_shape = (batch_size, config.SEQ_LENGTH, len(config.TARGET_COLS))
    assert (
        final_pred.shape == expected_out_shape
    ), f"Prediction shape mismatch. Expected {expected_out_shape}, got {final_pred.shape}"

    print("Forward pass verified successfully.")

    # 7. Training Step Simulation
    print("Simulating training step (Loss & Backward)...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    optimizer.zero_grad()
    loss = criterion(final_pred, y)
    print(f"Calculated Loss: {loss.item():.6f}")

    loss.backward()
    optimizer.step()
    print("Backward pass and optimization step completed.")

    # 8. Metric Calculation
    print("\nVerifying Metric (MCRMSE)...")
    # Calculate metric on the batch
    score = utils.mcrmse(y, final_pred)
    print(f"Batch MCRMSE Score: {score:.6f}")

    # Manual Logic Verification for MCRMSE
    # Create dummy tensors where prediction is exactly +1.0 off for scored columns
    # Scored columns in config: reactivity, deg_Mg_pH10, deg_Mg_50C
    # These correspond to indices 0, 1, 3 in TARGET_COLS
    dummy_true = torch.zeros((2, config.SEQ_LENGTH, 5))
    dummy_pred = torch.zeros((2, config.SEQ_LENGTH, 5))

    scored_indices = [0, 1, 3]  # Indices for reactivity, deg_Mg_pH10, deg_Mg_50C
    for idx in scored_indices:
        dummy_pred[:, :, idx] = 1.0

    # RMSE for each scored column should be sqrt(1^2) = 1.0. Mean of RMSEs = 1.0.
    dummy_score = utils.mcrmse(dummy_true, dummy_pred)
    assert (
        abs(dummy_score - 1.0) < 1e-5
    ), f"Metric logic check failed. Expected 1.0, got {dummy_score}"
    print("Metric logic verified successfully.")

    # 9. Submission Formatting
    print("\nGenerating sample submission...")
    # Create dummy test IDs corresponding to the batch size
    test_ids = [f"id_demo_{i}" for i in range(batch_size)]

    # Detach predictions to numpy
    preds_np = final_pred.detach().cpu().numpy()

    submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    utils.format_submission(test_ids, preds_np, submission_path)

    # Verify Submission File
    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"Submission file created at {submission_path}")
        print(f"Shape: {sub_df.shape}")

        # Expected rows: batch_size * SEQ_LENGTH
        expected_rows = batch_size * config.SEQ_LENGTH
        assert (
            len(sub_df) == expected_rows
        ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

        # Expected columns: id_seqpos + TARGET_COLS
        expected_cols = ["id_seqpos"] + config.TARGET_COLS
        assert (
            list(sub_df.columns) == expected_cols
        ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

        # Check ID format
        assert (
            sub_df.iloc[0]["id_seqpos"] == "id_demo_0_0"
        ), "Submission ID format appears incorrect."

        print("Submission format verified successfully.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
