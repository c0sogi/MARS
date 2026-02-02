import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import provided library modules
from library import config, utils, data, model, loss, train


def main():
    print("=== Starting Demonstration of RNA Degradation Prediction Pipeline ===\n")

    # 1. Configuration Overrides for Speed
    # We override constants in the config module to ensure the demo runs quickly.
    print("[1] Configuring hyperparameters for rapid demonstration...")
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 8  # Smaller batch size for demonstration
    config.EARLY_STOPPING_PATIENCE = 1

    # Ensure reproducibility
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"    Device: {device}")
    print(f"    Batch Size: {config.BATCH_SIZE}")
    print(f"    Epochs: {config.NUM_EPOCHS}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")
    # We load cached data if available, otherwise process from metadata
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=True)

    # Fetch a single batch to verify shapes
    x_batch, p_idx_batch, y_batch = next(iter(train_loader))

    # Expected shapes:
    # x: (Batch, Seq_Len=107, Channels=18)
    # p_idx: (Batch, Seq_Len=107)
    # y: (Batch, Seq_Len=107, Targets=5)
    print(f"    Input shape: {x_batch.shape}")
    print(f"    Partner Index shape: {p_idx_batch.shape}")
    print(f"    Target shape: {y_batch.shape}")

    assert x_batch.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
        18,
    ), "Incorrect Input Shape"
    assert p_idx_batch.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
    ), "Incorrect Partner Index Shape"
    assert y_batch.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
        5,
    ), "Incorrect Target Shape"
    print("    Data shapes verified successfully.")

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture (HC-HSGFN)...")
    net = model.HCHSGFN().to(device)

    # Move batch to device
    x_dev = x_batch.to(device)
    p_idx_dev = p_idx_batch.to(device)

    # Pass 1: Initial prediction (No feedback)
    print("    Running Pass 1 (Zero Feedback)...")
    pred1 = net(x_dev, p_idx_dev, y_prev=None)
    assert pred1.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
        5,
    ), "Pass 1 Output Shape Mismatch"

    # Pass 2: Refinement (With feedback)
    print("    Running Pass 2 (Iterative Refinement)...")
    pred2 = net(x_dev, p_idx_dev, y_prev=pred1.detach())
    assert pred2.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
        5,
    ), "Pass 2 Output Shape Mismatch"

    print("    Model forward passes verified successfully.")

    # 4. Loss Function Verification
    print("\n[4] Verifying Masked MCRMSE Loss...")
    criterion = loss.MaskedMCRMSELoss().to(device)
    y_dev = y_batch.to(device)

    # Calculate loss
    loss_val = criterion(pred2, y_dev)
    print(f"    Calculated Loss: {loss_val.item():.6f}")

    assert torch.is_tensor(loss_val), "Loss is not a tensor"
    assert loss_val.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss_val), "Loss is NaN"
    print("    Loss function verified successfully.")

    # 5. Training Loop Demonstration
    print("\n[5] Demonstrating Training Loop (1 Epoch)...")
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)

    # We use the provided train_one_epoch function
    epoch_loss = train.train_one_epoch(net, train_loader, optimizer, criterion, device)
    print(f"    Epoch 1 Training Loss: {epoch_loss:.6f}")

    # Validate
    val_score = train.validate(net, val_loader, device)
    print(f"    Validation MCRMSE: {val_score:.6f}")

    # Save this "best" model for the submission step
    torch.save(
        net.state_dict(), os.path.join(config.WORKING_DIR, "demo_best_model.pth")
    )
    print("    Training cycle completed.")

    # 6. Submission Generation Verification
    print("\n[6] Verifying Submission Generation...")
    submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    # Generate submission using the trained model
    train.generate_submission(net, test_loader, device, submission_path)

    # Verify the output file
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"    Submission file created at: {submission_path}")
    print(f"    Submission shape: {df_sub.shape}")
    print(f"    Columns: {df_sub.columns.tolist()}")

    # Check row count: 240 test samples * 107 positions = 25680 rows
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    print("    Submission format verified successfully.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
