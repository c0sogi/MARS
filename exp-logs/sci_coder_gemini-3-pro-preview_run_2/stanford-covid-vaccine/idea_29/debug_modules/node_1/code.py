import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library import config, utils, data, model, train


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Define a separate working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Monkey-patch the configuration to use the demo directory
    config.WORKING_DIR = DEMO_DIR
    config.PATHS["TRAIN_CACHE"] = os.path.join(DEMO_DIR, "train_data.npz")
    config.PATHS["VAL_CACHE"] = os.path.join(DEMO_DIR, "val_data.npz")
    config.PATHS["TEST_CACHE"] = os.path.join(DEMO_DIR, "test_data.npz")
    config.PATHS["SUBMISSION"] = os.path.join(DEMO_DIR, "submission.csv")
    config.PATHS["MODEL_SAVE"] = os.path.join(DEMO_DIR, "best_model.pth")

    # Override training parameters for speed
    config.TRAIN_PARAMS["debug"] = True
    config.TRAIN_PARAMS["max_debug_samples"] = 32  # Small subset
    config.TRAIN_PARAMS["batch_size"] = 4
    config.TRAIN_PARAMS["num_epochs"] = 2
    config.TRAIN_PARAMS["patience"] = 2

    # Set seed for reproducibility
    utils.set_seed(42)
    print("    Configuration updated. Debug mode enabled.")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading and Processing...")

    # Load dataloaders (this will trigger processing and caching)
    train_loader, val_loader, test_loader = data.get_dataloaders(
        debug=config.TRAIN_PARAMS["debug"],
        load_cached_data=False,  # Force reprocessing to test logic
    )

    # Fetch one batch
    inputs, partner_indices, targets, masks, ids = next(iter(train_loader))

    print(f"    Batch shapes -> Inputs: {inputs.shape}, Targets: {targets.shape}")

    # Assertions
    # Inputs from loader should have 18 channels:
    # Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18
    assert (
        inputs.shape[1] == config.DATA_CONFIG["seq_length"]
    ), "Incorrect sequence length"
    assert inputs.shape[2] == 18, f"Expected 18 input channels, got {inputs.shape[2]}"

    # Targets should have 5 channels
    assert targets.shape[2] == 5, "Expected 5 target channels"

    # Partner indices should be integers within range [0, seq_len)
    assert partner_indices.dtype == torch.long, "Partner indices must be LongTensor"
    assert (
        partner_indices.max() < config.DATA_CONFIG["seq_length"]
    ), "Partner index out of bounds"

    print("    Data logic verified.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = utils.get_device()
    net = model.SR_DCN().to(device)

    # Prepare inputs for the model
    # The model expects inputs + recycling channels (5) = 23 channels
    B, L, _ = inputs.shape
    recycling_channels = torch.zeros(B, L, 5).to(device)

    # Concatenate static inputs with recycling channels
    model_inputs = torch.cat([inputs.to(device), recycling_channels], dim=2)

    # Forward pass
    outputs = net(model_inputs, partner_indices.to(device))

    print(f"    Output shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (B, L, 5), "Output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model produced NaNs"

    print("    Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Loss Function (MCRMSE)...")

    criterion = utils.MCRMSELoss()
    # Scored indices are [0, 1, 3] by default (reactivity, deg_Mg_pH10, deg_Mg_50C)

    # Create synthetic data
    # Batch=1, Len=1, Channels=5
    # Mask=1 (valid)
    pred = torch.zeros(1, 1, 5)
    targ = torch.zeros(1, 1, 5)
    mask = torch.ones(1, 1)

    # Introduce error in column 0 (reactivity)
    # Error = 2.0 -> Squared Error = 4.0 -> RMSE = 2.0
    pred[0, 0, 0] = 2.0

    # Introduce error in column 1 (deg_Mg_pH10)
    # Error = 0.0 -> RMSE = 0.0

    # Introduce error in column 3 (deg_Mg_50C)
    # Error = 4.0 -> Squared Error = 16.0 -> RMSE = 4.0
    pred[0, 0, 3] = 4.0

    # Expected MCRMSE = Mean([2.0, 0.0, 4.0]) = 2.0

    loss = criterion(pred, targ, mask)

    print(f"    Calculated Loss: {loss.item():.4f}, Expected: 2.0000")

    # Allow small float error
    assert abs(loss.item() - 2.0) < 1e-5, "Loss calculation incorrect"

    print("    Loss function verified.")

    # -------------------------------------------------------------------------
    # 5. Full Training Loop Integration
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Integration Test)...")

    # This calls the provided train_model function which handles the loop,
    # validation, saving, and submission generation.
    train.train_model()

    print("    Training loop execution complete.")

    # -------------------------------------------------------------------------
    # 6. Output Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Output Files...")

    model_path = config.PATHS["MODEL_SAVE"]
    sub_path = config.PATHS["SUBMISSION"]

    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    # Check submission format
    df_sub = pd.read_csv(sub_path)
    print(f"    Submission shape: {df_sub.shape}")
    print(f"    Submission columns: {df_sub.columns.tolist()}")

    required_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    for col in required_cols:
        assert col in df_sub.columns, f"Missing column {col} in submission"

    # Check if rows match test set expansion
    # We used debug mode with max 32 samples.
    # Test set size in debug is min(240, 32) = 32.
    # Rows = 32 * 107 = 3424
    expected_rows = (
        min(240, config.TRAIN_PARAMS["max_debug_samples"])
        * config.DATA_CONFIG["seq_length"]
    )
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    print("    Output files verified.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
