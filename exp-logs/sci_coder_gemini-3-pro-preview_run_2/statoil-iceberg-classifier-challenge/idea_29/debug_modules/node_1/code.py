import os
import torch
import numpy as np
import pandas as pd
import sys

# Import provided library modules
from library import config, utils, data_loader, model, train


def run_demo():
    print("=" * 50)
    print("STARTING DEMO EXECUTION")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. SETUP & CONFIGURATION OVERRIDES
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Override config for speed and isolation
    config.WORKING_DIR = "./working/demo_execution"
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")
    config.SUBMISSION_DIR = os.path.join(config.WORKING_DIR)
    config.SUBMISSION_FILE = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    config.DEBUG = True
    config.DEBUG_SIZE = 50  # Small subset for speed
    config.EPOCHS = 1  # Only 1 epoch for demo
    config.N_FOLDS = 2  # Only 2 folds to verify CV loop
    config.BATCH_SIZE = 8

    # Re-create directories based on new config
    config.setup_directories()

    # Set seed
    utils.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Working Directory: {config.WORKING_DIR}")
    print(f"    Device: {device}")

    # ---------------------------------------------------------
    # 2. DATA LOADER VERIFICATION
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Test process_data
    # We force load_cached_data=False to ensure processing logic runs,
    # but since we are in debug mode, it will be fast.
    data_dict = data_loader.process_data(load_cached_data=False)

    # Verify keys
    expected_keys = [
        "X_train",
        "y_train",
        "inc_train",
        "X_val",
        "y_val",
        "inc_val",
        "X_test",
        "inc_test",
        "ch_mins",
        "ch_maxs",
    ]
    for k in expected_keys:
        assert k in data_dict, f"Missing key in processed data: {k}"

    # Verify shapes (Train set)
    # Note: process_data returns full sets, get_loaders handles debug slicing
    X_train_full = data_dict["X_train"]
    print(f"    Full Train Data Shape: {X_train_full.shape}")
    assert len(X_train_full.shape) == 4, "X_train should be 4D (N, C, H, W)"
    assert X_train_full.shape[1] == 3, "Should have 3 channels"
    assert (
        X_train_full.shape[2] == 75 and X_train_full.shape[3] == 75
    ), "Image size should be 75x75"

    # Test Dataset and DataLoader
    print("    Initializing DataLoaders (Debug Mode)...")
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        batch_size=config.BATCH_SIZE, debug=True, load_cached_data=True
    )

    # Fetch one batch to verify
    batch = next(iter(train_loader))
    imgs, incs, labels = batch

    print(
        f"    Batch Shapes -> Imgs: {imgs.shape}, Incs: {incs.shape}, Labels: {labels.shape}"
    )

    # Assertions
    assert (
        imgs.shape[0] == config.BATCH_SIZE or imgs.shape[0] == config.DEBUG_SIZE
    ), f"Batch size mismatch. Expected <= {config.BATCH_SIZE}, got {imgs.shape[0]}"
    assert imgs.shape[1] == 3, "Batch images should have 3 channels"
    assert incs.shape[1] == 1, "Incidence angles should be (Batch, 1)"
    assert labels.shape[1] == 1, "Labels should be (Batch, 1)"

    print("    Data Loading verification passed.")

    # ---------------------------------------------------------
    # 3. MODEL VERIFICATION
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    net = model.WBDIN().to(device)

    # Create dummy inputs
    dummy_img = torch.randn(config.BATCH_SIZE, 3, 75, 75).to(device)
    dummy_inc = torch.randn(config.BATCH_SIZE, 1).to(device)

    # Forward pass
    output = net(dummy_img, dummy_inc)

    print(f"    Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected ({config.BATCH_SIZE}, 1), got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("    Model verification passed.")

    # ---------------------------------------------------------
    # 4. TRAINING LOOP COMPONENT VERIFICATION
    # ---------------------------------------------------------
    print("\n[4] Verifying Training Steps...")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train one epoch
    print("    Running train_one_epoch...")
    train_loss = train.train_one_epoch(net, train_loader, criterion, optimizer, device)
    print(f"    Train Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"

    # Validate one epoch
    print("    Running validate_one_epoch...")
    val_loss = train.validate_one_epoch(net, val_loader, criterion, device)
    print(f"    Val Loss: {val_loss:.4f}")
    assert isinstance(val_loss, float), "Val loss should be a float"

    print("    Training steps verification passed.")

    # ---------------------------------------------------------
    # 5. FULL PIPELINE EXECUTION
    # ---------------------------------------------------------
    print("\n[5] Executing Full Training Pipeline (Simulated)...")
    print("    This will run the stratified K-Fold loop with reduced epochs/data.")

    # We call the main run_training function.
    # It uses the config values we overrode earlier (N_FOLDS=2, EPOCHS=1, DEBUG=True).
    train.run_training()

    # Verify submission file
    if os.path.exists(config.SUBMISSION_FILE):
        df_sub = pd.read_csv(config.SUBMISSION_FILE)
        print(f"    Submission file generated at: {config.SUBMISSION_FILE}")
        print(f"    Submission shape: {df_sub.shape}")
        print(f"    First 3 rows:\n{df_sub.head(3)}")

        # Assertions
        assert (
            "id" in df_sub.columns and "is_iceberg" in df_sub.columns
        ), "Submission file missing required columns"
        assert len(df_sub) > 0, "Submission file is empty"

        # In debug mode, we slice the test set too.
        # config.DEBUG_SIZE is 50. So submission should have min(50, n_test) rows.
        # The test set has 321 rows. So we expect 50 rows.
        assert (
            len(df_sub) == config.DEBUG_SIZE
        ), f"Expected {config.DEBUG_SIZE} rows in debug submission, got {len(df_sub)}"

        print("    Pipeline execution verification passed.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n" + "=" * 50)
    print("DEMO EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
