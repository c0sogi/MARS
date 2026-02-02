import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.data import get_loaders
from library.model import GatedDenseNet
from library.engine import fit, predict, generate_submission_csv


def run_demo():
    print("=== Starting RNA Degradation Prediction Demo ===")

    # 1. Configure for Speed and Demonstration
    # We modify the Config global state to run a fast, small-scale experiment.
    print("Configuring experiment settings...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.WORKING_DIR = "./working/demo_execution"

    # Update paths based on new working dir
    Config.setup_directories()
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Force re-processing of data to demonstrate the pipeline (disable cache loading)
    # Note: We use unique cache names for the demo to avoid conflicts
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_data_demo_v1.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_data_demo_v1.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_data_demo_v1.npz")

    # 2. Data Loading
    print("\nLoading and processing data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # 3. Verify Data Shapes
    print("Verifying data shapes...")
    # Fetch one batch
    inputs, partner_indices, targets, ids = next(iter(train_loader))

    # Expected shapes:
    # inputs: (Batch, 18, 107) -> 18 channels (4 seq + 3 struct + 7 loop + 4 partner)
    # partner_indices: (Batch, 107)
    # targets: (Batch, 107, 5)

    print(f"Input shape: {inputs.shape}")
    print(f"Partner Indices shape: {partner_indices.shape}")
    print(f"Targets shape: {targets.shape}")

    assert inputs.shape == (
        Config.BATCH_SIZE,
        18,
        107,
    ), f"Expected input shape ({Config.BATCH_SIZE}, 18, 107), got {inputs.shape}"
    assert partner_indices.shape == (
        Config.BATCH_SIZE,
        107,
    ), f"Expected partner indices shape ({Config.BATCH_SIZE}, 107), got {partner_indices.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Expected targets shape ({Config.BATCH_SIZE}, 107, 5), got {targets.shape}"

    print("Data verification passed.")

    # 4. Model Initialization
    print("\nInitializing model...")
    device = Config.DEVICE
    model = GatedDenseNet().to(device)

    # Quick forward pass check
    with torch.no_grad():
        dummy_out = model(inputs.to(device), partner_indices.to(device))
        assert dummy_out.shape == (
            Config.BATCH_SIZE,
            107,
            5,
        ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 107, 5), got {dummy_out.shape}"
    print("Model initialized and forward pass verified.")

    # 5. Training
    print("\nStarting training loop...")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    best_metric = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_PATH,
    )

    print(f"Training finished. Best MCRMSE: {best_metric}")

    # Verify model file exists
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."

    # 6. Inference
    print("\nRunning inference on test set...")
    # Load best model weights
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    preds, test_ids = predict(model, test_loader, device)

    print(f"Predictions shape: {preds.shape}")
    print(f"Number of test IDs: {len(test_ids)}")

    # Expected predictions shape: (N_test_samples, 107, 5)
    # Since we used DEBUG=True, N_test_samples should be min(total_test, DEBUG_SUBSET_SIZE)
    # The test set has 240 samples, DEBUG_SUBSET_SIZE is 50.
    expected_samples = min(240, Config.DEBUG_SUBSET_SIZE)

    assert preds.shape == (
        expected_samples,
        107,
        5,
    ), f"Prediction shape mismatch. Expected ({expected_samples}, 107, 5), got {preds.shape}"

    # 7. Submission Generation
    print("\nGenerating submission file...")
    generate_submission_csv(preds, test_ids, Config.SUBMISSION_PATH)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    # Check row count: N_samples * Seq_Len (107)
    expected_rows = expected_samples * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    run_demo()
