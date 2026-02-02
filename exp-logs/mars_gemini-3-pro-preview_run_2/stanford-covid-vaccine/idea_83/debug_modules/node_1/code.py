import os
import shutil
import numpy as np
import pandas as pd
import torch

# Import from provided libraries
from library.config import Config
from library.utils import set_seed
from library.data import get_loader
from library.model import HC_HIGFN
from library.loss import MaskedMCRMSE
from library.train import run_training


def main():
    print("Initializing Demonstration Script...")

    # =========================================================================
    # 1. Configuration Override for Demo
    # =========================================================================
    # We modify the Config class attributes directly to isolate this run
    # and ensure it executes quickly.

    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update Config paths
    Config.IDEA_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.METADATA_DIR = "./metadata"  # Ensure we point to existing metadata

    os.makedirs(Config.IDEA_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    Config.MODEL_SAVE_PATH = os.path.join(Config.IDEA_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.PATIENCE = 1
    Config.CACHE_VERSION = "demo_v1"  # Force new cache generation for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print(f"Config updated. Working directory: {DEMO_DIR}")

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # =========================================================================
    # 2. Data Pipeline Verification
    # =========================================================================
    print("\n--- Verifying Data Pipeline ---")

    # Load Train Loader
    train_loader = get_loader(
        "train", batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    # Fetch one batch
    inputs, partner_indices, targets = next(iter(train_loader))

    # Assert Shapes
    # Inputs: (Batch, Seq_Len, Channels=18)
    # Channels: 4(Seq) + 3(Struct) + 7(Loop) + 4(Partner) = 18
    print(f"Input shape: {inputs.shape}")
    assert inputs.shape == (
        Config.BATCH_SIZE,
        107,
        18,
    ), f"Expected input shape ({Config.BATCH_SIZE}, 107, 18), got {inputs.shape}"

    # Partner Indices: (Batch, Seq_Len)
    print(f"Partner indices shape: {partner_indices.shape}")
    assert partner_indices.shape == (
        Config.BATCH_SIZE,
        107,
    ), f"Expected partner indices shape ({Config.BATCH_SIZE}, 107), got {partner_indices.shape}"

    # Targets: (Batch, Seq_Len, 5)
    print(f"Targets shape: {targets.shape}")
    assert targets.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Expected targets shape ({Config.BATCH_SIZE}, 107, 5), got {targets.shape}"

    print("Data Pipeline verification passed.")

    # =========================================================================
    # 3. Model Architecture Verification
    # =========================================================================
    print("\n--- Verifying Model Architecture ---")

    model = HC_HIGFN().to(device)
    model.train()  # Set to train mode

    # Move batch to device
    inputs = inputs.to(device)
    partner_indices = partner_indices.to(device)
    targets = targets.to(device)

    # Forward Pass
    y_final, y_aux = model(inputs, partner_indices)

    # Assert Output Shapes (Batch, Seq_Len, 5)
    print(f"Model Output (Final) shape: {y_final.shape}")
    assert y_final.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 107, 5), got {y_final.shape}"

    assert y_aux.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), "Auxiliary output shape mismatch."

    print("Model Architecture verification passed.")

    # =========================================================================
    # 4. Loss Function Verification
    # =========================================================================
    print("\n--- Verifying Loss Function ---")

    criterion = MaskedMCRMSE()
    loss = criterion(y_final, targets)

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("Loss Function verification passed.")

    # =========================================================================
    # 5. Full Training Loop & Submission Generation
    # =========================================================================
    print("\n--- Executing Training Loop (1 Epoch) ---")

    # We use the provided run_training function.
    # By setting debug=True, it limits epochs (though we set Config.EPOCHS=1 anyway).
    # It will train, validate, save the best model, and generate a submission.
    run_training(debug=True)

    # =========================================================================
    # 6. Submission Validation
    # =========================================================================
    print("\n--- Validating Submission File ---")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    # Expected rows: 240 test samples * 107 positions = 25680
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Expected columns
    expected_cols = ["id_seqpos"] + Config.ALL_TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Column mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check for NaN values
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("Submission file validation passed.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
