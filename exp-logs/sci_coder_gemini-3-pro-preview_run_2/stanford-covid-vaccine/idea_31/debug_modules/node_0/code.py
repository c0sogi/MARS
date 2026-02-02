import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import components from the provided library
from library.config import Config
from library.dataset import RNADataset
from library.model import SR_DCN
from library.loss import MCRMSELoss
from library.engine import RNAEngine, set_seed


def run_demo():
    print(">>> Starting SR-DCN Demo Execution")

    # =========================================================================
    # 1. Configuration Override for Demo
    # =========================================================================
    # We modify the Config class attributes directly to isolate the demo environment
    # and speed up execution.
    print("Configuring environment...")

    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Redirect cache files to the demo directory
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_data.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_data.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_data.npz")

    # Redirect submission output
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Reduce hyperparameters for rapid demonstration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.PATIENCE = 2

    # Ensure reproducibility
    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")

    # =========================================================================
    # 2. Data Loading & Verification
    # =========================================================================
    print("\n>>> Loading Datasets...")

    # Load datasets (this triggers data_utils.process_data)
    # We force load_cached_data=False to demonstrate processing logic,
    # though in practice True is preferred.
    train_dataset = RNADataset(mode="train", load_cached_data=False)
    val_dataset = RNADataset(mode="val", load_cached_data=False)

    # Verify dataset sizes
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Assertions to ensure data loaded correctly
    assert len(train_dataset) > 0, "Train dataset should not be empty."
    assert len(val_dataset) > 0, "Val dataset should not be empty."

    # Inspect a single sample
    inputs, partner_idx, targets, sample_id = train_dataset[0]

    print(f"Sample ID: {sample_id}")
    print(f"Input Shape: {inputs.shape} (Expected: {Config.SEQ_LEN}, 18)")
    print(f"Target Shape: {targets.shape} (Expected: {Config.SEQ_LEN}, 5)")

    # Validate shapes
    # Input channels: 4(Seq) + 3(Struct) + 7(Loop) + 4(PartnerID) = 18
    assert inputs.shape == (Config.SEQ_LEN, 18), "Incorrect input tensor shape."
    assert partner_idx.shape == (Config.SEQ_LEN,), "Incorrect partner index shape."
    assert targets.shape == (Config.SEQ_LEN, 5), "Incorrect target tensor shape."

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # =========================================================================
    # 3. Model Architecture & Forward Pass Verification
    # =========================================================================
    print("\n>>> Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SR_DCN().to(device)

    # Prepare a batch for testing
    b_inputs, b_partner_idx, b_targets, _ = next(iter(train_loader))
    b_inputs = b_inputs.to(device)
    b_partner_idx = b_partner_idx.to(device)
    b_targets = b_targets.to(device)

    # Test Pass 1 (Cold Start)
    preds_1 = model(b_inputs, b_partner_idx, recycling=None)
    print(f"Prediction Shape (Pass 1): {preds_1.shape}")

    assert preds_1.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Model output shape mismatch."

    # Test Pass 2 (Recycling)
    # We use the detached output of Pass 1 as input for Pass 2
    preds_2 = model(b_inputs, b_partner_idx, recycling=preds_1.detach())
    assert preds_2.shape == preds_1.shape, "Recycling pass output shape mismatch."

    # =========================================================================
    # 4. Loss Function Verification
    # =========================================================================
    print("\n>>> Verifying MCRMSE Loss...")

    criterion = MCRMSELoss().to(device)
    loss = criterion(preds_2, b_targets)

    print(f"Computed Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() >= 0, "Loss must be non-negative."

    # =========================================================================
    # 5. Training Loop (Engine)
    # =========================================================================
    print("\n>>> Executing Training Loop...")

    engine = RNAEngine(device=device)

    # Run training
    # This will train for Config.EPOCHS (2) and save 'best_model.pth'
    best_score = engine.run_training(train_loader, val_loader, epochs=Config.EPOCHS)

    print(f"Training completed. Best Validation MCRMSE: {best_score:.6f}")

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved."

    # =========================================================================
    # 6. Inference & Submission
    # =========================================================================
    print("\n>>> Generating Submission...")

    # Load Test Data
    test_dataset = RNADataset(mode="test", load_cached_data=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Generate predictions
    engine.generate_submission(test_loader)

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")

    # Expected rows: 240 test samples * 107 sequence length = 25680
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Expected columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    print("Submission file validation passed.")
    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
