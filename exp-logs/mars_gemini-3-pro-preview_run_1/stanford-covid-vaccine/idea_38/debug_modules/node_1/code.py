import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.data import (
    get_dataloaders,
    parse_structure_to_pairs,
    get_pair_distance_vector,
)
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.train import Trainer


def main():
    print("=== Starting Demo Execution ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override paths to use a separate demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Override training parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Use main process for simplicity in demo

    # Setup directories based on new config
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test MCRMSE Calculation
    # Scenario: 2 samples, 5 positions, 3 targets.
    # Sample 1: Error 0.1 on all targets.
    # Sample 2: Error 0.1 on all targets.
    # RMSE per column should be 0.1. Mean of RMSEs should be 0.1.
    y_true_dummy = np.zeros((2, 5, 3))
    y_pred_dummy = np.ones((2, 5, 3)) * 0.1
    score = calculate_mcrmse(y_true_dummy, y_pred_dummy)

    assert np.isclose(
        score, 0.1
    ), f"MCRMSE Verification Failed. Expected 0.1, got {score}"
    print("MCRMSE function verified.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Processing Logic
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Processing Logic...")

    # Test Structure Parsing
    # Structure: .((..)).  Length 8
    # Pairs: (1, 6), (2, 5). Indices are 0-based.
    # Index 1 '(' pairs with Index 6 ')'
    # Index 2 '(' pairs with Index 5 ')'
    test_struct = ".((..))."
    pairs = parse_structure_to_pairs(test_struct)

    assert pairs[1] == 6 and pairs[6] == 1, "Pair parsing failed for outer pair"
    assert pairs[2] == 5 and pairs[5] == 2, "Pair parsing failed for inner pair"
    assert 0 not in pairs, "Unpaired base incorrectly paired"

    # Test Distance Vector
    # Index 1: paired with 6 -> dist 6-1 = 5
    # Index 6: paired with 1 -> dist 1-6 = -5
    dists = get_pair_distance_vector(test_struct, len(test_struct))
    assert dists[1] == 5.0, f"Distance vector failed. Expected 5.0, got {dists[1]}"
    assert dists[6] == -5.0, f"Distance vector failed. Expected -5.0, got {dists[6]}"
    assert dists[0] == 0.0, "Unpaired distance should be 0"

    print("Structure parsing and distance vector generation verified.")

    # -------------------------------------------------------------------------
    # 4. Data Loading
    # -------------------------------------------------------------------------
    print("\n[4] Loading Data (forcing re-process from metadata)...")

    # We force load_cached_data=False to ensure the processing pipeline runs
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Batch
    batch = next(iter(train_loader))
    seqs = batch["sequence"]
    targets = batch["target"]
    ids = batch["id"]

    print(f"Loaded batch of size {len(ids)}")
    print(f"Sequence Shape: {seqs.shape}")
    print(f"Target Shape: {targets.shape}")

    assert seqs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), "Incorrect sequence tensor shape"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "Incorrect target tensor shape"
    assert not torch.isnan(targets).any(), "Targets contain NaNs"

    print("DataLoaders initialized and batch structure verified.")

    # -------------------------------------------------------------------------
    # 5. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[5] Initializing Model & Running Forward Pass...")

    model = RNAModel(config=Config)
    model.to(Config.DEVICE)

    # Move batch to device
    b_seq = seqs.to(Config.DEVICE)
    b_loop = batch["loop_type"].to(Config.DEVICE)
    b_dist = batch["pair_dist"].to(Config.DEVICE)

    # Forward
    preds = model(b_seq, b_loop, b_dist)

    print(f"Prediction Shape: {preds.shape}")
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "Model output shape mismatch"

    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 6. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying MaskedMSELoss...")

    criterion = MaskedMSELoss(seq_scored=Config.SEQ_SCORED)

    # Case 1: Error only in unscored region (index >= 68)
    # Should result in 0 loss
    t_dummy = torch.zeros(2, 107, 3)
    p_dummy_unscored = torch.zeros(2, 107, 3)
    p_dummy_unscored[:, 70:, :] = 100.0  # Large error in unscored tail

    loss_unscored = criterion(p_dummy_unscored, t_dummy)
    assert (
        loss_unscored.item() == 0.0
    ), f"Loss should be 0 for unscored errors, got {loss_unscored.item()}"

    # Case 2: Error in scored region
    # Set error of 1.0 at index 0 for all targets/samples
    p_dummy_scored = torch.zeros(2, 107, 3)
    p_dummy_scored[:, 0, :] = 1.0

    # MSE = Sum(Errors^2) / N
    # Errors are 1.0 for: 2 samples * 1 pos * 3 targets = 6 elements
    # Total elements in mean: 2 samples * 68 pos * 3 targets = 408 elements
    # Expected Loss = 6 / 408
    expected_loss = 6.0 / (2 * 68 * 3)
    loss_scored = criterion(p_dummy_scored, t_dummy)

    assert torch.isclose(
        loss_scored, torch.tensor(expected_loss), atol=1e-6
    ), f"Loss calculation mismatch. Expected {expected_loss}, got {loss_scored.item()}"

    print("MaskedMSELoss verified.")

    # -------------------------------------------------------------------------
    # 7. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[7] Executing Training Loop (1 Epoch)...")

    trainer = Trainer(model, train_loader, val_loader, test_loader, Config)

    # Train one epoch
    train_loss = trainer.train_one_epoch(0)
    print(f"Training Loss: {train_loss:.6f}")

    # Validate
    val_mcrmse = trainer.validate()
    print(f"Validation MCRMSE: {val_mcrmse:.6f}")

    # Manually save model since we are running steps individually instead of trainer.fit()
    if val_mcrmse < trainer.best_mcrmse:
        trainer.best_mcrmse = val_mcrmse
        torch.save(trainer.model.state_dict(), Config.MODEL_SAVE_PATH)

    # Check if model saved (Trainer logic saves if better than inf, which it should be)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training and Validation cycle completed.")

    # -------------------------------------------------------------------------
    # 8. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[8] Generating Submission...")

    trainer.generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Rows: {len(sub_df)}")
    print(f"Submission File Columns: {list(sub_df.columns)}")

    # Expected rows: 240 test samples * 107 positions
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Check for required columns
    required_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column {col} in submission"

    print("Submission generated and verified.")
    print("\n=== Demo Execution Finished Successfully ===")


if __name__ == "__main__":
    main()
