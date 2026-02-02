import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed, mcrmse, parse_structure_to_pairs
from library.data import get_dataloaders, RNADataset, process_data
from library.model import HCTDBiGRU
from library.layers import DisentangledInteraction
from library.train import train_epoch, validate


def main():
    print("==== Starting RNA Degradation Library Demo ====")

    # 1. Setup and Configuration Override for Demo Speed
    # We modify the Config class attributes directly to run a fast, lightweight demo.
    print("\n[1] Configuring environment...")

    # Set paths to a specific demo directory in working
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Reduce compute load
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SUBSET_SIZE = 16  # Small subset for quick execution
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.HIDDEN_DIM = 64  # Smaller model for demo
    Config.STEM_FILTERS = 32

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # 2. Verify Utility Logic (Structure Parsing)
    print("\n[2] Verifying Structure Parsing Logic...")
    test_structure = "..((...)).."
    # Expected for "..((...))..":
    # Indices 0,1 unpaired -> self.
    # Index 2 '(' pairs with 8 ')'.
    # Index 3 '(' pairs with 7 ')'.
    # Indices 4,5,6 unpaired -> self.
    # Indices 9,10 unpaired -> self.

    p_idx, p_mask = parse_structure_to_pairs(test_structure)

    print(f"    Structure: {test_structure}")
    print(f"    Indices:   {p_idx}")
    print(f"    Mask:      {p_mask}")

    # Assertions
    assert len(p_idx) == len(test_structure)
    assert p_idx[2] == 8 and p_idx[8] == 2, "Pairing logic failed for outer bracket"
    assert p_idx[3] == 7 and p_idx[7] == 3, "Pairing logic failed for inner bracket"
    assert p_mask[2] == 1.0 and p_mask[0] == 0.0, "Mask logic failed"
    print("    Structure parsing verification passed.")

    # 3. Data Loading
    print("\n[3] Initializing DataLoaders (Debug Mode)...")
    # We force `load_cached_data=False` to demonstrate processing from metadata
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
        debug=True,
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    inputs = batch["inputs"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    pair_mask = batch["pair_mask"].to(device)
    targets = batch["targets"].to(device)

    print(f"    Batch keys: {list(batch.keys())}")
    print(
        f"    Inputs shape: {inputs.shape} (Expected: {Config.BATCH_SIZE}, {Config.SEQ_LEN}, {Config.INPUT_DIM})"
    )
    print(
        f"    Targets shape: {targets.shape} (Expected: {Config.BATCH_SIZE}, {Config.SEQ_LEN}, {Config.NUM_TARGETS})"
    )

    assert inputs.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)
    assert pair_indices.shape == (Config.BATCH_SIZE, Config.SEQ_LEN)
    assert targets.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    print("    DataLoader verification passed.")

    # 4. Model Instantiation and Forward Pass
    print("\n[4] Instantiating HC-TD-BiGRU Model...")
    model = HCTDBiGRU().to(device)

    # Forward Pass
    outputs = model(inputs, pair_indices, pair_mask)
    print(f"    Output shape: {outputs.shape}")

    assert outputs.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert not torch.isnan(outputs).any(), "Model produced NaN values"
    print("    Model forward pass passed.")

    # 5. Metric Verification (MCRMSE)
    print("\n[5] Verifying Metric (MCRMSE)...")
    # Create dummy data: 2 samples, 3 columns scored
    # Sample 1: Error 1.0 on all cols -> MSE=1.0 -> RMSE=1.0
    # Sample 2: Error 0.0 on all cols -> MSE=0.0 -> RMSE=0.0
    # Col RMSEs: sqrt((1+0)/2) = sqrt(0.5) = 0.707...
    # MCRMSE: mean(0.707...) = 0.707...

    y_t = torch.tensor(
        [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]
    )  # (2, 2, 3)
    y_p = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]
    )  # (2, 2, 3)

    # Errors: Sample 0 -> diff 1, Sample 1 -> diff 0
    # RMSE per column:
    # Col 0: sqrt( (1^2 + 0^2)/2 ) = sqrt(0.5) = 0.7071
    # Col 1: 0.7071
    # Col 2: 0.7071
    # Mean: 0.7071

    score = mcrmse(y_t, y_p)
    print(f"    Calculated Score: {score.item():.4f}")
    assert (
        abs(score.item() - 0.7071) < 1e-3
    ), f"Metric calculation incorrect. Got {score.item()}"
    print("    Metric verification passed.")

    # 6. Training Loop Simulation
    print("\n[6] Running Training Simulation...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Define criterion (MCRMSE on all columns for training)
    def criterion(y_true, y_pred):
        return mcrmse(y_true, y_pred, scored_indices=None)

    best_val_score = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)

        # Validate
        val_score = validate(model, val_loader, device)

        print(
            f"    Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.4f}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"    >>> Saved Best Model")

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    print("    Training simulation complete.")

    # 7. Inference and Submission
    print("\n[7] Generating Submission...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            ids = batch["ids"]

            out = model(inputs, pair_indices, pair_mask)
            preds_list.append(out.cpu().numpy())
            ids_list.extend(ids)

    all_preds = np.concatenate(preds_list, axis=0)

    # Check shape of predictions
    # debug subset size for test might differ if dataset length < subset size,
    # but here we asked for 16 and test has 240, so we expect 16.
    expected_rows = min(Config.DEBUG_SUBSET_SIZE, 240)
    assert all_preds.shape[0] == expected_rows
    assert all_preds.shape[1] == Config.SEQ_LEN

    # Generate CSV logic (simplified from library.train.generate_submission for verification)
    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos]
            row_dict = {"id_seqpos": row_id}
            for j, col in enumerate(target_cols):
                row_dict[col] = float(row_values[j])
            submission_rows.append(row_dict)

    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print(f"    Submission shape: {sub_df.shape}")

    # Verify submission content
    assert "id_seqpos" in sub_df.columns
    assert "reactivity" in sub_df.columns
    assert len(sub_df) == expected_rows * Config.SEQ_LEN

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
