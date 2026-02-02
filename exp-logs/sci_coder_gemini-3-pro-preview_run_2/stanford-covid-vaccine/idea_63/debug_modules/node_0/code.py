import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Import library modules
# Note: We assume the file structure provided in the prompt exists.
import library.config as config
from library.data import get_dataloaders, RNADataset
from library.model import HS_GFDN
from library.loss import MaskedMCRMSE
from library.train import train_one_epoch, validate, inference, generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading
    print("\n[1/6] Loading DataLoaders...")
    # We load cached data if available to speed up the process
    train_loader_full, val_loader_full, test_loader_full = get_dataloaders(
        load_cached_data=True
    )

    # Verify DataLoaders are not empty
    assert len(train_loader_full) > 0, "Train loader is empty"
    assert len(val_loader_full) > 0, "Val loader is empty"
    assert len(test_loader_full) > 0, "Test loader is empty"
    print("DataLoaders loaded successfully.")

    # 3. Create Mini-DataLoaders for Speed
    # We will use only 2 batches (approx 32 samples) for training/val/test to keep runtime short
    print("\n[2/6] Creating Mini-DataLoaders for rapid verification...")
    subset_size = 32
    batch_size = config.BATCH_SIZE

    # Helper to create subset loader
    def create_mini_loader(loader, size):
        dataset = loader.dataset
        # Ensure we don't exceed dataset size
        indices = list(range(min(len(dataset), size)))
        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,  # No shuffle for deterministic demo
            num_workers=0,  # 0 workers for simple debugging/demo
            pin_memory=True,
        )

    mini_train_loader = create_mini_loader(train_loader_full, subset_size)
    mini_val_loader = create_mini_loader(val_loader_full, subset_size)
    mini_test_loader = create_mini_loader(test_loader_full, subset_size)

    print(f"Mini-Train batches: {len(mini_train_loader)}")
    print(f"Mini-Val batches: {len(mini_val_loader)}")
    print(f"Mini-Test batches: {len(mini_test_loader)}")

    # 4. Model Instantiation and Forward Pass Check
    print("\n[3/6] Initializing Model and checking Forward Pass...")
    model = HS_GFDN().to(device)

    # Get one batch from mini_train_loader
    batch = next(iter(mini_train_loader))
    inputs = batch["inputs"].to(device)
    partner_indices = batch["partner_indices"].to(device)
    targets = batch["targets"].to(device)

    print(f"Input Shape: {inputs.shape}")  # Should be (B, 107, 18)
    print(f"Target Shape: {targets.shape}")  # Should be (B, 107, 5)

    # Run forward pass (Static path + Feedback path implicitly handled in train_one_epoch,
    # but here we call forward directly which does the full pass if feedback provided or default)
    # The model.forward signature is (inputs, partner_indices, feedback=None)
    # By default feedback is None -> Zero feedback pass
    outputs = model(inputs, partner_indices)

    print(f"Output Shape: {outputs.shape}")

    # Validation: Check output shape matches (Batch, Seq_Len, 5)
    assert outputs.shape == (
        inputs.shape[0],
        config.SEQ_LENGTH,
        5,
    ), f"Expected output shape {(inputs.shape[0], config.SEQ_LENGTH, 5)}, got {outputs.shape}"

    # 5. Loss Function Check
    print("\n[4/6] Verifying Loss Function...")
    criterion = MaskedMCRMSE()
    loss = criterion(outputs, targets)

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # 6. Training Loop Simulation
    print("\n[5/6] Simulating Training and Validation Loop...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch of training on the mini subset
    train_loss = train_one_epoch(model, mini_train_loader, optimizer, criterion, device)
    print(f"Mini-Epoch Train Loss: {train_loss:.6f}")

    # Run validation on the mini subset
    val_mcrmse = validate(model, mini_val_loader, device)
    print(f"Mini-Validation MCRMSE: {val_mcrmse:.6f}")

    # 7. Inference and Submission
    print("\n[6/6] Running Inference and Generating Submission...")
    preds, ids = inference(model, mini_test_loader, device)

    print(f"Predictions Shape: {preds.shape}")
    print(f"Number of IDs: {len(ids)}")

    # Verify predictions shape: (N_samples, 107, 5)
    assert preds.shape[0] == len(ids), "Mismatch between predictions and IDs count"
    assert (
        preds.shape[1] == config.SEQ_LENGTH
    ), "Incorrect sequence length in predictions"
    assert preds.shape[2] == 5, "Incorrect number of target columns in predictions"

    # Generate Submission File
    submission_file = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    generate_submission(preds, ids, submission_file)

    # Verify file creation
    assert os.path.exists(submission_file), "Submission file was not created"

    # Verify file content format
    df_sub = pd.read_csv(submission_file)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    expected_cols = ["id_seqpos"] + config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), "Submission columns do not match expected format"

    # Check row count: N_samples * Seq_Len
    expected_rows = len(ids) * config.SEQ_LENGTH
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, found {len(df_sub)}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
