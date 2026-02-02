import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, mcrmse_loss
from library.data import get_dataloaders
from library.model import MaskedBiGRU
from library.engine import run_training

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print("Initializing Demo Configuration...")

    # Set a fixed seed for reproducibility
    seed_everything(42)

    # Override Config paths to use a separate demo directory
    # Note: We must update dependent paths manually since they were defined at class level
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Override hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Use 0 workers for simple debugging/demo
    Config.HIDDEN_DIM = 64  # Smaller model for speed
    Config.CNN_FILTERS = 64

    # Create necessary directories
    Config.setup()

    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Verification
    # -------------------------------------------------------------------------
    print("\nLoading Data (Debug Mode)...")

    # Load subset of data (debug=True takes top 100 samples)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force re-creation for demo
        debug=True,
    )

    # Fetch a single batch from train loader to verify shapes and masking
    masked_input, original_input, targets, mask = next(iter(train_loader))

    print(f"Train Batch Shapes:")
    print(
        f"  Masked Input:   {masked_input.shape} (Expected: {Config.BATCH_SIZE}, 107, 14)"
    )
    print(f"  Original Input: {original_input.shape}")
    print(
        f"  Targets:        {targets.shape}      (Expected: {Config.BATCH_SIZE}, 107, 5)"
    )
    print(
        f"  Mask:           {mask.shape}         (Expected: {Config.BATCH_SIZE}, 107)"
    )

    # Assertions for Data Integrity
    assert masked_input.shape == (Config.BATCH_SIZE, 107, 14), "Incorrect input shape"
    assert targets.shape == (Config.BATCH_SIZE, 107, 5), "Incorrect target shape"

    # Verify Masking Logic:
    # In training mode, mask should have some non-zero values (1.0 indicates masked)
    mask_sum = mask.sum().item()
    print(f"  Total Masked Positions in Batch: {mask_sum}")
    assert mask_sum > 0, "No masking applied in training loader!"

    # Verify that masked positions in 'masked_input' are indeed zeroed out
    # We check one masked position
    if mask_sum > 0:
        batch_idx, seq_idx = torch.where(mask > 0)
        b, s = batch_idx[0], seq_idx[0]
        assert torch.all(
            masked_input[b, s] == 0
        ), "Input features at masked position are not zero!"
        assert not torch.all(
            original_input[b, s] == 0
        ), "Original features at masked position are zero!"

    # Fetch a batch from val loader to verify NO masking
    val_in, _, _, val_mask = next(iter(val_loader))
    assert val_mask.sum().item() == 0, "Validation loader should not apply masking!"

    # -------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\nInitializing Model...")
    device = torch.device(Config.DEVICE)
    model = MaskedBiGRU().to(device)

    # Dummy Forward Pass
    dummy_input = masked_input.to(device)
    reg_out, recon_out = model(dummy_input)

    print("Model Output Shapes:")
    print(f"  Regression Output:     {reg_out.shape}")
    print(f"  Reconstruction Output: {recon_out.shape}")

    assert reg_out.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), "Incorrect regression output shape"
    assert recon_out.shape == (
        Config.BATCH_SIZE,
        107,
        14,
    ), "Incorrect reconstruction output shape"

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\nStarting Training Loop...")

    # Run training (returns best validation MCRMSE)
    best_score = run_training(train_loader, val_loader)

    print(f"Training finished. Best Val Score: {best_score}")

    # Verify model file was created
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint not found!"

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\nRunning Inference on Test Set...")

    # Load the best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    model.to(device)

    preds = []

    with torch.no_grad():
        for inputs, _, _, _ in test_loader:
            inputs = inputs.to(device)
            # Forward pass
            reg_out, _ = model(inputs)
            # Move to CPU and numpy
            preds.append(reg_out.cpu().numpy())

    # Concatenate predictions: (N_samples, 107, 5)
    preds = np.concatenate(preds, axis=0)
    print(f"Prediction Array Shape: {preds.shape}")

    # We need to map these predictions to IDs.
    # Since we used debug=True, we have a subset of the test set.
    # We need to load the corresponding IDs from the test metadata.
    # Note: get_dataloaders(debug=True) slices the data *after* loading.
    # We replicate that slicing on the dataframe to ensure alignment.

    test_df = pd.read_parquet(Config.TEST_DATA_PATH)
    # The debug logic in get_dataloaders slices [:100]
    test_df_subset = test_df.iloc[:100].reset_index(drop=True)

    assert (
        len(test_df_subset) == preds.shape[0]
    ), "Mismatch between test dataframe subset and predictions"

    # Prepare Submission Data
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in test_df_subset.iterrows():
        sample_id = row["id"]
        sample_preds = preds[idx]  # Shape (107, 5)

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            submission_row = [row_id] + row_values
            submission_rows.append(submission_row)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows, columns=["id_seqpos"] + target_cols)

    print(f"Submission DataFrame Shape: {submission_df.shape}")
    print("First 5 rows of submission:")
    print(submission_df.head())

    # Save Submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Final Validation of Submission File
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert saved_df.shape == (100 * 107, 6), "Saved submission has incorrect dimensions"
    assert (
        list(saved_df.columns) == ["id_seqpos"] + target_cols
    ), "Saved submission has incorrect columns"

    print("\nDemo Execution Completed Successfully.")
