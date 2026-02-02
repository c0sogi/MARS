import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, compute_mcrmse
from library.data import get_dataloaders
from library.model import DeepStabilizedBiGRU
from library.train import MCRMSELoss, train_one_epoch, validate


def main():
    # =========================================================================
    # 1. Setup and Configuration Override
    # =========================================================================
    print("Step 1: Configuring environment for demo execution...")

    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    # Set reproducible seed
    set_seed(42)

    # Override Config for a fast demonstration
    Config.WORKING_DIR = "./working/demo_execution"
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Small subset for speed
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Create necessary directories
    Config.create_directories()

    print(f"  Working Directory: {Config.WORKING_DIR}")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Debug Mode: {Config.DEBUG}")

    # =========================================================================
    # 2. Data Loading Demonstration
    # =========================================================================
    print("\nStep 2: Loading DataLoaders...")

    # Load dataloaders (Train, Val, Test)
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # Verify Train Loader
    print("  Verifying Train Loader batch structure...")
    batch = next(iter(train_loader))

    features = batch["features"]
    pair_indices = batch["pair_indices"]
    pair_mask = batch["pair_mask"]
    targets = batch["targets"]
    ids = batch["ids"]

    # Assertions for shapes
    # Features: (Batch, Seq_Len, Input_Channels=14)
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_CHANNELS,
    ), f"Feature shape mismatch: {features.shape}"

    # Pair Indices: (Batch, Seq_Len)
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Pair indices shape mismatch: {pair_indices.shape}"

    # Pair Mask: (Batch, Seq_Len, 1)
    assert pair_mask.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        1,
    ), f"Pair mask shape mismatch: {pair_mask.shape}"

    # Targets: (Batch, Seq_Len, Num_Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Targets shape mismatch: {targets.shape}"

    print("  Data shapes verified successfully.")

    # =========================================================================
    # 3. Model Initialization and Forward Pass
    # =========================================================================
    print("\nStep 3: Initializing Model and running Forward Pass...")

    model = DeepStabilizedBiGRU().to(Config.DEVICE)

    # Move batch to device
    features = features.to(Config.DEVICE)
    pair_indices = pair_indices.to(Config.DEVICE)
    pair_mask = pair_mask.to(Config.DEVICE)

    # Run Forward Pass
    outputs = model(features, pair_indices, pair_mask)

    # Verify Output Shape: (Batch, Seq_Len, Num_Targets)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Model output shape mismatch: {outputs.shape}"

    print(f"  Model output shape: {outputs.shape}")
    print("  Forward pass successful.")

    # =========================================================================
    # 4. Training and Validation Loop Demonstration
    # =========================================================================
    print("\nStep 4: Demonstrating Training and Validation Steps...")

    criterion = MCRMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train for one epoch
    print("  Running training step...")
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, Config.DEVICE, epoch=0
    )

    assert isinstance(train_loss, float), "Train loss should be a float."
    assert train_loss > 0, "Train loss should be positive."
    print(f"  Train Loss: {train_loss:.4f}")

    # Validate
    print("  Running validation step...")
    val_score = validate(model, val_loader, Config.DEVICE)

    assert isinstance(val_score, float), "Validation score should be a float."
    print(f"  Validation MCRMSE: {val_score:.4f}")

    # =========================================================================
    # 5. Inference and Submission Generation
    # =========================================================================
    print("\nStep 5: Generating Submission from Test Data...")

    model.eval()
    all_preds = []
    all_ids = []

    # Run inference on test set
    with torch.no_grad():
        for batch in test_loader:
            f = batch["features"].to(Config.DEVICE)
            pi = batch["pair_indices"].to(Config.DEVICE)
            pm = batch["pair_mask"].to(Config.DEVICE)
            batch_ids = batch["ids"]

            out = model(f, pi, pm)

            # Move to CPU
            all_preds.append(out.cpu().numpy())
            all_ids.extend(batch_ids)

    # Concatenate predictions: (Total_Test_Samples, Seq_Len, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Verify dimensions
    # In debug mode, we have Config.DEBUG_SUBSET_SIZE samples
    expected_samples = Config.DEBUG_SUBSET_SIZE
    assert all_preds.shape == (
        expected_samples,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Prediction shape mismatch: {all_preds.shape}"

    print(f"  Predictions generated for {len(all_ids)} samples.")

    # Format Submission
    # We need to flatten: id_seqpos, and 5 target columns
    # id_seqpos format: {id}_{seqpos}

    submission_data = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for t_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_preds[t_idx]

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Verify Submission DataFrame
    expected_rows = expected_samples * Config.SEQ_LEN
    assert (
        len(submission_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"

    assert (
        list(submission_df.columns) == ["id_seqpos"] + target_cols
    ), "Submission columns mismatch."

    print(f"  Submission DataFrame constructed with {len(submission_df)} rows.")
    print("  First 3 rows:")
    print(submission_df.head(3).to_string(index=False))

    # Save (Optional, but good practice to verify write permissions)
    sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"  Submission saved to {sub_path}")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
