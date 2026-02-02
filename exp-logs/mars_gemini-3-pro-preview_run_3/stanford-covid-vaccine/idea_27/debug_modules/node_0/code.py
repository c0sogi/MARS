import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import DeepInputAwareBiGRU
from library.engine import fit


def main():
    print("Initializing Demonstration Script...")

    # =========================================================================
    # 1. Configuration Overrides for Speed and Demonstration
    # =========================================================================
    # Modify Config to run a fast demo
    Config.EPOCHS = 2
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 64  # Small subset for quick execution
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Setup environment (directories, seeds)
    Config.setup()
    set_seed(Config.SEED)

    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("\nLoading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force processing to demonstrate pipeline
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Verification: Check batch structure
    sample_batch = next(iter(train_loader))
    print(f"Batch keys: {sample_batch.keys()}")
    print(f"Features shape: {sample_batch['features'].shape}")
    print(f"Targets shape: {sample_batch['targets'].shape}")

    assert (
        sample_batch["features"].shape[1] == Config.SEQ_LEN
    ), "Incorrect Sequence Length"
    assert (
        sample_batch["features"].shape[2] == Config.INPUT_CHANNELS
    ), "Incorrect Feature Channels"

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("\nInitializing Model...")
    model = DeepInputAwareBiGRU().to(Config.DEVICE)

    # Verification: Check forward pass
    with torch.no_grad():
        feats = sample_batch["features"].to(Config.DEVICE)
        p_idx = sample_batch["pair_indices"].to(Config.DEVICE)
        p_mask = sample_batch["pair_masks"].to(Config.DEVICE)
        out = model(feats, p_idx, p_mask)

    print(f"Model Output Shape: {out.shape}")
    assert out.shape == (
        feats.shape[0],
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), "Model output shape mismatch"

    # =========================================================================
    # 4. Training
    # =========================================================================
    print("\nStarting Training Loop...")
    # The fit function handles training, validation, early stopping, and saving best model
    model = fit(model, train_loader, val_loader, device=Config.DEVICE)

    # =========================================================================
    # 5. Inference on Test Set
    # =========================================================================
    print("\nRunning Inference on Test Set...")
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(Config.DEVICE)
            p_idx = batch["pair_indices"].to(Config.DEVICE)
            p_mask = batch["pair_masks"].to(Config.DEVICE)
            ids = batch["ids"]

            # Forward pass
            # Output: (B, 107, 5)
            preds = model(feats, p_idx, p_mask)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all predictions
    # Shape: (Total_Test_Samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    print(f"Total Test Predictions: {all_preds.shape}")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    print("\nGenerating Submission File...")

    # Target columns as specified in the competition
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    submission_rows = []

    num_samples = len(all_ids)
    seq_len = Config.SEQ_LEN  # 107

    for i in range(num_samples):
        sample_id = all_ids[i]
        sample_preds = all_preds[i]  # (107, 5)

        for j in range(seq_len):
            row_id = f"{sample_id}_{j}"
            probs = sample_preds[j]

            row_data = {
                "id_seqpos": row_id,
                "reactivity": probs[0],
                "deg_Mg_pH10": probs[1],
                "deg_pH10": probs[2],
                "deg_Mg_50C": probs[3],
                "deg_50C": probs[4],
            }
            submission_rows.append(row_data)

    submission_df = pd.DataFrame(submission_rows)

    # Save to file
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")
    print(f"Submission Shape: {submission_df.shape}")

    # =========================================================================
    # 7. Final Verification
    # =========================================================================
    assert os.path.exists(submission_path), "Submission file was not created."

    # Expected rows: Num_Test_Samples * 107
    expected_rows = len(all_ids) * Config.SEQ_LEN
    assert (
        len(submission_df) == expected_rows
    ), f"Expected {expected_rows} rows, found {len(submission_df)}"

    # Check columns
    expected_cols = ["id_seqpos"] + target_cols
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch"

    print("\nDemonstration Completed Successfully.")


if __name__ == "__main__":
    main()
