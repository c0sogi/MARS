import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_device
from library.data import process_data, RNADataset
from library.model import GCSDNModel
from library.train import train_one_epoch, validate as validate_fn


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()

    # Override Config for Fast Baseline
    Config.EPOCHS = 15

    print(f"Device: {device}")
    print(f"Training for {Config.EPOCHS} epochs (Fast Baseline)...")

    # 2. Data Loading
    train_data = process_data(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        is_test=False,
        cache_name=Config.CACHE_TRAIN,
    )
    val_data = process_data(
        os.path.join(Config.METADATA_DIR, "val.csv"),
        is_test=False,
        cache_name=Config.CACHE_VAL,
    )

    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    model = GCSDNModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    # 4. Training Loop
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate_fn(model, val_loader, device)

        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # 5. Final Evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    final_metric = validate_fn(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis on Validation Set...")

    # Get predictions and targets for analysis
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, p_idx, targets in val_loader:
            features = features.to(device)
            p_idx = p_idx.to(device)
            _, y2 = model(features, p_idx)
            all_preds.append(y2.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 5, L)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 5, L)

    # Calculate RMSE per sample (focused on scored columns and length)
    scored_cols = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_len = Config.SEQ_SCORED

    preds_s = all_preds[:, scored_cols, :scored_len]
    targs_s = all_targets[:, scored_cols, :scored_len]

    # MSE per sample: Mean over columns and length
    mse_per_sample = np.mean((preds_s - targs_s) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load metadata for correlation
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))

    # Ensure alignment (dataset order is preserved if shuffle=False)
    if len(val_df) != len(rmse_per_sample):
        print("Warning: Mismatch in validation set size for analysis.")
    else:
        # Features to correlate
        val_df["error_rmse"] = rmse_per_sample

        # 1. Signal to Noise
        if "signal_to_noise" in val_df.columns:
            corr_sn, _ = pearsonr(val_df["error_rmse"], val_df["signal_to_noise"])
            print(f"Correlation (Error vs Signal_to_Noise): {corr_sn:.4f}")

        # 2. Sequence Length (constant 107, but checking just in case)
        # 3. Base Counts
        val_df["count_A"] = val_df["sequence"].apply(lambda x: x.count("A"))
        corr_A, _ = pearsonr(val_df["error_rmse"], val_df["count_A"])
        print(f"Correlation (Error vs Count A): {corr_A:.4f}")

        val_df["count_G"] = val_df["sequence"].apply(lambda x: x.count("G"))
        corr_G, _ = pearsonr(val_df["error_rmse"], val_df["count_G"])
        print(f"Correlation (Error vs Count G): {corr_G:.4f}")

    # 7. Submission
    THRESHOLD = 0.47142532743789534
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        test_data = process_data(
            os.path.join(Config.METADATA_DIR, "test.csv"),
            is_test=True,
            cache_name=Config.CACHE_TEST,
        )
        test_dataset = RNADataset(test_data, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        preds_map = {}

        with torch.no_grad():
            for features, p_idx, ids in test_loader:
                features = features.to(device)
                p_idx = p_idx.to(device)

                _, y2 = model(features, p_idx)
                y_np = y2.cpu().numpy()  # (B, 5, L)

                for i, sample_id in enumerate(ids):
                    # Map predictions to id_seqpos for all 107 positions
                    for pos in range(Config.SEQ_LENGTH):
                        row_id = f"{sample_id}_{pos}"
                        preds_map[row_id] = y_np[i, :, pos]

        # Create Submission DataFrame
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        submission_data = []

        # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # Indices: 0, 1, 2, 3, 4

        # Efficient construction
        # We need to preserve the order of sample_sub
        # Create a dictionary for fast lookup

        # Pre-allocate array
        num_rows = len(sample_sub)
        result_array = np.zeros((num_rows, 5), dtype=np.float32)

        # Iterate and fill
        # Note: This loop can be slow if not optimized, but 25k rows is fast enough in Python
        ids_seqpos = sample_sub["id_seqpos"].values

        found_count = 0
        for idx, row_id in enumerate(ids_seqpos):
            if row_id in preds_map:
                result_array[idx] = preds_map[row_id]
                found_count += 1

        print(f"Matched {found_count}/{num_rows} rows.")

        submission_df = pd.DataFrame(
            result_array,
            columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
        )
        submission_df.insert(0, "id_seqpos", sample_sub["id_seqpos"])

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
