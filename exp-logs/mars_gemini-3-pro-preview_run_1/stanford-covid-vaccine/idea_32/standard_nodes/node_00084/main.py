import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, metric_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss


def main():
    # 1. Configuration and Setup
    config = Config()
    seed_everything(config.seed)
    device = torch.device(config.device)

    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # Load validation dataframe for failure analysis later
    df_val = pd.read_parquet(config.val_file)

    # 3. Model Initialization
    model = RNAModel(config).to(device)

    # 4. Training Setup
    criterion = MaskedMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_mcrmse = float("inf")

    # 5. Training Loop
    print("Starting Training...")
    for epoch in range(config.epochs):
        model.train()
        train_loss_accum = 0.0

        # Training Step
        for seq, loop, dist, target, mask in train_loader:
            seq, loop, dist = seq.to(device), loop.to(device), dist.to(device)
            target, mask = target.to(device), mask.to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model(seq, loop, dist)

            # Calculate loss
            loss = criterion(preds, target, mask)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        scheduler.step()
        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation Step
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for seq, loop, dist, target, mask in val_loader:
                seq, loop, dist = seq.to(device), loop.to(device), dist.to(device)
                target = target.to(device)

                preds = model(seq, loop, dist)

                # We only score the first 68 positions (pred_len)
                # Slice: (B, 68, 3)
                preds_sliced = preds[:, : config.pred_len, :]
                targets_sliced = target[:, : config.pred_len, :]

                val_preds_list.append(preds_sliced.cpu().numpy())
                val_targets_list.append(targets_sliced.cpu().numpy())

        val_preds = np.concatenate(val_preds_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)

        # Calculate MCRMSE
        current_mcrmse = metric_mcrmse(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {current_mcrmse:.6f}"
        )

        # Checkpointing
        if current_mcrmse < best_mcrmse:
            best_mcrmse = current_mcrmse
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  New best model saved! ({best_mcrmse:.6f})")

    # 6. Final Evaluation & Failure Analysis
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    model.eval()

    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for seq, loop, dist, target, mask in val_loader:
            seq, loop, dist = seq.to(device), loop.to(device), dist.to(device)
            target = target.to(device)
            preds = model(seq, loop, dist)

            val_preds_list.append(preds[:, : config.pred_len, :].cpu().numpy())
            val_targets_list.append(target[:, : config.pred_len, :].cpu().numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    final_metric = metric_mcrmse(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")
    # Calculate MSE per sample (average over positions and channels)
    # Shape: (N_samples, 68, 3) -> (N_samples,)
    sample_mse = np.mean((val_targets - val_preds) ** 2, axis=(1, 2))

    # Ensure df_val matches the order of the loader (SequentialSampler for eval)
    # The loader preserves order, and we read parquet directly.
    # Assuming the loader iterates through the dataframe in index order.
    if len(df_val) != len(sample_mse):
        print(
            f"Warning: Validation set size mismatch. DF: {len(df_val)}, Preds: {len(sample_mse)}"
        )
        # Truncate to match if needed (though shouldn't happen with correct config)
        min_len = min(len(df_val), len(sample_mse))
        df_val = df_val.iloc[:min_len]
        sample_mse = sample_mse[:min_len]

    df_val = df_val.copy()
    df_val["error_mse"] = sample_mse

    # Feature Engineering for Analysis
    df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

    analysis_cols = ["signal_to_noise", "SN_filter", "len_A", "len_G", "len_C", "len_U"]
    # Filter columns that exist
    analysis_cols = [c for c in analysis_cols if c in df_val.columns]

    print("Correlation between Error (MSE) and Features:")
    correlations = (
        df_val[analysis_cols + ["error_mse"]].corr()["error_mse"].drop("error_mse")
    )
    print(correlations.sort_values(ascending=False))

    # 7. Submission
    threshold = 0.6199890971183777
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")

        test_ids = []
        test_preds_list = []

        with torch.no_grad():
            for seq, loop, dist, _, _ in test_loader:
                seq, loop, dist = seq.to(device), loop.to(device), dist.to(device)

                # Predict
                preds = model(seq, loop, dist)  # (B, 107, 3)
                test_preds_list.append(preds.cpu().numpy())

        # Concatenate all batches: (N_test, 107, 3)
        all_test_preds = np.concatenate(test_preds_list, axis=0)

        # Get IDs from test dataframe (order is preserved)
        df_test = pd.read_parquet(config.test_file)
        ids = df_test["id"].values

        # Prepare submission data
        # We need to flatten: id_seqpos for 0..106
        # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # Model outputs: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (2)

        submission_rows = []

        for i, sample_id in enumerate(ids):
            sample_preds = all_test_preds[i]  # (107, 3)

            for seqpos in range(config.seq_len):
                row_id = f"{sample_id}_{seqpos}"

                # Extract predictions
                reactivity = float(sample_preds[seqpos, 0])
                deg_Mg_pH10 = float(sample_preds[seqpos, 1])
                deg_Mg_50C = float(sample_preds[seqpos, 2])

                # Fill missing columns with 0.0
                deg_pH10 = 0.0
                deg_50C = 0.0

                submission_rows.append(
                    {
                        "id_seqpos": row_id,
                        "reactivity": reactivity,
                        "deg_Mg_pH10": deg_Mg_pH10,
                        "deg_pH10": deg_pH10,
                        "deg_Mg_50C": deg_Mg_50C,
                        "deg_50C": deg_50C,
                    }
                )

        # Create DataFrame
        submission_df = pd.DataFrame(submission_rows)

        # Save
        submission_df.to_csv(config.submission_path, index=False)
        print(f"Submission saved to {config.submission_path}")

    else:
        print(
            f"\nMetric {final_metric} >= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
