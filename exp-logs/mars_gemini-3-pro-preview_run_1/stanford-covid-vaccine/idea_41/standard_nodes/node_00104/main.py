import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import load_data
from library.model import CapacityStabilizedBiGRU
from library.engine import train_one_epoch, validate


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure working directories exist
    os.makedirs(Config.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(Config.model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    # 2. Load Data
    print("Loading data...")
    # load_cached_data=True will use .npz files in working/ if available, or create them
    train_ds, val_ds, test_ds = load_data(load_cached_data=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = CapacityStabilizedBiGRU(Config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

    criterion = nn.MSELoss()

    # 4. Training Loop
    best_mcrmse = float("inf")

    print("Starting training...")
    for epoch in range(Config.epochs):
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, device, criterion
        )
        val_mcrmse = validate(model, val_loader, device)

        scheduler.step()

        # Save best model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.model_save_path)

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

    # 5. Final Evaluation
    print("Loading best model for final evaluation...")
    if os.path.exists(Config.model_save_path):
        model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    final_val_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis...")
    model.eval()
    val_errors = []
    val_ids = []

    # Collect errors per sample
    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            y = batch["y"].to(device)

            preds = model(seq, loop, dist)

            # Slice to scored positions
            preds_sliced = preds[:, : Config.pred_len, :]
            y_sliced = y[:, : Config.pred_len, :]

            # Calculate MSE per sample (average over positions and channels)
            # Shape: (B, L, 3) -> (B,)
            sq_err = (preds_sliced - y_sliced) ** 2
            mse_per_sample = sq_err.mean(dim=(1, 2)).cpu().numpy()

            val_errors.extend(mse_per_sample)

            # We don't have IDs in the batch dict from RNADataset directly,
            # but the loader iterates sequentially. We can match by index later
            # or just use the order since shuffle=False.

    # Load validation metadata to get features
    df_val = pd.read_parquet(Config.val_data_path)

    # Ensure alignment
    if len(val_errors) != len(df_val):
        print("Warning: Mismatch in validation set size for failure analysis.")
    else:
        df_val["error_mse"] = val_errors

        # calculate GC content
        df_val["gc_content"] = df_val["sequence"].apply(
            lambda x: (x.count("G") + x.count("C")) / len(x)
        )

        # Correlations
        correlations = {}
        features_to_check = ["signal_to_noise", "SN_filter", "gc_content", "seq_length"]

        print("Correlation between Error (MSE) and features:")
        for feat in features_to_check:
            if feat in df_val.columns:
                corr = df_val["error_mse"].corr(df_val[feat])
                correlations[feat] = corr
                print(f"  {feat}: {corr:.4f}")

    # 7. Submission
    threshold = 0.6199890971183777
    if final_val_metric < threshold:
        print(
            f"\nMetric ({final_val_metric:.6f}) < Threshold ({threshold:.6f}). Generating submission..."
        )

        model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)

                preds = model(seq, loop, dist)  # (B, 107, 3)
                all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)

        # Format for submission
        # Targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        # Submission columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

        submission_rows = []
        test_ids = test_ds.ids

        for i, sample_id in enumerate(test_ids):
            sample_pred = all_preds[i]  # (107, 3)

            for seqpos in range(Config.seq_len):
                row_id = f"{sample_id}_{seqpos}"

                reactivity = float(sample_pred[seqpos, 0])
                deg_Mg_pH10 = float(sample_pred[seqpos, 1])
                deg_Mg_50C = float(sample_pred[seqpos, 2])

                # Fill others with 0
                deg_pH10 = 0.0
                deg_50C = 0.0

                submission_rows.append(
                    [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
                )

        columns = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        sub_df = pd.DataFrame(submission_rows, columns=columns)

        sub_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")

    else:
        print(
            f"\nMetric ({final_val_metric:.6f}) >= Threshold ({threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
