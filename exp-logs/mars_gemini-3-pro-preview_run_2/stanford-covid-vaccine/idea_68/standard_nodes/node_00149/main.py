import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library import config, dataset, model, train_utils


def main():
    # 1. Setup
    config.set_seed()
    device = config.get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Using full dataset as it is small enough for fast training
    train_dataset = dataset.RNADataset(
        split="train", load_cached_data=True, debug=False
    )
    val_dataset = dataset.RNADataset(split="val", load_cached_data=True, debug=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing HI-GFDN model...")
    net = model.HIGFDN().to(device)

    # 4. Training
    print("Starting training loop...")
    # run_training handles the loop, validation, early stopping, and saving best model
    train_utils.run_training(
        net,
        train_loader,
        val_loader,
        device,
        epochs=config.EPOCHS,
        patience=config.PATIENCE,
    )

    # 5. Final Validation
    print("Loading best model for final evaluation...")
    if os.path.exists(config.MODEL_PATH):
        net.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model weights not found. Using current weights.")

    # Compute final metric on the full validation set
    final_metric = train_utils.validate(net, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Load validation metadata
    val_df = pd.read_csv(config.VAL_CSV)

    # Calculate error per sample to correlate with metadata
    net.eval()
    sample_errors = []
    scored_cols_indices = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_len = config.SCORED_LEN

    with torch.no_grad():
        for x, p, y in val_loader:
            x, p, y = x.to(device), p.to(device), y.to(device)

            # Inference
            z = net.static_encoder(x)
            batch_size, _, length = x.shape
            y_prev = torch.zeros((batch_size, 5, length), device=device, dtype=x.dtype)
            y_pred_1 = net.recurrent_decoder(z, y_prev, p)
            y_pred_2 = net.recurrent_decoder(z, y_pred_1, p)

            # Calculate MCRMSE per sample
            preds_masked = y_pred_2[:, scored_cols_indices, :scored_len]
            targets_masked = y[:, scored_cols_indices, :scored_len]

            # MSE per element: (B, 3, 68)
            mse = (preds_masked - targets_masked) ** 2

            # RMSE per column per sample: (B, 3) -> Mean over sequence (dim 2) -> Sqrt
            rmse_per_col = torch.sqrt(torch.mean(mse, dim=2))

            # Mean over columns (dim 1) -> (B,)
            mcrmse_per_sample = torch.mean(rmse_per_col, dim=1)

            sample_errors.extend(mcrmse_per_sample.cpu().numpy())

    # Add errors to dataframe
    # Note: val_loader with shuffle=False preserves order of val_df
    if len(sample_errors) == len(val_df):
        val_df["error"] = sample_errors

        # Correlation with signal_to_noise
        if "signal_to_noise" in val_df.columns:
            corr_snr = val_df["error"].corr(val_df["signal_to_noise"])
            print(f"Correlation between Error and signal_to_noise: {corr_snr}")

        # Correlation with mean_reactivity
        if "mean_reactivity" in val_df.columns:
            corr_react = val_df["error"].corr(val_df["mean_reactivity"])
            print(f"Correlation between Error and mean_reactivity: {corr_react}")
    else:
        print(
            f"Warning: Mismatch in sample counts (Val DF: {len(val_df)}, Errors: {len(sample_errors)})"
        )

    # 7. Submission
    threshold = 0.47142532743789534

    if final_metric < threshold:
        print(
            f"Metric {final_metric} meets threshold {threshold}. Generating submission..."
        )

        # Load Test Data
        test_dataset = dataset.RNADataset(
            split="test", load_cached_data=True, debug=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

        all_preds = []

        # Inference on Test Set
        with torch.no_grad():
            for x, p, _ in test_loader:
                x, p = x.to(device), p.to(device)

                z = net.static_encoder(x)
                batch_size, _, length = x.shape
                y_prev = torch.zeros(
                    (batch_size, 5, length), device=device, dtype=x.dtype
                )
                y_pred_1 = net.recurrent_decoder(z, y_prev, p)
                y_pred_2 = net.recurrent_decoder(z, y_pred_1, p)

                # Move to CPU and store
                all_preds.append(y_pred_2.cpu().numpy())

        # Concatenate predictions: (N_total, 5, 107)
        all_preds = np.concatenate(all_preds, axis=0)
        # Transpose to (N_total, 107, 5)
        all_preds = all_preds.transpose(0, 2, 1)

        # Prepare Submission DataFrame
        ids = test_dataset.ids
        submission_rows = []
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(ids):
            sample_preds = all_preds[i]
            for pos in range(config.SEQ_LEN):
                row_id = f"{sample_id}_{pos}"
                row_dict = {"id_seqpos": row_id}
                for j, col in enumerate(target_cols):
                    row_dict[col] = float(sample_preds[pos, j])
                submission_rows.append(row_dict)

        submission_df = pd.DataFrame(submission_rows)

        # Save to ./submission/submission.csv
        os.makedirs("./submission", exist_ok=True)
        sub_path = "./submission/submission.csv"
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"Metric {final_metric} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
