import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library import config, utils, data, model, train


def main():
    # 1. Setup
    utils.set_seed(42)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # 2. Training
    # Run training for 15 epochs to ensure a fast but effective baseline.
    # The library.train.run_training function handles the loop, saving best_model.pth, etc.
    print("\n=== Starting Training ===")
    train.run_training(epochs=15)

    # 3. Validation & Metric Calculation
    print("\n=== Starting Validation ===")

    # Load the best model
    net = model.RecurrentDenseNet().to(device)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # Load validation data
    # We use load_cached_data=True to speed up loading
    _, val_loader = data.get_loaders(load_cached_data=True)

    all_preds = []
    all_targets = []
    all_masks = []

    # Inference loop
    with torch.no_grad():
        for x_static, partner_idx, y in val_loader:
            x_static = x_static.to(device)
            partner_idx = partner_idx.to(device)
            y = y.to(device)
            B = x_static.shape[0]

            # Recurrent Inference Strategy
            # Pass 1: Initial guess with zero recycling
            x_recycled_1 = torch.zeros((B, config.SEQ_LEN, 5), device=device)
            pred_1 = net(x_static, x_recycled_1, partner_idx)

            # Pass 2: Refinement using Pass 1 output
            x_recycled_2 = pred_1
            pred_2 = net(x_static, x_recycled_2, partner_idx)

            # Store results on CPU to save GPU memory
            all_preds.append(pred_2.cpu())
            all_targets.append(y.cpu())

            # Create scoring mask (1.0 for first 68 positions, 0.0 otherwise)
            mask = torch.zeros((B, config.SEQ_LEN))
            mask[:, : config.PRED_LEN] = 1.0
            all_masks.append(mask)

    # Concatenate all batches
    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    masks_tensor = torch.cat(all_masks, dim=0)

    # Compute MCRMSE manually to ensure precision and correctness
    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = config.SCORED_INDICES
    rmse_list = []

    for idx in scored_indices:
        p = preds_tensor[:, :, idx]
        t = targets_tensor[:, :, idx]
        m = masks_tensor

        # Squared Error
        diff_sq = (p - t) ** 2

        # Apply mask
        diff_sq = diff_sq * m

        # Mean Squared Error over valid positions
        # Sum of errors / Sum of mask (count of valid positions)
        mse = diff_sq.sum() / m.sum()
        rmse = torch.sqrt(mse)
        rmse_list.append(rmse)

    # Final Metric is the mean of the column-wise RMSEs
    final_metric = torch.mean(torch.stack(rmse_list)).item()

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Performing Failure Analysis ===")

    # Retrieve sample IDs to map errors back to metadata
    val_data_dict = data.process_data("val", load_cached_data=True)
    val_ids = val_data_dict["ids"]

    # Load metadata
    if os.path.exists(config.VAL_CSV):
        val_meta_df = pd.read_csv(config.VAL_CSV)

        # Calculate per-sample error (MCRMSE per sample)
        per_sample_errors = []
        num_samples = len(val_ids)

        for i in range(num_samples):
            sample_rmses = []
            for idx in scored_indices:
                p = preds_tensor[i, : config.PRED_LEN, idx]
                t = targets_tensor[i, : config.PRED_LEN, idx]
                mse = torch.mean((p - t) ** 2)
                sample_rmses.append(torch.sqrt(mse))

            # Mean of RMSEs for this sample
            sample_metric = torch.mean(torch.stack(sample_rmses)).item()
            per_sample_errors.append(sample_metric)

        # Create analysis DataFrame
        analysis_df = pd.DataFrame(
            {"id": val_ids, "error_magnitude": per_sample_errors}
        )

        # Merge with metadata
        merged_df = pd.merge(analysis_df, val_meta_df, on="id", how="inner")

        # Calculate correlations
        features_to_check = [
            "signal_to_noise",
            "SN_filter",
            "mean_reactivity",
            "seq_length",
        ]
        print("Correlation between Error Magnitude and Features:")

        for feature in features_to_check:
            if feature in merged_df.columns:
                corr = merged_df["error_magnitude"].corr(merged_df[feature])
                print(f"  {feature}: {corr:.6f}")
            else:
                print(f"  {feature}: Not found in metadata")
    else:
        print("Validation metadata CSV not found. Skipping detailed failure analysis.")

    # 5. Submission
    threshold = 0.5417620723771521
    if final_metric < threshold:
        print(
            f"\nMetric {final_metric} is better than threshold {threshold}. Generating submission..."
        )
        train.generate_submission()
    else:
        print(
            f"\nMetric {final_metric} did not beat threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
