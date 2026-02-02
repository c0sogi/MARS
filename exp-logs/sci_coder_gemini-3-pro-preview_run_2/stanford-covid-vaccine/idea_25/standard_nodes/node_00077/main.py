import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import set_seed, MetricTracker
from library.data import get_dataloaders
from library.model import StackedInteractionDenseNet
from library.train import train_model

# ------------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# ------------------------------------------------------------------------------
# Limit epochs to ensure execution completes quickly (Baseline requirement)
Config.EPOCHS = 15
# Increase batch size slightly for speed
Config.BATCH_SIZE = 32


def main():
    # Set global seeds for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # --------------------------------------------------------------------------
    # 1. Model Training
    # --------------------------------------------------------------------------
    print("Starting training pipeline...")
    # train_model() handles data loading, training loop, and saving the best model
    train_model()

    # --------------------------------------------------------------------------
    # 2. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\nStarting validation and failure analysis...")

    # Initialize a fresh model and load the best weights saved during training
    model = StackedInteractionDenseNet().to(device)
    model_path = Config.MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Validation Data
    # We need the loader for inference and the dataframe for metadata correlation
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )
    val_df = pd.read_csv(Config.VAL_CSV)

    # Metric Tracker for the official score
    tracker = MetricTracker()

    # Store predictions for failure analysis
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            # Inference
            outputs = model(inputs, partner_indices)

            # Update global metric
            tracker.update(outputs, targets, mask)

            # Collect data for analysis
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    # Compute and print the final metric
    final_metric = tracker.compute()
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Concatenate all batches
    preds_arr = np.concatenate(all_preds, axis=0)  # (N_samples, Seq_Len, 5)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N_samples, Seq_Len, 5)
    masks_arr = np.concatenate(all_masks, axis=0)  # (N_samples, Seq_Len)

    # Calculate per-sample RMSE for the scored columns only
    scored_indices = Config.SCORED_TARGET_INDICES

    # Filter for scored columns
    p_filt = preds_arr[:, :, scored_indices]
    t_filt = targets_arr[:, :, scored_indices]

    # Expand mask for broadcasting: (N, L) -> (N, L, 1)
    m_exp = masks_arr[:, :, None]

    # Squared Error
    sq_diff = (p_filt - t_filt) ** 2
    masked_sq_diff = sq_diff * m_exp

    # Sum errors per sample (over sequence and channels)
    sample_sse = np.sum(masked_sq_diff, axis=(1, 2))

    # Count valid positions per sample
    # Each valid sequence position contributes 'len(scored_indices)' values
    sample_counts = np.sum(m_exp, axis=(1, 2)) * len(scored_indices)

    # Avoid division by zero
    sample_mse = sample_sse / (sample_counts + 1e-12)
    sample_rmse = np.sqrt(sample_mse)

    # Add error metric to dataframe subset (ensure lengths match)
    # val_loader might drop last incomplete batch if configured, but default is usually False.
    # We assume standard behavior where len(preds) == len(val_df).
    if len(sample_rmse) == len(val_df):
        val_df["error_rmse"] = sample_rmse

        print("\nFailure Analysis Correlations:")

        # Correlation with Signal to Noise
        if "signal_to_noise" in val_df.columns:
            corr_sn, _ = pearsonr(val_df["signal_to_noise"], val_df["error_rmse"])
            print(f"  Error vs Signal_to_Noise: {corr_sn:.4f}")

        # Correlation with Mean Reactivity
        if "mean_reactivity" in val_df.columns:
            corr_mr, _ = pearsonr(val_df["mean_reactivity"], val_df["error_rmse"])
            print(f"  Error vs Mean Reactivity: {corr_mr:.4f}")

        # Correlation with GC Content (derived feature)
        val_df["gc_content"] = val_df["sequence"].apply(
            lambda x: (x.count("G") + x.count("C")) / len(x)
        )
        corr_gc, _ = pearsonr(val_df["gc_content"], val_df["error_rmse"])
        print(f"  Error vs GC Content: {corr_gc:.4f}")
    else:
        print(
            f"Warning: Mismatch in validation set size ({len(val_df)}) and predictions ({len(sample_rmse)}). Skipping detailed failure analysis."
        )

    # --------------------------------------------------------------------------
    # 3. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.5417620723771521

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Load Test Data
        _, _, test_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                partner_indices = batch["partner_indices"].to(device)
                ids = batch["id"]

                outputs = model(inputs, partner_indices)

                test_preds.append(outputs.cpu().numpy())
                test_ids.extend(ids)

        # Concatenate predictions: (N_test, 107, 5)
        test_preds_arr = np.concatenate(test_preds, axis=0)

        # Prepare Submission DataFrame
        # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # These correspond to indices 0, 1, 2, 3, 4 in the model output
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        submission_data = []

        for i, sample_id in enumerate(test_ids):
            sample_pred = test_preds_arr[i]  # Shape (107, 5)

            for seqpos in range(Config.SEQ_LENGTH):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_pred[seqpos]

                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(target_cols):
                    row_dict[col_name] = row_values[col_idx]

                submission_data.append(row_dict)

        submission_df = pd.DataFrame(submission_data)

        # Ensure column order
        submission_df = submission_df[["id_seqpos"] + target_cols]

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission generation skipped.")


if __name__ == "__main__":
    main()
