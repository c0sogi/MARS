import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from tqdm import tqdm

# Import library modules
from library.config import Config
from library.utils import set_seed, GlobalMCRMSE
from library.engine import run_training
from library.model import RNAModel
from library.data import get_dataloaders


def main():
    print("Initializing Run...")
    set_seed(Config.SEED)

    # Override Config for execution
    # Increasing to 50 epochs to ensure full convergence as per Lesson 00062
    Config.NUM_EPOCHS = 50

    # --------------------------------------------------------------------------
    # 1. Training
    # --------------------------------------------------------------------------
    print("\nStarting Training...")
    # run_training returns the best validation score (MCRMSE)
    # It also saves the best model to Config.WORKING_DIR/best_model.pth
    best_val_score_from_engine = run_training(debug=False)

    # --------------------------------------------------------------------------
    # 2. Validation Inference & Metric Confirmation
    # --------------------------------------------------------------------------
    print("\nRunning Validation Inference...")
    device = Config.DEVICE

    # Load Best Model
    model = RNAModel().to(device)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get Loaders
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Containers for Failure Analysis
    all_val_preds = []
    all_val_targets = []
    all_val_ids = []

    # Metric Accumulator
    metric_fn = GlobalMCRMSE()

    with torch.no_grad():
        for inputs, partner_indices, targets, ids in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Forward
            outputs = model(inputs, partner_indices)

            # Update Metric
            metric_fn.update(outputs, targets)

            # Store for Failure Analysis
            # We only care about the scored positions for error analysis
            # outputs: (B, 107, 5) -> slice to (B, 68, 5)
            # targets: (B, 107, 5) -> slice to (B, 68, 5)
            # We also only care about the scored columns [0, 1, 3]

            # Move to CPU for analysis
            all_val_preds.append(outputs.cpu().numpy())
            all_val_targets.append(targets.cpu().numpy())
            all_val_ids.extend(ids)

    # Compute Final Metric
    final_metric = metric_fn.compute()
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 3. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Concatenate batches
    val_preds_arr = np.concatenate(all_val_preds, axis=0)  # (N, 107, 5)
    val_targets_arr = np.concatenate(all_val_targets, axis=0)  # (N, 107, 5)

    # Slice to scored region and columns for error calculation
    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = Config.SCORED_TARGET_INDICES
    seq_scored = Config.SEQ_SCORED

    preds_scored = val_preds_arr[:, :seq_scored, scored_indices]
    targets_scored = val_targets_arr[:, :seq_scored, scored_indices]

    # Calculate RMSE per sample
    # (y - y_hat)^2
    sq_diff = (targets_scored - preds_scored) ** 2
    # Mean over sequence and targets per sample
    mse_per_sample = np.mean(sq_diff, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({"id": all_val_ids, "error_rmse": rmse_per_sample})

    # Load Metadata to get features
    val_meta_df = pd.read_csv(Config.VAL_CSV)

    # Merge
    analysis_df = analysis_df.merge(val_meta_df, on="id", how="left")

    # Feature Engineering for Correlation
    # 1. Signal to Noise
    # 2. Sequence Length (Constant 107, so skip)
    # 3. GC Content
    analysis_df["gc_content"] = analysis_df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )

    # Compute Correlations
    correlations = {}
    features_to_check = ["signal_to_noise", "gc_content", "mean_reactivity"]

    print("Correlation between Error (RMSE) and Features:")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs if any
            valid_data = analysis_df[[feat, "error_rmse"]].dropna()
            if len(valid_data) > 0:
                corr, _ = stats.pearsonr(valid_data[feat], valid_data["error_rmse"])
                correlations[feat] = corr
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Insufficient data")
        else:
            print(f"  {feat}: Not found in metadata")

    # --------------------------------------------------------------------------
    # 4. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.5417620723771521

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Prepare list for submission rows
        submission_rows = []

        # Column names for submission
        # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # Target cols in output: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # Indices: 0, 1, 2, 3, 4

        with torch.no_grad():
            for inputs, partner_indices, _, ids in test_loader:
                inputs = inputs.to(device)
                partner_indices = partner_indices.to(device)

                # Inference
                outputs = model(inputs, partner_indices)  # (B, 107, 5)
                outputs_np = outputs.cpu().numpy()

                # Format
                batch_size = len(ids)
                seq_len = outputs_np.shape[1]

                for b in range(batch_size):
                    sample_id = ids[b]
                    sample_preds = outputs_np[b]  # (107, 5)

                    for pos in range(seq_len):
                        # Row ID
                        row_id = f"{sample_id}_{pos}"

                        # Values
                        vals = sample_preds[pos]

                        # Append dict
                        submission_rows.append(
                            {
                                "id_seqpos": row_id,
                                "reactivity": vals[0],
                                "deg_Mg_pH10": vals[1],
                                "deg_pH10": vals[2],
                                "deg_Mg_50C": vals[3],
                                "deg_50C": vals[4],
                            }
                        )

        # Create DataFrame
        submission_df = pd.DataFrame(submission_rows)

        # Save
        os.makedirs("submission", exist_ok=True)
        sub_path = "submission/submission.csv"
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}. Rows: {len(submission_df)}")

    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
