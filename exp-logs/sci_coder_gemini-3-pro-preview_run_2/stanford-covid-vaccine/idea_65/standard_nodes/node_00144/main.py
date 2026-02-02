import os
import pandas as pd
import numpy as np
import torch
import scipy.stats as stats

from library.config import SEED, WORKING_DIR, SUBMISSION_PATH, SCORED_LEN, device
from library.utils import seed_everything
from library.engine import fit, inference, validate
from library.data import get_dataloaders
from library.model import SSRFN


def main():
    # 1. Setup
    seed_everything(SEED)
    print("Initializing Fast Baseline Pipeline...")

    # 2. Training
    # We limit epochs to 10 for a fast baseline execution as requested.
    print("Starting Training...")
    best_model_path, test_loader = fit(
        epochs=10, load_cached_data=True, working_dir=WORKING_DIR
    )

    # 3. Validation & Metric Calculation
    print("Loading best model for validation...")

    # Re-initialize model and load weights
    model = SSRFN().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get validation loader explicitly for analysis
    _, val_loader, _ = get_dataloaders(working_dir=WORKING_DIR, load_cached_data=True)

    # Compute Final Metric
    val_score = validate(model, val_loader, device)
    print(f"Final Validation Metric: {val_score}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")

    all_preds = []
    all_targets = []

    # Collect predictions for analysis
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # SSRFN returns final prediction (y2) in eval mode
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Slice to scored length and columns [reactivity, deg_Mg_pH10, deg_Mg_50C]
    # Indices: 0, 1, 3
    valid_preds = all_preds[:, :SCORED_LEN, [0, 1, 3]]
    valid_targets = all_targets[:, :SCORED_LEN, [0, 1, 3]]

    # Compute RMSE per sample (averaging over length and channels)
    # Shape: (N_samples,)
    mse_per_sample = np.mean((valid_preds - valid_targets) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load metadata to correlate errors with features
    val_meta_path = "./metadata/val.csv"
    if os.path.exists(val_meta_path):
        val_df = pd.read_csv(val_meta_path)

        # Ensure alignment
        if len(val_df) == len(rmse_per_sample):
            val_df["rmse_error"] = rmse_per_sample

            features_to_analyze = ["signal_to_noise", "mean_reactivity", "SN_filter"]
            print("Correlation between Model Error (RMSE) and Input Features:")

            for feat in features_to_analyze:
                if feat in val_df.columns:
                    # Handle potential NaNs just in case
                    valid_mask = val_df[feat].notna() & val_df["rmse_error"].notna()
                    if valid_mask.sum() > 1:
                        corr, _ = stats.pearsonr(
                            val_df.loc[valid_mask, feat],
                            val_df.loc[valid_mask, "rmse_error"],
                        )
                        print(f"  {feat}: {corr:.4f}")
        else:
            print(
                f"Warning: Metadata length ({len(val_df)}) does not match prediction length ({len(rmse_per_sample)}). Skipping correlation analysis."
            )
    else:
        print(f"Warning: Metadata file not found at {val_meta_path}")

    # 5. Submission
    THRESHOLD = 0.47142532743789534

    if val_score < THRESHOLD:
        print(
            f"\nValidation score ({val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        inference(best_model_path, test_loader, submission_path=SUBMISSION_PATH)
    else:
        print(
            f"\nValidation score ({val_score}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
