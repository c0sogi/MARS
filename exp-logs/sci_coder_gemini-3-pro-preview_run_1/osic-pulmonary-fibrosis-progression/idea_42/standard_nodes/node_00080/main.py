import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# 1. Configuration Overrides for Fast Baseline
# We modify the config module before importing other components to limit runtime
import library.config as config

config.EPOCHS = 15  # Reduced from 50 to 15 for fast baseline execution
config.PATIENCE = 5  # Stricter early stopping to save time
# config.DEBUG remains False to ensure we use the full dataset for accurate metric calculation

# 2. Import Library Modules
from library.train import train_model
from library.model import CRHDAN
from library.data import get_dataloaders
from library.utils import seed_everything, laplace_log_likelihood_loss
from library.config import DEVICE, SEED, WORKING_DIR


def main():
    # Set fixed seeds for full reproducibility
    seed_everything(SEED)

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # ---------------------------------------------------------
    # 1. Train the Model
    # ---------------------------------------------------------
    print("Initializing training pipeline...")
    # train_model() handles the training loop, validation monitoring, and saving the best checkpoint.
    # It returns the absolute path to the best model weights.
    best_model_path = train_model()

    if not os.path.exists(best_model_path):
        print("Error: Best model file not found after training.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\nLoading best model for validation evaluation...")

    # Initialize model and load weights
    model = CRHDAN().to(DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # Retrieve dataloaders (re-using the library function)
    # We ignore train_loader here as we are in evaluation mode
    _, val_loader, test_loader = get_dataloaders()

    # Containers for accumulating batch results
    val_targets = []
    val_preds_fvc = []
    val_preds_sigma = []

    # Containers for metadata to perform failure analysis
    meta_data = {"Age": [], "Percent": [], "Weeks_Diff": [], "Baseline_FVC": []}

    print("Running inference on hold-out validation set...")
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch: (axial, coronal, tab_vec, weeks_diff, base_fvc, target_fvc)
            axial = batch[0].to(DEVICE)
            coronal = batch[1].to(DEVICE)
            tab_vec = batch[2].to(DEVICE)
            weeks_diff = batch[3].to(DEVICE)
            base_fvc = batch[4].to(DEVICE)
            target_fvc = batch[5].to(DEVICE)

            # Forward pass (Inference)
            pred_fvc, pred_sigma = model(axial, coronal, tab_vec, weeks_diff, base_fvc)

            # Store predictions and targets
            val_targets.append(target_fvc)
            val_preds_fvc.append(pred_fvc)
            val_preds_sigma.append(pred_sigma)

            # Store metadata for correlation analysis
            # tab_vec structure: [Age(norm), Percent(norm), Sex, Smoke]
            meta_data["Age"].append(tab_vec[:, 0].cpu())
            meta_data["Percent"].append(tab_vec[:, 1].cpu())
            meta_data["Weeks_Diff"].append(weeks_diff.cpu())
            meta_data["Baseline_FVC"].append(base_fvc.cpu())

    # Concatenate results from all batches
    val_targets = torch.cat(val_targets)
    val_preds_fvc = torch.cat(val_preds_fvc)
    val_preds_sigma = torch.cat(val_preds_sigma)

    # Compute Final Metric
    # The loss function returns the negative metric (Loss), so we negate it.
    final_loss = laplace_log_likelihood_loss(
        val_targets, val_preds_fvc, val_preds_sigma
    )
    final_metric = -final_loss.item()

    # Print the metric in the required format
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n=== Failure Analysis ===")
    # Calculate absolute prediction errors
    errors = torch.abs(val_targets - val_preds_fvc).cpu().numpy()

    # Flatten metadata lists
    flat_meta = {k: torch.cat(v).cpu().numpy() for k, v in meta_data.items()}

    print("Correlation between Absolute Error and Input Features:")
    for feat_name, feat_vals in flat_meta.items():
        # Compute Pearson correlation if the feature has variance
        if len(np.unique(feat_vals)) > 1:
            corr, _ = pearsonr(errors, feat_vals)
            print(f"  {feat_name}: Pearson r = {corr:.4f}")
        else:
            print(f"  {feat_name}: Constant value (N/A)")

    # ---------------------------------------------------------
    # 3. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        submission_data = []

        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch: (axial, coronal, tab_vec, weeks_diff, base_fvc, pat_week_id)
                axial = batch[0].to(DEVICE)
                coronal = batch[1].to(DEVICE)
                tab_vec = batch[2].to(DEVICE)
                weeks_diff = batch[3].to(DEVICE)
                base_fvc = batch[4].to(DEVICE)
                pat_week_ids = batch[5]  # Tuple of strings

                # Forward pass
                pred_fvc, pred_sigma = model(
                    axial, coronal, tab_vec, weeks_diff, base_fvc
                )

                # Move to CPU for processing
                pred_fvc_np = pred_fvc.cpu().numpy()
                pred_sigma_np = pred_sigma.cpu().numpy()

                # Format rows for CSV
                for i in range(len(pat_week_ids)):
                    submission_data.append(
                        {
                            "Patient_Week": pat_week_ids[i],
                            "FVC": pred_fvc_np[i],
                            "Confidence": pred_sigma_np[i],
                        }
                    )

        # Create DataFrame and save
        sub_df = pd.DataFrame(submission_data)
        sub_df = sub_df[
            ["Patient_Week", "FVC", "Confidence"]
        ]  # Ensure correct column order

        save_path = "./submission/submission.csv"
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved successfully to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
