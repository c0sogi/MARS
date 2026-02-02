import os
import torch
import numpy as np
import pandas as pd
import scipy.stats

# Import library modules
from library.config import Config
from library.data_loader import get_dataloaders
from library.trainer import Trainer
from library.utils import set_seed, haversine_distance


def analyze_failures(model, loader, device):
    """
    Performs failure analysis on the validation set by correlating prediction errors
    with input features (SatCount, Signal Strength, Uncertainty).
    """
    print("\nRunning Failure Analysis on Validation Set...")
    model.eval()

    errors = []
    feat_sat_counts = []
    feat_mean_cn0 = []
    feat_mean_pr_unc = []

    with torch.no_grad():
        for sat_feat, glob_feat, mask, target, wls_lla in loader:
            # Move to device
            sat_feat_dev = sat_feat.to(device)
            glob_feat_dev = glob_feat.to(device)
            mask_dev = mask.to(device)

            # Reshape for TCN input: (Batch, Seq_Len=1, ...)
            sf_in = sat_feat_dev.unsqueeze(1)
            gf_in = glob_feat_dev.unsqueeze(1)
            m_in = mask_dev.unsqueeze(1)

            # Forward pass
            output = model(sf_in, gf_in, m_in).squeeze(1)  # (Batch, 2)

            # Calculate Distance Error (Haversine)
            # Reconstruct coordinates: WLS + Residual
            wls_np = wls_lla.numpy()
            pred_res = output.cpu().numpy()
            gt_res = target.numpy()

            pred_lat = wls_np[:, 0] + pred_res[:, 0]
            pred_lon = wls_np[:, 1] + pred_res[:, 1]
            gt_lat = wls_np[:, 0] + gt_res[:, 0]
            gt_lon = wls_np[:, 1] + gt_res[:, 1]

            dists = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
            errors.extend(dists)

            # Extract Features for Correlation
            # Note: Features are normalized, but correlation is invariant to linear scaling.

            # 1. Satellite Count (Global Feature Index 0)
            sat_counts = glob_feat[:, 0].numpy()
            feat_sat_counts.extend(sat_counts)

            # 2. Mean Cn0 (Sat Feature Index 0) and 3. Mean PrUnc (Sat Feature Index 4)
            # We must respect the mask to calculate the mean over valid satellites only
            mask_np = mask.numpy()
            sat_feat_np = sat_feat.numpy()

            # Avoid division by zero
            valid_counts = mask_np.sum(axis=1)
            valid_counts[valid_counts == 0] = 1.0

            # Cn0
            cn0_vals = sat_feat_np[:, :, 0]
            mean_cn0 = (cn0_vals * mask_np).sum(axis=1) / valid_counts
            feat_mean_cn0.extend(mean_cn0)

            # Pseudorange Uncertainty
            pr_unc_vals = sat_feat_np[:, :, 4]
            mean_pr_unc = (pr_unc_vals * mask_np).sum(axis=1) / valid_counts
            feat_mean_pr_unc.extend(mean_pr_unc)

    # Create DataFrame and compute correlations
    df_analysis = pd.DataFrame(
        {
            "Error": errors,
            "SatCount": feat_sat_counts,
            "Mean_Cn0": feat_mean_cn0,
            "Mean_PrUnc": feat_mean_pr_unc,
        }
    )

    # Calculate Pearson correlation with Error
    correlations = (
        df_analysis.corr()["Error"].drop("Error").sort_values(ascending=False)
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)
    print("-" * 50)


def main():
    # 1. Configure for Fast Baseline
    # Override Config defaults to ensure execution within time limits
    Config.DEBUG = True
    Config.DEBUG_SIZE = 10000  # Use a subset of 10k samples for fast training/val
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 64  # Increase batch size for speed
    Config.NUM_WORKERS = 2  # Moderate worker count
    Config.CACHE_DATA = True  # Use cached npz files if available

    # Set random seed for reproducibility
    set_seed(Config.SEED)

    print("Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Debug Mode: {Config.DEBUG} (Size: {Config.DEBUG_SIZE})")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print("-" * 50)

    # 2. Load Data
    # This handles metadata loading, dataset creation, and normalization
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Initialize Trainer and Model
    trainer = Trainer(train_loader, val_loader, test_loader)

    # 4. Train Model
    trainer.fit()

    # 5. Validation Assessment
    print("\nEvaluating on Validation Set...")
    val_score, val_loss = trainer.evaluate(val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    analyze_failures(trainer.model, val_loader, trainer.device)

    # 7. Submission Generation (Conditional)
    # Threshold defined in task requirements
    SUBMISSION_THRESHOLD = 3.8442371867640412

    if val_score < SUBMISSION_THRESHOLD:
        print(
            f"Validation score ({val_score}) is better than threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Generating submission...")
        trainer.predict()
    else:
        print(
            f"Validation score ({val_score}) did not meet threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
