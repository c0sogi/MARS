import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

# Import library modules
from library.config import Config
from library import utils, data, model, train


def main():
    # 1. Setup
    print("--- Setting up environment ---")
    Config.setup()
    utils.seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Train Model
    print("\n--- Starting Training ---")
    # We run the training module which handles the training loop and saves the best model
    # The dataset is small (~1k rows), so 50 epochs is very fast (minutes).
    best_metric_score = train.run_training(debug=False)

    # 3. Validation Inference & Failure Analysis
    print("\n--- Running Validation Inference ---")

    # Load Data Loaders
    _, val_loader, sub_loader, sample_sub = data.get_dataloaders(debug=False)

    # Load Best Model
    net = model.CLRNet().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    net.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)
    )
    net.eval()

    val_preds_mu = []
    val_preds_sigma = []
    val_targets = []

    # Inference Loop (No Grad)
    with torch.no_grad():
        for imgs, clin_data, targets in val_loader:
            imgs = imgs.to(device)
            clin_data = clin_data.to(device)

            # Forward pass
            preds = net(imgs, clin_data)

            mu_std = preds[:, 0]
            raw_sigma = preds[:, 1]

            # Softplus for positivity (standardized space)
            sigma_std = F.softplus(raw_sigma) + 1e-6

            val_preds_mu.extend(mu_std.cpu().numpy())
            val_preds_sigma.extend(sigma_std.cpu().numpy())
            val_targets.extend(targets.numpy().flatten())

    # Inverse Transform
    val_preds_mu = np.array(val_preds_mu)
    val_preds_sigma = np.array(val_preds_sigma)
    val_targets = np.array(val_targets)

    pred_mu_ml, pred_sigma_ml = utils.inverse_transform(val_preds_mu, val_preds_sigma)
    target_ml = val_targets * Config.TARGET_STD + Config.TARGET_MEAN

    # Calculate Final Metric
    final_metric = utils.calculate_metric(target_ml, pred_mu_ml, pred_sigma_ml)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    abs_error = np.abs(target_ml - pred_mu_ml)

    # Get metadata features from the validation dataset dataframe
    # The loader preserves order, so we can map directly
    val_df = val_loader.dataset.df.copy()
    val_df["Abs_Error"] = abs_error
    val_df["Pred_FVC"] = pred_mu_ml
    val_df["Pred_Sigma"] = pred_sigma_ml

    # Features to analyze
    analysis_cols = ["Weeks", "Percent", "Age", "Base_FVC"]

    print("Correlation between Absolute Error and Features:")
    correlations = (
        val_df[analysis_cols + ["Abs_Error"]].corr()["Abs_Error"].drop("Abs_Error")
    )
    print(correlations)

    # 4. Submission Generation
    threshold = -6.573619738753321
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        sub_preds_mu = []
        sub_preds_sigma = []

        with torch.no_grad():
            for imgs, clin_data, _ in sub_loader:
                imgs = imgs.to(device)
                clin_data = clin_data.to(device)

                preds = net(imgs, clin_data)

                mu_std = preds[:, 0]
                raw_sigma = preds[:, 1]
                sigma_std = F.softplus(raw_sigma) + 1e-6

                sub_preds_mu.extend(mu_std.cpu().numpy())
                sub_preds_sigma.extend(sigma_std.cpu().numpy())

        # Inverse Transform
        sub_preds_mu = np.array(sub_preds_mu)
        sub_preds_sigma = np.array(sub_preds_sigma)

        pred_mu_ml, pred_sigma_ml = utils.inverse_transform(
            sub_preds_mu, sub_preds_sigma
        )

        # Apply Submission-Specific Clipping
        # "confidence values are clipped at 70 ml"
        pred_sigma_final = np.maximum(pred_sigma_ml, 70.0)

        # Construct Submission DataFrame
        # sample_sub is already ordered correctly matching sub_loader
        submission = pd.DataFrame(
            {
                "Patient_Week": sample_sub["Patient_Week"],
                "FVC": pred_mu_ml,
                "Confidence": pred_sigma_final,
            }
        )

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission.head())

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
