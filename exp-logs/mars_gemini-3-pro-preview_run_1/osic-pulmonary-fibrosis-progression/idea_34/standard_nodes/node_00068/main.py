import os
import sys
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.train import run_training
from library.model import ASADAN
from library.data import get_dataloaders
from library.utils import seed_everything, score


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Override Config for fast baseline execution
    # 25 epochs is sufficient for this small dataset to converge or trigger early stopping
    Config.EPOCHS = 25
    Config.PATIENCE = 6

    # Ensure output directories exist
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    print("Initializing ASA-DAN Pipeline...")

    # ==========================================
    # 2. Training
    # ==========================================
    # run_training handles dataloading, training loop, and saving best model
    # It returns the path to the best model checkpoint
    best_model_path = run_training(debug=False)

    # ==========================================
    # 3. Validation Inference
    # ==========================================
    print("\nRunning Validation Inference...")
    device = torch.device(Config.DEVICE)

    # Load the best model
    model = ASADAN().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get dataloaders (re-using get_dataloaders to ensure consistency)
    _, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    val_targets = []
    val_preds = []
    val_sigmas = []
    val_meta = []  # To store features for failure analysis

    # Inference loop without gradient calculation
    with torch.no_grad():
        for batch in val_loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            static_features = batch["static_features"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            week = batch["week"].to(device)
            target = batch["target"].to(device)

            fvc_pred, sigma_pred = model(
                img_axial, img_coronal, static_features, baseline_fvc, week
            )

            # Store results on CPU
            val_targets.append(target.cpu().numpy())
            val_preds.append(fvc_pred.cpu().numpy())
            val_sigmas.append(sigma_pred.cpu().numpy())

            # Collect metadata for failure analysis
            # static_features structure from LungDataset: [Age, Sex, Smoking, Percent]
            # We also want Baseline_FVC and Week for analysis
            feats = static_features.cpu().numpy()
            b_fvc = baseline_fvc.cpu().numpy().reshape(-1, 1)
            wk = week.cpu().numpy().reshape(-1, 1)

            # Concatenate: Age, Sex, Smoking, Percent, Baseline_FVC, Week
            meta_batch = np.hstack([feats, b_fvc, wk])
            val_meta.append(meta_batch)

    # Concatenate all batches
    val_targets = np.concatenate(val_targets)
    val_preds = np.concatenate(val_preds)
    val_sigmas = np.concatenate(val_sigmas)
    val_meta = np.concatenate(val_meta)

    # ==========================================
    # 4. Metric Calculation & Failure Analysis
    # ==========================================
    # Calculate Final Metric using the official scoring function
    final_metric = score(
        val_targets,
        val_preds,
        val_sigmas,
        max_error=Config.MAX_ERROR,
        confidence_clip=Config.CONFIDENCE_CLIP,
    )

    print(f"Final Validation Metric: {final_metric}")

    print("\nFailure Analysis (Correlation with Absolute Error):")
    abs_errors = np.abs(val_targets - val_preds)

    # Create DataFrame for correlation analysis
    columns = ["Age", "Sex", "Smoking", "Percent", "Baseline_FVC", "Week"]
    df_analysis = pd.DataFrame(val_meta, columns=columns)
    df_analysis["Error"] = abs_errors

    # Calculate correlation of features with the error magnitude
    correlations = df_analysis.corr()["Error"].sort_values(ascending=False)
    # Drop the self-correlation
    correlations = correlations.drop("Error")
    print(correlations)

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = []
        test_sigmas = []

        with torch.no_grad():
            for batch in test_loader:
                img_axial = batch["img_axial"].to(device)
                img_coronal = batch["img_coronal"].to(device)
                static_features = batch["static_features"].to(device)
                baseline_fvc = batch["baseline_fvc"].to(device)
                week = batch["week"].to(device)

                fvc_pred, sigma_pred = model(
                    img_axial, img_coronal, static_features, baseline_fvc, week
                )

                test_preds.append(fvc_pred.cpu().numpy())
                test_sigmas.append(sigma_pred.cpu().numpy())

        test_preds = np.concatenate(test_preds)
        test_sigmas = np.concatenate(test_sigmas)

        # Load test metadata to get Patient_Week IDs
        # The loader preserves order (shuffle=False), so we can directly assign predictions
        df_test = pd.read_csv(Config.TEST_CSV)

        submission = pd.DataFrame(
            {
                "Patient_Week": df_test["Patient_Week"],
                "FVC": test_preds,
                "Confidence": test_sigmas,
            }
        )

        # Ensure directory exists and save
        os.makedirs("./submission", exist_ok=True)
        save_path = "./submission/submission.csv"
        submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
