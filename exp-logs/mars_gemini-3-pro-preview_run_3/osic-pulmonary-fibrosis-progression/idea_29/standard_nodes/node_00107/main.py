import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, CTDataset
from library.model import MAOPDSNet
from library.train import run_training


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Override Config for fast baseline execution
    Config.EPOCHS = 20

    print(f"Starting execution with {Config.EPOCHS} epochs...")

    # 2. Run Training
    # This will train the model and save the best version to Config.MODEL_PATH
    run_training(patience=5)

    # 3. Load Best Model and Data for Validation
    device = torch.device(Config.DEVICE)
    train_loader, val_loader, test_loader, stats = get_dataloaders()

    model = MAOPDSNet().to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 4. Validation Inference & Metric Calculation
    print("\nRunning Validation Inference...")
    all_targets_raw = []
    all_mu_raw = []
    all_sigma_raw = []

    # For failure analysis
    val_errors = []
    val_features = {
        "Age": [],
        "Baseline_FVC": [],
        "Weeks": [],
        "Sex": [],
        "Smoking": [],
    }

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)

            outputs = model(images, tabular)

            # Extract predictions
            mu_scaled = outputs[:, 0].cpu().numpy()
            raw_sigma_scaled = outputs[:, 1]
            sigma_scaled = F.softplus(raw_sigma_scaled).cpu().numpy() + 1e-6
            targets_scaled = targets.cpu().numpy()

            # Inverse Transform
            mu_raw = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_raw = sigma_scaled * Config.TARGET_STD
            targets_raw = targets_scaled * Config.TARGET_STD + Config.TARGET_MEAN

            all_mu_raw.extend(mu_raw)
            all_sigma_raw.extend(sigma_raw)
            all_targets_raw.extend(targets_raw)

            # Collect data for failure analysis
            # Tabular input structure: [base_fvc_scaled, t_rel, age_scaled, sex, smoke]
            tabular_np = tabular.cpu().numpy()

            # Reconstruct raw features roughly for correlation (scaling doesn't affect correlation magnitude much, but let's be clean)
            # We can just use the scaled values for correlation analysis as linear transform doesn't change Pearson corr.
            val_features["Baseline_FVC"].extend(tabular_np[:, 0])
            val_features["Weeks"].extend(tabular_np[:, 1])  # This is t_rel
            val_features["Age"].extend(tabular_np[:, 2])
            val_features["Sex"].extend(tabular_np[:, 3])
            val_features["Smoking"].extend(tabular_np[:, 4])

            # Calculate errors
            batch_errors = np.abs(targets_raw - mu_raw)
            val_errors.extend(batch_errors)

    # Calculate Final Metric
    final_metric = calculate_metric(
        np.array(all_targets_raw), np.array(all_mu_raw), np.array(all_sigma_raw)
    )

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nFailure Analysis (Correlation with Absolute Error):")
    val_errors = np.array(val_errors)
    for feature_name, values in val_features.items():
        values = np.array(values)
        if len(np.unique(values)) > 1:
            corr = np.corrcoef(values, val_errors)[0, 1]
            print(f"  {feature_name}: {corr:.4f}")
        else:
            print(f"  {feature_name}: N/A (Constant)")

    # 6. Submission Generation
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.4f}) > Threshold ({THRESHOLD:.4f}). Generating submission..."
        )

        # Load Sample Submission and Test Metadata
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)
        test_meta = pd.read_csv(Config.TEST_CSV)

        # Parse Patient_Week
        sample_sub["Patient"] = sample_sub["Patient_Week"].apply(
            lambda x: x.split("_")[0]
        )
        sample_sub["Weeks"] = sample_sub["Patient_Week"].apply(
            lambda x: int(x.split("_")[1])
        )

        # Merge with Test Metadata to get static features (Age, Sex, Smoking, Baseline FVC, image_path)
        # Note: In test set, the provided FVC is the Baseline FVC.
        test_meta_renamed = test_meta.rename(columns={"FVC": "Baseline_FVC"})

        # We need to preserve the order of sample_submission
        submission_df = sample_sub.merge(test_meta_renamed, on="Patient", how="left")

        # Create Dataset and Loader for Submission
        # We use the same stats from training for normalization
        sub_ds = CTDataset(submission_df, mode="test", cache=True, stats=stats)
        sub_loader = DataLoader(
            sub_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        sub_preds_mu = []
        sub_preds_sigma = []

        with torch.no_grad():
            for batch in sub_loader:
                images = batch["image"].to(device)
                tabular = batch["tabular"].to(device)

                outputs = model(images, tabular)

                mu_scaled = outputs[:, 0].cpu().numpy()
                raw_sigma_scaled = outputs[:, 1]
                sigma_scaled = F.softplus(raw_sigma_scaled).cpu().numpy() + 1e-6

                # Inverse Transform
                mu_raw = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN
                sigma_raw = sigma_scaled * Config.TARGET_STD

                sub_preds_mu.extend(mu_raw)
                sub_preds_sigma.extend(sigma_raw)

        # Format Submission
        sample_sub["FVC"] = sub_preds_mu
        sample_sub["Confidence"] = sub_preds_sigma

        # Apply Clipping Constraints
        sample_sub["Confidence"] = sample_sub["Confidence"].apply(lambda x: max(x, 70))

        # Select final columns
        final_submission = sample_sub[["Patient_Week", "FVC", "Confidence"]]

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        final_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric:.4f}) <= Threshold ({THRESHOLD:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
