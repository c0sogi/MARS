import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.data_utils import DataUtils
from library.dataset import LungDataset
from library.train_eval import train_model
from library.model import TQSAN
from library.loss import LaplaceLogLikelihoodLoss


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    Config.setup()
    print("Configuration loaded. Device:", Config.device)

    # ==========================================
    # 2. Data Loading & Preparation
    # ==========================================
    print("\nLoading metadata...")
    train_df = pd.read_csv(Config.train_csv)
    val_df = pd.read_csv(Config.val_csv)
    test_df = pd.read_csv(Config.test_csv)

    print("Preparing datasets (this may take time for image processing)...")
    # Prepare dictionaries containing image paths and tensors
    train_data = DataUtils.prepare_dataset(
        train_df, Config.cache_dir, mode="train", load_cached_data=True
    )
    val_data = DataUtils.prepare_dataset(
        val_df, Config.cache_dir, mode="val", load_cached_data=True
    )

    # Instantiate PyTorch Datasets
    train_dataset = LungDataset(train_data, mode="train")
    val_dataset = LungDataset(val_data, mode="val")

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("\nStarting training...")
    # train_model handles the loop, early stopping, and saving the best model
    train_model(train_dataset, val_dataset)

    # ==========================================
    # 4. Validation Assessment
    # ==========================================
    print("\nLoading best model for validation...")
    model = TQSAN()
    model.load_state_dict(
        torch.load(Config.model_save_path, map_location=Config.device)
    )
    model.to(Config.device)
    model.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    criterion = LaplaceLogLikelihoodLoss()
    total_metric = 0.0
    n_samples = 0

    # Store predictions for failure analysis
    val_preds_fvc = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            axial = batch["axial"].to(Config.device)
            coronal = batch["coronal"].to(Config.device)
            meta = batch["meta"].to(Config.device)
            target = batch["target"].to(Config.device)
            dt = batch["dt"].to(Config.device)
            base_fvc = batch["base_fvc"].to(Config.device)

            # Inference
            preds = model(axial, coronal, meta)

            # Compute Metric
            score = criterion.metric(preds, target, dt, base_fvc)

            batch_sz = target.size(0)
            total_metric += score.item() * batch_sz
            n_samples += batch_sz

            # Calculate predicted FVC for analysis: Base + alpha * dt
            alpha = preds[:, 0:1]
            fvc_pred = base_fvc + alpha * dt

            val_preds_fvc.extend(fvc_pred.cpu().numpy().flatten())
            val_targets.extend(target.cpu().numpy().flatten())

    final_metric = total_metric / n_samples
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")
    val_preds_fvc = np.array(val_preds_fvc)
    val_targets = np.array(val_targets)

    # Calculate Absolute Error
    abs_errors = np.abs(val_preds_fvc - val_targets)

    # Create analysis dataframe
    analysis_df = val_df.copy()
    analysis_df["AbsError"] = abs_errors

    # Calculate correlations
    # We select relevant numeric columns available in validation metadata
    corr_features = ["Age", "Percent", "Weeks", "AbsError"]
    if "Baseline_FVC" in analysis_df.columns:
        corr_features.append("Baseline_FVC")

    correlations = (
        analysis_df[corr_features].corr()["AbsError"].sort_values(ascending=False)
    )

    print("Correlation between Absolute Error and Input Features:")
    print(correlations)

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    threshold = -6.510164260864258

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Prepare Test Data
        test_data = DataUtils.prepare_dataset(
            test_df, Config.cache_dir, mode="test", load_cached_data=True
        )
        test_dataset = LungDataset(test_data, mode="test")

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        all_fvc = []
        all_conf = []

        with torch.no_grad():
            for batch in test_loader:
                axial = batch["axial"].to(Config.device)
                coronal = batch["coronal"].to(Config.device)
                meta = batch["meta"].to(Config.device)
                dt = batch["dt"].to(Config.device)
                base_fvc = batch["base_fvc"].to(Config.device)

                # Predict parameters
                preds = model(axial, coronal, meta)

                alpha = preds[:, 0:1]
                sigma_base = preds[:, 1:2]
                sigma_growth = preds[:, 2:3]

                # Calculate FVC and Confidence
                # FVC = Baseline + alpha * time_delta
                fvc_pred = base_fvc + alpha * dt

                # Confidence = Sigma_base + Sigma_growth * |time_delta|
                sigma = sigma_base + sigma_growth * torch.abs(dt)

                # Clip Confidence (min 70)
                sigma_clipped = torch.clamp(sigma, min=70.0)

                all_fvc.extend(fvc_pred.cpu().numpy().flatten())
                all_conf.extend(sigma_clipped.cpu().numpy().flatten())

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "Patient_Week": test_df["Patient_Week"],
                "FVC": all_fvc,
                "Confidence": all_conf,
            }
        )

        # Save to ./submission/submission.csv
        out_dir = "./submission"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "submission.csv")
        submission.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
