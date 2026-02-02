import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihood
from library.data import get_dataloaders
from library.model import EADSNet
from library.train import train_one_epoch, evaluate


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Override Config for fast baseline execution
    Config.EPOCHS = 25

    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # load_cached_data=True uses the pre-processed .npy files in ./working
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    model = EADSNet().to(device)

    # 4. Optimizer & Scheduler (Differential Learning Rates)
    backbone_params = list(model.image_encoder.backbone.parameters())
    backbone_ids = list(map(id, backbone_params))
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LEARNING_RATE_BACKBONE},
            {"params": head_params, "lr": Config.LEARNING_RATE_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_score = -float("inf")

    # We suppress most printing to comply with "Only print required info"
    # but we will print the final result.

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate (using library function which computes the official metric)
        val_score = evaluate(model, val_loader, device, target_scaler)

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 6. Final Evaluation & Failure Analysis
    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    model.eval()

    # We need to gather predictions and metadata for failure analysis
    # The library evaluate() only returns the score, so we run a custom inference loop here.

    val_preds_mu = []
    val_preds_sigma = []
    val_targets = []
    val_patient_weeks = []

    # We also need features for correlation.
    # Since val_loader is not shuffled, we can align with the dataframe,
    # but extracting from batch is safer.
    val_features = {"Age": [], "Baseline_FVC": [], "Relative_Weeks": []}

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            time_rel = batch["time"].to(device)

            # Forward
            mu_scaled, sigma_scaled = model(images, tabular, time_rel)

            # Inverse Transform
            mu_orig = target_scaler.inverse_transform(mu_scaled.cpu()).flatten().numpy()
            sigma_orig = (
                target_scaler.inverse_transform_sigma(sigma_scaled.cpu())
                .flatten()
                .numpy()
            )

            val_preds_mu.extend(mu_orig)
            val_preds_sigma.extend(sigma_orig)
            val_targets.extend(batch["raw_fvc"].numpy())
            val_patient_weeks.extend(batch["patient_week"])

            # Extract features for analysis
            # tabular is [Baseline_FVC, Age, Sex, Smoking] (already scaled/encoded)
            # We want original values for interpretation, but scaled is fine for correlation magnitude.
            # However, let's try to get raw values if possible or just use scaled.
            # Correlation is scale-invariant, so scaled values are fine.
            val_features["Baseline_FVC"].extend(tabular[:, 0].cpu().numpy())
            val_features["Age"].extend(tabular[:, 1].cpu().numpy())
            val_features["Relative_Weeks"].extend(time_rel.cpu().flatten().numpy())

    # Calculate Final Metric
    y_true = np.array(val_targets)
    y_pred = np.array(val_preds_mu)
    sigma = np.array(val_preds_sigma)

    final_metric = LaplaceLogLikelihood(y_true, y_pred, sigma)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nFailure Analysis (Correlation with Absolute Error):")
    abs_error = np.abs(y_true - y_pred)

    # Load val_df to get 'Percent' which might not be in the model input but is useful for analysis
    val_df = pd.read_csv(Config.VAL_CSV)
    # Ensure alignment: val_loader is no-shuffle, so order should match val_df
    # We can double check length
    if len(val_df) == len(abs_error):
        # Calculate correlations
        # 1. Age
        corr_age, _ = pearsonr(abs_error, val_features["Age"])
        print(f"  Error vs Age: {corr_age:.4f}")

        # 2. Baseline FVC
        corr_base, _ = pearsonr(abs_error, val_features["Baseline_FVC"])
        print(f"  Error vs Baseline FVC: {corr_base:.4f}")

        # 3. Relative Weeks
        corr_time, _ = pearsonr(abs_error, val_features["Relative_Weeks"])
        print(f"  Error vs Relative Time: {corr_time:.4f}")

        # 4. Percent (from DF)
        if "Percent" in val_df.columns:
            corr_pct, _ = pearsonr(abs_error, val_df["Percent"].values)
            print(f"  Error vs Percent: {corr_pct:.4f}")
    else:
        print(
            "  Warning: Validation DataFrame length mismatch. Skipping detailed feature analysis."
        )

    # 7. Submission
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        submission_rows = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                tabular = batch["tabular"].to(device)
                time_rel = batch["time"].to(device)
                patient_weeks = batch["patient_week"]

                mu_scaled, sigma_scaled = model(images, tabular, time_rel)

                mu_scaled = mu_scaled.cpu()
                sigma_scaled = sigma_scaled.cpu()

                mu_orig = target_scaler.inverse_transform(mu_scaled).flatten().numpy()
                sigma_orig = (
                    target_scaler.inverse_transform_sigma(sigma_scaled)
                    .flatten()
                    .numpy()
                )

                # Clip confidence at 70ml
                sigma_final = np.maximum(sigma_orig, Config.SIGMA_CLIP)

                for pw, fvc, conf in zip(patient_weeks, mu_orig, sigma_final):
                    submission_rows.append(
                        {"Patient_Week": pw, "FVC": fvc, "Confidence": conf}
                    )

        sub_df = pd.DataFrame(submission_rows)
        sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
