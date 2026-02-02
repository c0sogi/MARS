import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, score_function, AverageMeter
from library.data import get_dataloaders
from library.model import MCDSRNet, train_one_epoch, validate


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for fast baseline execution
    # Cite Lesson 00011: Uncertainty calibration requires more epochs
    Config.EPOCHS = 50

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing Fast Baseline Run (Epochs={Config.EPOCHS})...")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    train_loader, val_loader, test_loader, stats = get_dataloaders(debug=Config.DEBUG)

    # -------------------------------------------------------------------------
    # 3. Model & Optimizer Setup
    # -------------------------------------------------------------------------
    model = MCDSRNet().to(device)

    # Differential Learning Rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate (Standard metric check)
        val_loss, val_metric = validate(model, val_loader, device, stats)

        # Step Scheduler
        scheduler.step()

        # Checkpoint
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)

        # Minimal logging
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{Config.EPOCHS} | Val Metric: {val_metric:.6f}")

    print("Training complete.")

    # -------------------------------------------------------------------------
    # 5. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nRunning Validation Assessment & Failure Analysis...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_preds_mu = []
    val_preds_sigma = []
    val_targets = []
    val_features = []

    fvc_mean = stats["FVC_mean"]
    fvc_std = stats["FVC_std"]

    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            tab = batch["tabular"].to(device)
            target_raw = batch["FVC_raw"].to(device)

            # Forward
            mu_scaled, sigma_scaled = model(img, tab)

            # Inverse Transform
            mu_raw = (mu_scaled * fvc_std + fvc_mean).cpu().numpy()
            sigma_raw = (sigma_scaled * fvc_std).cpu().numpy()
            target_np = target_raw.cpu().numpy()
            tab_np = tab.cpu().numpy()

            val_preds_mu.extend(mu_raw)
            val_preds_sigma.extend(sigma_raw)
            val_targets.extend(target_np)
            val_features.extend(tab_np)

    val_preds_mu = np.array(val_preds_mu)
    val_preds_sigma = np.array(val_preds_sigma)
    val_targets = np.array(val_targets)
    val_features = np.array(val_features)

    # Compute Final Metric on whole set
    final_metric = score_function(val_targets, val_preds_mu, val_preds_sigma)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error
    abs_error = np.abs(val_targets - val_preds_mu)

    # Features: [base_FVC_scaled, base_Percent_scaled, t_rel, Age_scaled, Sex_encoded, Smoking_encoded]
    feature_names = ["Base_FVC", "Base_Percent", "Time_Rel", "Age", "Sex", "Smoking"]

    print("Failure Analysis (Correlation with Absolute Error):")
    for i, name in enumerate(feature_names):
        feat_vals = val_features[:, i]
        # Check for constant values to avoid warnings
        if np.std(feat_vals) > 1e-6:
            corr, _ = pearsonr(abs_error, feat_vals)
            print(f"  Error vs {name}: {corr:.4f}")
        else:
            print(f"  Error vs {name}: N/A (Constant)")

    # -------------------------------------------------------------------------
    # 6. Submission Logic
    # -------------------------------------------------------------------------
    threshold = -6.573619738753321

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        results = []
        with torch.no_grad():
            for batch in test_loader:
                img = batch["image"].to(device)
                tab = batch["tabular"].to(device)
                patient_weeks = batch["patient_week"]

                mu_scaled, sigma_scaled = model(img, tab)

                # Inverse Transform
                mu_raw = (mu_scaled * fvc_std + fvc_mean).cpu().numpy()
                sigma_raw = (sigma_scaled * fvc_std).cpu().numpy()

                # Post-processing: Clip sigma strictly for submission
                # The metric uses max(sigma, 70), so we ensure output >= 70
                sigma_raw = np.maximum(sigma_raw, Config.MIN_UNCERTAINTY)

                for pw, fvc, conf in zip(patient_weeks, mu_raw, sigma_raw):
                    results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

        submission_df = pd.DataFrame(results)
        submission_df = submission_df[["Patient_Week", "FVC", "Confidence"]]
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
