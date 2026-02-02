import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.data import get_dataloaders
from library.model import SCOSRNet
from library.train import train_one_epoch, validate, run_inference

# Monkey patch Config for fast baseline execution
# Increasing epochs slightly to allow convergence with smooth loss
Config.EPOCHS = 25


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # The get_dataloaders function handles loading metadata and creating loaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Retrieve normalization stats needed for training/validation
    stats = train_loader.dataset.stats
    fvc_mean = stats.get("fvc_mean", 2500.0)
    fvc_std = stats.get("fvc_std", 500.0)

    # 3. Model Initialization
    model = SCOSRNet().to(device)

    # 4. Optimizer & Scheduler Setup
    # Differential learning rates as per library.train logic
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
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_metric = -float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, fvc_mean, fvc_std
        )
        # Validate
        val_metric = validate(model, val_loader, device, fvc_mean, fvc_std)

        scheduler.step()

        # Save Best Model
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 6. Final Validation & Failure Analysis
    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    model.eval()

    val_preds_mu = []
    val_preds_sigma = []
    val_targets = []

    # Lists to store metadata for correlation analysis
    meta_percent = []
    meta_weeks = []
    meta_age = []
    meta_sex = []
    meta_smoking = []

    with torch.no_grad():
        for images, clinical, targets, meta in val_loader:
            images = images.to(device)
            clinical = clinical.to(device)
            targets = targets.to(device)

            mu_final, sigma_final, _, _ = model(images, clinical)

            # Denormalize
            mu_dn = mu_final * fvc_std + fvc_mean
            sigma_dn = sigma_final * fvc_std
            targets_dn = targets * fvc_std + fvc_mean

            val_preds_mu.append(mu_dn.cpu())
            val_preds_sigma.append(sigma_dn.cpu())
            val_targets.append(targets_dn.cpu())

            # Extract metadata features for failure analysis
            # Clinical tensor structure: [Age(norm), Sex, Smoke, RelTime]
            clin_cpu = clinical.cpu().numpy()

            # meta items are tensors/lists from the dataloader collate
            meta_percent.extend(meta["Percent"].numpy())
            meta_weeks.extend(meta["Weeks"].numpy())
            meta_age.extend(clin_cpu[:, 0])  # Normalized age
            meta_sex.extend(clin_cpu[:, 1])
            meta_smoking.extend(clin_cpu[:, 2])

    # Concatenate results
    all_mu = torch.cat(val_preds_mu)
    all_sigma = torch.cat(val_preds_sigma)
    all_targets = torch.cat(val_targets)

    # 7. Compute Final Metric
    # Using the library function to ensure consistency with competition metric
    final_metric = laplace_log_likelihood(all_targets, all_mu, all_sigma).item()
    print(f"Final Validation Metric: {final_metric}")

    # 8. Failure Analysis
    # Calculate absolute error
    abs_error = torch.abs(all_targets - all_mu).numpy().flatten()

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "AbsError": abs_error,
            "Percent": meta_percent,
            "Weeks": meta_weeks,
            "Age": meta_age,
            "Sex": meta_sex,
            "Smoking": meta_smoking,
        }
    )

    print("Failure Analysis (Correlation with AbsError):")
    features = ["Percent", "Weeks", "Age", "Sex", "Smoking"]
    for feat in features:
        if feat in analysis_df.columns:
            # Check for variance to avoid warnings
            if analysis_df[feat].nunique() > 1:
                corr, _ = pearsonr(analysis_df[feat], analysis_df["AbsError"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: NaN (Constant value)")

    # 9. Submission
    # Threshold defined in task description
    metric_threshold = -6.573619738753321

    if final_metric > metric_threshold:
        run_inference(test_loader, device)


if __name__ == "__main__":
    main()
