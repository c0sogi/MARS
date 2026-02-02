import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import scipy.stats as stats
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import UDSRNet
from library.train import (
    train_one_epoch,
    validate,
    LaplaceLogLikelihoodLoss,
    generate_submission,
)


def run():
    # 1. Configuration & Setup
    Config.setup()

    # Override submission paths to match competition/task requirements
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    Config.SUBMISSION_DIR = submission_dir
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Fast Baseline Settings: Limit epochs to ensure quick execution
    Config.EPOCHS = 10

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, scalers = get_dataloaders(
        load_cached_data=True
    )
    target_scaler = scalers["target_scaler"]

    # 3. Model Initialization
    print("Initializing UDSRNet...")
    model = UDSRNet().to(device)

    # 4. Optimizer & Scheduler Setup
    # Separate parameters for differential learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = LaplaceLogLikelihoodLoss()

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_score = -float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = validate(
            model, val_loader, criterion, device, target_scaler
        )

        scheduler.step()

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training complete. Best Validation Score during training: {best_score}")

    # 6. Final Validation & Metric
    # Load the best model to ensure we evaluate the optimal state
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Compute final metric on the full validation set
    _, final_val_score = validate(model, val_loader, criterion, device, target_scaler)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_score}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")
    model.eval()

    all_errors = []
    all_tabular = []

    # Parameters for inverse scaling
    scale = target_scaler.scale_[0]
    mean = target_scaler.mean_[0]

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            target_raw = batch["fvc_raw"].to(device)  # Ground truth FVC (unscaled)

            mu_scaled, _ = model(imgs, tabular)

            # Inverse transform prediction to original scale
            mu_real = mu_scaled.cpu().numpy() * scale + mean
            y_true = target_raw.cpu().numpy()

            # Calculate absolute error
            errors = np.abs(y_true - mu_real)
            all_errors.extend(errors)
            all_tabular.extend(tabular.cpu().numpy())

    all_errors = np.array(all_errors)
    all_tabular = np.array(all_tabular)

    # Feature names corresponding to the tensor order in OSICDataset
    # [Baseline_FVC_Scaled, Relative_Time, Age_Scaled, Sex_Code, Smoking_Code]
    feature_names = ["Baseline_FVC", "Relative_Time", "Age", "Sex", "Smoking"]

    print("Correlation between Absolute Error and Input Features:")
    for i, name in enumerate(feature_names):
        feat_vals = all_tabular[:, i]
        # Check for constant arrays to avoid warnings
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr, _ = stats.pearsonr(all_errors, feat_vals)
        print(f"  {name}: {corr:.4f}")

    # 8. Submission
    # Threshold defined in task
    threshold = -6.573619738753321

    if final_val_score > threshold:
        print(
            f"\nValidation score ({final_val_score}) exceeds threshold ({threshold}). Generating submission..."
        )
        generate_submission(model, scalers, device)
    else:
        print(
            f"\nValidation score ({final_val_score}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
