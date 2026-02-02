import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.loss import MetricAlignedLaplaceLoss
from library.data import get_dataloaders
from library.model import CRDSNet
from library.train import train_one_epoch, evaluate, generate_submission


def perform_failure_analysis(model, loader, device):
    """
    Analyzes model errors on the validation set.
    """
    model.eval()

    all_errors = []
    all_features = []

    target_mean = Config.TARGET_MEAN
    target_std = Config.TARGET_STD

    # Feature names based on OSICDataset implementation
    feature_names = [
        "Baseline_FVC_Scaled",
        "Relative_Time",
        "Age_Scaled",
        "Sex_Code",
        "Smoking_Code",
        "Baseline_Percent_Scaled",
    ]

    with torch.no_grad():
        for images, tabular, targets in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(images, tabular)

            # Inverse transform predictions and targets
            mu_scaled = preds[:, 0].cpu().numpy()
            mu_pred = mu_scaled * target_std + target_mean

            targets_np = targets.cpu().numpy()
            targets_orig = targets_np * target_std + target_mean

            # Calculate Absolute Error
            errors = np.abs(targets_orig - mu_pred)

            all_errors.append(errors)
            all_features.append(tabular.cpu().numpy())

    all_errors = np.concatenate(all_errors)
    all_features = np.concatenate(all_features, axis=0)

    print("\n=== Failure Analysis (Correlation with Absolute Error) ===")
    for i, name in enumerate(feature_names):
        feature_vals = all_features[:, i]
        # Calculate Pearson correlation
        if np.std(feature_vals) > 0 and np.std(all_errors) > 0:
            corr, _ = pearsonr(feature_vals, all_errors)
            print(f"{name}: {corr:.4f}")
        else:
            print(f"{name}: NaN (Constant values)")
    print("========================================================\n")


def main():
    # 1. Setup
    # Override epochs for fast baseline execution
    Config.NUM_EPOCHS = 20
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")
    print(f"Training for {Config.NUM_EPOCHS} epochs...")

    # 2. Data Loading
    # Using load_cached_data=True to speed up execution
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    model = CRDSNet().to(device)

    # 4. Optimizer & Scheduler
    # Differential learning rates
    backbone_ids = list(map(id, model.backbone.parameters()))
    head_params = filter(
        lambda p: id(p) not in backbone_ids and p.requires_grad, model.parameters()
    )
    backbone_params = filter(lambda p: p.requires_grad, model.backbone.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    criterion = MetricAlignedLaplaceLoss()

    # 5. Training Loop
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, device
        )
        val_loss, val_metric = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Simple logging
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Metric: {val_metric:.6f}"
        )

        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, final_metric = evaluate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission
    # Threshold from prompt: -6.573619738753321
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
