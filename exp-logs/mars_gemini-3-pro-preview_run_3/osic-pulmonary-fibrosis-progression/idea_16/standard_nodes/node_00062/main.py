import sys
import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss
from library.data import get_dataloaders
from library.model import RaliNet
from library.train import train_one_epoch, evaluate, generate_submission


def analyze_failures(model, loader, device, stats):
    """
    Performs failure analysis on the validation set by correlating
    absolute prediction errors with input features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()
    all_preds = []

    # Extract scaling stats for inverse transformation
    fvc_std = stats["fvc_std"]
    fvc_mean = stats["fvc_mean"]

    # Collect predictions
    with torch.no_grad():
        for images, tabular, targets in loader:
            images = images.to(device)
            tabular = tabular.to(device)

            preds = model(images, tabular)

            # Inverse transform FVC prediction (Column 0)
            fvc_pred_std = preds[:, 0].cpu().numpy()
            fvc_pred_real = fvc_pred_std * fvc_std + fvc_mean
            all_preds.extend(fvc_pred_real)

    # Get the dataframe from the dataset
    # Note: val_loader is not shuffled, so order matches the dataframe
    df = loader.dataset.df.copy()

    # Ensure alignment
    if len(df) != len(all_preds):
        print(
            f"Warning: Dataframe length ({len(df)}) mismatch with predictions ({len(all_preds)})"
        )
        return

    df["Pred_FVC"] = all_preds
    df["Abs_Error"] = (df["FVC"] - df["Pred_FVC"]).abs()

    # Calculate correlations
    features_to_check = ["Age", "Weeks", "Base_FVC", "Percent"]
    print("Correlation between Absolute Error and Features:")

    for feat in features_to_check:
        if feat in df.columns:
            corr = df[feat].corr(df["Abs_Error"])
            print(f"  {feat}: {corr:.6f}")
        else:
            print(f"  {feat}: Not found in metadata")


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using debug=False to use full dataset (small enough for fast execution)
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=False,
        load_cached_data=True,
    )

    # 3. Model Initialization
    model = RaliNet().to(device)

    # 4. Optimizer & Scheduler
    # Differential Learning Rates
    backbone_ids = list(map(id, model.backbone.parameters()))
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLogLikelihoodLoss()

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_metric = -float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metric = evaluate(model, val_loader, criterion, device, stats)
        scheduler.step()

        # Save checkpoint if improved
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

        # Optional: Print progress occasionally or every epoch
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Metric: {val_metric:.6f}"
            )

    print("Training complete.")

    # 6. Final Evaluation
    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Compute metric on full validation set
    _, final_metric = evaluate(model, val_loader, criterion, device, stats)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device, stats)

    # 8. Conditional Submission
    submission_threshold = -6.573619738753321

    if final_metric > submission_threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({submission_threshold}). Generating submission..."
        )
        generate_submission(model, test_loader, device, stats)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({submission_threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
