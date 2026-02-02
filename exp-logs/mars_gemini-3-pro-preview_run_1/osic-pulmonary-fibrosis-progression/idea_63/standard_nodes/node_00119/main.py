import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, AverageMeter
from library.data import get_dataloaders
from library.model import BBSLNet
from library.engine import train_fn, eval_fn, inference_fn


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between Error Magnitude and Input Features.
    """
    model.eval()

    data_records = []

    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            time_delta = batch["time_delta"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            preds = model(img_ax, img_cor, tabular, time_delta, baseline_fvc)

            # Calculate Absolute Error
            fvc_pred = preds[:, 0]
            abs_error = torch.abs(fvc_pred - targets).cpu().numpy()

            # Extract metadata for correlation
            # Tabular is [Age_norm, Sex, Smoking, Percent_norm]
            # We need to denormalize or just use as is for correlation (rank correlation is invariant to scale)
            # But let's use the raw values if possible, or just the tensor values.
            # Since we don't have easy access to raw values here without looking up the DF,
            # we will use the tensor values which preserve rank.

            # Tabular features
            tab_np = tabular.cpu().numpy()
            age = tab_np[:, 0]
            percent = tab_np[:, 3]

            # Baseline FVC
            base_fvc_np = baseline_fvc.cpu().numpy()

            # Time Delta
            delta_np = time_delta.cpu().numpy()

            for i in range(len(abs_error)):
                data_records.append(
                    {
                        "Error": abs_error[i],
                        "Age": age[i],
                        "Percent": percent[i],
                        "Baseline_FVC": base_fvc_np[i],
                        "Time_Delta": delta_np[i],
                    }
                )

    df_analysis = pd.DataFrame(data_records)

    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS (Correlation with Error)")
    print("=" * 40)

    features = ["Age", "Percent", "Baseline_FVC", "Time_Delta"]
    for feat in features:
        if feat in df_analysis.columns:
            # Spearman correlation (monotonic relationship)
            corr, _ = spearmanr(df_analysis["Error"], df_analysis[feat])
            print(f"Error vs {feat}: {corr:.4f}")

    print("=" * 40 + "\n")


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = torch.device(Config.device)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, sub_loader = get_dataloaders()

    # 3. Model Initialization
    print("Initializing BBSL-Net...")
    model = BBSLNet()
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.scheduler_T_max
    )

    # Loss Function
    loss_fn = LaplaceLogLikelihoodLoss()

    # 4. Training Loop
    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(train_loader, model, optimizer, device, loss_fn)

        # Validation
        val_loss = eval_fn(val_loader, model, device, loss_fn)

        # Update Scheduler
        scheduler.step()

        # Print progress (optional, but good for log checking)
        # print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Early Stopping & Model Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.model_save_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 5. Final Evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    # Calculate Final Metric on Validation Set
    # Note: The loss function computes -metric. So Metric = -Loss.
    final_loss = eval_fn(val_loader, model, device, loss_fn)
    final_metric = -final_loss

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Conditional Submission
    # Threshold: -6.510164260864258
    # Higher is better.
    threshold = -6.510164260864258

    if final_metric > threshold:
        print(
            f"Metric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )
        inference_fn(sub_loader, model, device)
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({threshold}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
