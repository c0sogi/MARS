import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.dataset import LungDataset
from library.model import DPSDAN
from library.loss import LaplaceLogLikelihoodLoss
from library.train import train_epoch, validate, predict_and_submit, set_seed


def run():
    # 1. Setup and Configuration
    # Override epochs for fast baseline execution
    Config.EPOCHS = 15

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")
    print(f"Training for {Config.EPOCHS} epochs...")

    # 2. Data Loading
    train_dataset = LungDataset(mode="train")
    val_dataset = LungDataset(mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = DPSDAN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    loss_fn = LaplaceLogLikelihoodLoss()

    # 4. Training Loop
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # Validate
        val_metric = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Metric: {val_metric:.4f}"
        )

        # Save Best
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)

    print(f"\nTraining finished. Best metric: {best_metric:.6f}")

    # 5. Final Validation and Failure Analysis
    print("\nRunning Failure Analysis on Best Model...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Collect predictions and metadata for analysis
    val_results = []

    # Constants for metric
    sqrt_2 = np.sqrt(2)
    max_err = Config.MAX_ERROR
    min_sig = Config.MIN_SIGMA

    with torch.no_grad():
        for batch in val_loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tab_dense = batch["tab_dense"].to(device)

            baseline_fvc = batch["baseline_fvc"].to(device)
            delta_week = batch["delta_week"].to(device)
            target_fvc = batch["target_fvc"].to(device)

            # Metadata for correlation
            # Re-extract from tab_dense or pass through dataset?
            # Easier to use the batch tensors.
            # Note: tab_dense has normalized features.
            # We can use baseline_fvc and derive others or just use what we have.

            alpha, sigma_base, sigma_growth = model(img_axial, img_coronal, tab_dense)

            pred_fvc = baseline_fvc + alpha * delta_week
            pred_sigma = sigma_base + sigma_growth * torch.abs(delta_week)

            # Move to CPU
            p_fvc = pred_fvc.cpu().numpy()
            p_sigma = pred_sigma.cpu().numpy()
            t_fvc = target_fvc.cpu().numpy()
            b_fvc = baseline_fvc.cpu().numpy()
            d_week = delta_week.cpu().numpy()

            # We also want original metadata to correlate (Age, Percent)
            # We can get patient IDs and look up in dataframe, or just use the batch index if aligned.
            # Since shuffle=False, we can align with dataset.df but batching makes it tricky.
            # Let's just use the available tensors and the error.

            for i in range(len(p_fvc)):
                # Metric calculation per sample
                abs_err = np.abs(t_fvc[i] - p_fvc[i])
                delta = min(abs_err, max_err)
                sigma_c = max(p_sigma[i], min_sig)
                metric = -(sqrt_2 * delta / sigma_c) - np.log(sqrt_2 * sigma_c)

                val_results.append(
                    {
                        "Target_FVC": t_fvc[i],
                        "Pred_FVC": p_fvc[i],
                        "Pred_Sigma": p_sigma[i],
                        "Abs_Error": abs_err,
                        "Metric": metric,
                        "Baseline_FVC": b_fvc[i],
                        "Delta_Week": d_week[i],
                    }
                )

    results_df = pd.DataFrame(val_results)

    # Calculate Final Metric on the whole set
    final_metric = results_df["Metric"].mean()
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    # We correlate Abs_Error with available features
    print("\nFailure Analysis (Correlation with Absolute Error):")
    features_to_check = ["Baseline_FVC", "Delta_Week", "Pred_Sigma", "Target_FVC"]

    for feat in features_to_check:
        if feat in results_df.columns:
            corr, _ = pearsonr(results_df["Abs_Error"], results_df[feat])
            print(f"  Error vs {feat}: {corr:.4f}")

    # 6. Conditional Submission
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        predict_and_submit(model, device)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    run()
