import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import NBCSLN
from library.utils import LaplaceLogLikelihoodLoss, calculate_metric
from library.engine import train_fn, eval_fn, inference_fn


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for a fast baseline execution
    Config.EPOCHS = 15

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Execution Device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading data...")
    # Use cached data to speed up loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("Initializing NBCSLN model...")
    model = NBCSLN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    loss_fn = LaplaceLogLikelihoodLoss()

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_runfile.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train Step
        train_loss = train_fn(train_loader, model, optimizer, device, loss_fn)

        # Validation Step
        val_metric = eval_fn(val_loader, model, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Metric: {val_metric:.4f}"
        )

        # Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best metric! Model saved.")

    # --------------------------------------------------------------------------
    # 5. Final Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("FINAL EVALUATION & FAILURE ANALYSIS")
    print("=" * 40)

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_results = []
    all_targets = []
    all_preds_fvc = []
    all_preds_sigma = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device).squeeze(-1)

            # Forward pass
            outputs = model(img_axial, img_coronal, tabular)
            alpha = outputs[:, 0]
            sigma_base = outputs[:, 1]
            sigma_growth = outputs[:, 2]

            # Reconstruct predictions
            baseline_fvc = meta[:, 0]
            week_diff = meta[:, 1]

            fvc_pred = baseline_fvc + alpha * week_diff
            sigma_pred = sigma_base + sigma_growth * torch.abs(week_diff)

            # Store for global metric calculation
            all_targets.append(target.cpu().numpy())
            all_preds_fvc.append(fvc_pred.cpu().numpy())
            all_preds_sigma.append(sigma_pred.cpu().numpy())

            # Store for failure analysis
            # Tabular structure: [Age_Norm, Sex_M, Sex_F, Smoke_Ex, Smoke_Never, Smoke_Curr, Percent_Norm]
            tabular_cpu = tabular.cpu().numpy()
            targets_cpu = target.cpu().numpy()
            preds_cpu = fvc_pred.cpu().numpy()
            week_diff_cpu = week_diff.cpu().numpy()

            for i in range(len(target)):
                error = abs(targets_cpu[i] - preds_cpu[i])
                val_results.append(
                    {
                        "Error": error,
                        "Age_Norm": tabular_cpu[i, 0],
                        "Percent_Norm": tabular_cpu[i, 6],
                        "Week_Diff": week_diff_cpu[i],
                        "Target_FVC": targets_cpu[i],
                    }
                )

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds_fvc = np.concatenate(all_preds_fvc)
    all_preds_sigma = np.concatenate(all_preds_sigma)

    # Compute and print Final Metric
    final_metric = calculate_metric(all_preds_fvc, all_preds_sigma, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    print("\nFailure Analysis (Correlation with Absolute Error):")
    df_results = pd.DataFrame(val_results)
    features_to_analyze = ["Age_Norm", "Percent_Norm", "Week_Diff", "Target_FVC"]

    for feat in features_to_analyze:
        if feat in df_results.columns:
            # Drop NaNs if any (shouldn't be, but safe practice)
            clean_df = df_results.dropna(subset=["Error", feat])
            if len(clean_df) > 1:
                corr, _ = pearsonr(clean_df["Error"], clean_df[feat])
                print(f"  Correlation Error vs {feat}: {corr:.4f}")
            else:
                print(f"  Correlation Error vs {feat}: N/A (Insufficient data)")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Use provided inference engine
        sub_df = inference_fn(test_loader, model, device)

        # Ensure correct format
        sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
