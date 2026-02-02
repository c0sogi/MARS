import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_loss
from library.data import get_dataloaders
from library.model import SLHDANetwork
from library.train import train_one_epoch, validate
from library.inference import predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to find correlations
    between error magnitude and input features.
    """
    model.eval()
    data_records = []

    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tab = batch["tabular"].to(device)
            target = batch["target"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)

            # Forward pass
            pred_fvc, pred_sigma = model(
                img_ax, img_cor, tab, weeks, base_fvc, base_week
            )

            # Move to CPU for analysis
            pred_fvc = pred_fvc.cpu().numpy()
            target = target.cpu().numpy()
            weeks_np = weeks.cpu().numpy()

            # Extract tabular features for correlation
            # Tabular vector structure from library/data.py:
            # [Norm_Age, Enc_Sex, Smoke_Ex, Smoke_Never, Smoke_Current, Norm_Percent]
            tab_np = tab.cpu().numpy()

            for i in range(len(target)):
                # Calculate absolute error
                abs_error = np.abs(target[i] - pred_fvc[i])

                # De-normalize features roughly for readability if needed,
                # but correlation works fine on normalized data.
                # We use the normalized values directly.
                record = {
                    "AbsError": abs_error,
                    "Weeks": weeks_np[i],
                    "Norm_Age": tab_np[i, 0],
                    "Norm_Percent": tab_np[i, 5],
                    "Sex_Enc": tab_np[i, 1],
                }
                data_records.append(record)

    df_analysis = pd.DataFrame(data_records)

    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    if len(df_analysis) > 0:
        # Calculate correlations
        correlations = df_analysis.corr()["AbsError"].sort_values(ascending=False)
        print("Correlation between Absolute Error and features:")
        print(correlations.drop("AbsError"))
    else:
        print("No validation data available for analysis.")


def main():
    # 1. Setup
    Config.setup()

    # Override Config for Fast Baseline
    # We reduce epochs to ensure execution within time limits while keeping full data
    Config.EPOCHS = 20

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # 2. Data Loading
    # We use debug=False to ensure we get a valid metric on the full validation set
    # The dataset is small enough that this is still fast.
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 3. Model Initialization
    model = SLHDANetwork().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_MIN_LR
    )

    # 5. Training Loop
    best_metric = -float("inf")

    print(f"\nStarting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metric = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Save Best Model
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

        # Logging (minimal)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Metric: {val_metric:.6f}"
            )

    # 6. Final Validation & Evaluation
    print("\nLoading best model for evaluation...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Compute Final Metric on full validation set
    final_metric = validate(model, val_loader, device)

    # STRICT OUTPUT FORMAT REQUIRED
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Conditional Submission
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions for test set
        model.eval()
        results = []

        with torch.no_grad():
            for batch in test_loader:
                img_ax = batch["img_axial"].to(device)
                img_cor = batch["img_coronal"].to(device)
                tab = batch["tabular"].to(device)
                weeks = batch["weeks"].to(device)
                base_fvc = batch["base_fvc"].to(device)
                base_week = batch["base_week"].to(device)
                patient_ids = batch["patient_id"]

                pred_fvc, pred_sigma = model(
                    img_ax, img_cor, tab, weeks, base_fvc, base_week
                )

                pred_fvc_np = pred_fvc.cpu().numpy()
                pred_sigma_np = pred_sigma.cpu().numpy()
                weeks_np = weeks.cpu().numpy()

                for i in range(len(patient_ids)):
                    pid = patient_ids[i]
                    wk = int(weeks_np[i])
                    results.append(
                        {
                            "Patient_Week": f"{pid}_{wk}",
                            "FVC": pred_fvc_np[i],
                            "Confidence": pred_sigma_np[i],
                        }
                    )

        # Save submission
        df_sub = pd.DataFrame(results)
        df_sub = df_sub[["Patient_Week", "FVC", "Confidence"]]
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
