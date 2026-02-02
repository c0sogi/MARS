import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders
from library.model import DALANet
from library.train import train_one_epoch, valid_one_epoch, LaplaceLogLikelihoodLoss


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline
    Config.EPOCHS = 15  # Sufficient for small dataset convergence

    # Ensure directories exist
    Config.setup()

    print(f"Running on device: {device}")
    print(f"Epochs: {Config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    model = DALANet()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    criterion = LaplaceLogLikelihoodLoss()

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("Starting training...")
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_runfile.pth")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = valid_one_epoch(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Save Best
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            # print(f"Epoch {epoch+1}: New best score {best_score:.4f}")

        # Optional: Print progress (minimal)
        # print(f"Epoch {epoch+1}/{Config.EPOCHS} - Loss: {train_loss:.4f} - Score: {val_score:.4f}")

    print("Training complete.")

    # -------------------------------------------------------------------------
    # 5. Final Validation and Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing validation analysis...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_results = []

    with torch.no_grad():
        for batch in val_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device).view(-1, 1)

            delta_week = (
                torch.tensor(batch["meta"]["Delta_Week"], dtype=torch.float32)
                .to(device)
                .view(-1, 1)
            )
            baseline_fvc = (
                torch.tensor(batch["meta"]["Baseline_FVC"], dtype=torch.float32)
                .to(device)
                .view(-1, 1)
            )

            # Inference
            fvc_pred, sigma_pred = model(
                axial, coronal, tabular, delta_week, baseline_fvc
            )

            # Collect data for analysis
            batch_size = axial.size(0)
            for i in range(batch_size):
                val_results.append(
                    {
                        "Patient_Week": batch["meta"]["Patient_Week"][i],
                        "FVC_True": target[i].item(),
                        "FVC_Pred": fvc_pred[i].item(),
                        "Sigma": sigma_pred[i].item(),
                        "Delta_Week": delta_week[i].item(),
                        "Baseline_FVC": baseline_fvc[i].item(),
                        # Extract raw tabular features for correlation (approximate reconstruction)
                        "Age_Scaled": batch["tabular"][i, 0].item(),
                        "Percent_Scaled": batch["tabular"][i, 5].item(),
                    }
                )

    val_df = pd.DataFrame(val_results)

    # Calculate Metric on full set
    # Note: score_function takes arrays
    final_metric = score_function(
        val_df["FVC_True"].values, val_df["FVC_Pred"].values, val_df["Sigma"].values
    )

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    val_df["Abs_Error"] = (val_df["FVC_True"] - val_df["FVC_Pred"]).abs()

    features_to_check = {
        "Age": val_df["Age_Scaled"],
        "Percent": val_df["Percent_Scaled"],
        "Baseline_FVC": val_df["Baseline_FVC"],
        "Delta_Week": val_df["Delta_Week"].abs(),  # Magnitude of time difference
    }

    print("Correlation between Absolute Error and Features:")
    for name, series in features_to_check.items():
        if series.std() > 0:
            corr, _ = pearsonr(val_df["Abs_Error"], series)
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: N/A (No variance)")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        submission_rows = []

        with torch.no_grad():
            for batch in test_loader:
                axial = batch["axial"].to(device)
                coronal = batch["coronal"].to(device)
                tabular = batch["tabular"].to(device)

                delta_week = (
                    torch.tensor(batch["meta"]["Delta_Week"], dtype=torch.float32)
                    .to(device)
                    .view(-1, 1)
                )
                baseline_fvc = (
                    torch.tensor(batch["meta"]["Baseline_FVC"], dtype=torch.float32)
                    .to(device)
                    .view(-1, 1)
                )

                # Inference
                fvc_pred, sigma_pred = model(
                    axial, coronal, tabular, delta_week, baseline_fvc
                )

                batch_size = axial.size(0)
                for i in range(batch_size):
                    # Clip confidence as per metric requirement (though metric formula does it, submission usually expects valid values)
                    # The task description says "confidence values are clipped at 70 ml to reflect...".
                    # We output the raw prediction, but ensuring it's not crazy small is good.
                    conf = max(sigma_pred[i].item(), 70.0)

                    submission_rows.append(
                        {
                            "Patient_Week": batch["meta"]["Patient_Week"][i],
                            "FVC": fvc_pred[i].item(),
                            "Confidence": conf,
                        }
                    )

        sub_df = pd.DataFrame(submission_rows)

        # Ensure columns are correct
        sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(sub_df.head())

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
