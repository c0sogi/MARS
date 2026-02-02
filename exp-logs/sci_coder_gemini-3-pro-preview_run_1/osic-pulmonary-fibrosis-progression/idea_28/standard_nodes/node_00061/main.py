import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.utils import seed_everything, LaplaceLogLikelihood
from library.dataset import FVCDataset, get_transforms
from library.model import VERNet
from library.engine import run_training, generate_submission_file


def main():
    # --- Configuration ---
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    PATIENCE = 5
    NUM_WORKERS = 2

    # Paths
    WORK_DIR = "./working"
    SAVE_PATH = os.path.join(WORK_DIR, "best_model_runfile.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Threshold for submission generation
    METRIC_THRESHOLD = -6.510164260864258

    # Ensure reproducibility
    seed_everything(SEED)

    print(f"Running on device: {DEVICE}")

    # --- 1. Training Phase ---
    print("\n" + "=" * 30)
    print("STARTING TRAINING")
    print("=" * 30)

    # Execute training using the provided engine
    # This handles the training loop, validation monitoring, and checkpointing
    best_val_metric_from_train = run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        save_path=SAVE_PATH,
        patience=PATIENCE,
        num_workers=NUM_WORKERS,
    )

    # --- 2. Validation & Failure Analysis Phase ---
    print("\n" + "=" * 30)
    print("VALIDATION & FAILURE ANALYSIS")
    print("=" * 30)

    # Load Validation Dataset
    val_dataset = FVCDataset(
        mode="val", transform=get_transforms("val"), load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Load the best model
    model = VERNet().to(DEVICE)
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        print(f"Loaded best model from {SAVE_PATH}")
    else:
        print("Error: Model checkpoint not found!")
        return

    model.eval()

    # Metric criterion (reduction='none' to get per-sample metric)
    criterion = LaplaceLogLikelihood(reduction="none")

    results = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_ax = batch["img_axial"].to(DEVICE)
            img_cor = batch["img_coronal"].to(DEVICE)
            tabular = batch["tabular"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            week = batch["week"].to(DEVICE)
            base_fvc = batch["baseline_fvc"].to(DEVICE)

            # Forward pass
            pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, week, base_fvc)

            # Calculate Loss per sample
            loss = criterion(pred_fvc, pred_sigma, target)

            # Metric is negative loss
            metric = -loss

            # Calculate Absolute Error
            abs_err = torch.abs(target - pred_fvc)

            # Move to CPU for analysis
            pred_fvc_np = pred_fvc.cpu().numpy()
            pred_sigma_np = pred_sigma.cpu().numpy()
            target_np = target.cpu().numpy()
            metric_np = metric.cpu().numpy()
            abs_err_np = abs_err.cpu().numpy()
            week_np = week.cpu().numpy()

            # Extract tabular features for correlation analysis
            # Tabular: [Age_Norm, Percent_Norm, Sex, Smoke_0, Smoke_1, Smoke_2]
            tabular_np = tabular.cpu().numpy()

            for i in range(len(pred_fvc_np)):
                results.append(
                    {
                        "Target_FVC": target_np[i],
                        "Pred_FVC": pred_fvc_np[i],
                        "Pred_Sigma": pred_sigma_np[i],
                        "Metric": metric_np[i],
                        "Abs_Error": abs_err_np[i],
                        "Week": week_np[i],
                        "Age_Norm": tabular_np[i, 0],
                        "Percent_Norm": tabular_np[i, 1],
                    }
                )

    # Create DataFrame for analysis
    df_results = pd.DataFrame(results)

    # Calculate Final Metric (Average over all validation samples)
    final_metric = df_results["Metric"].mean()

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    print("\n--- Failure Analysis: Correlation with Absolute Error ---")
    features_to_check = ["Age_Norm", "Percent_Norm", "Week", "Target_FVC"]
    for feat in features_to_check:
        corr = df_results[feat].corr(df_results["Abs_Error"])
        print(f"Correlation ({feat} vs Abs_Error): {corr:.6f}")

    # --- 3. Submission Phase ---
    print("\n" + "=" * 30)
    print("SUBMISSION GENERATION")
    print("=" * 30)

    if final_metric > METRIC_THRESHOLD:
        print(
            f"Validation Metric ({final_metric}) exceeds threshold ({METRIC_THRESHOLD})."
        )
        print("Generating submission file...")

        generate_submission_file(
            model_path=SAVE_PATH,
            output_path=SUBMISSION_PATH,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    else:
        print(
            f"Validation Metric ({final_metric}) is below threshold ({METRIC_THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
