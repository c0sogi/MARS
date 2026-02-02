import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings

# Import from the provided library
from library.config import (
    Paths,
    Training,
    System,
    setup_directories,
    Model as ModelConfig,
)
from library.utils import seed_everything, laplace_log_likelihood
from library.dataset import LungDataset
from library.model import PyramidDualAxisNet
from library.loss import LaplaceLogLikelihoodLoss
from library.engine import train_one_epoch, evaluate

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    setup_directories()
    seed_everything(Training.SEED)
    device = torch.device(System.DEVICE)

    # 2. Data Loading
    try:
        train_df = pd.read_csv(Paths.TRAIN_CSV)
        val_df = pd.read_csv(Paths.VAL_CSV)
        test_df = pd.read_csv(Paths.TEST_CSV)
    except FileNotFoundError as e:
        print(f"Error loading metadata: {e}")
        return

    # Initialize Datasets
    # cache_images=True ensures we use the preprocessed .npy files if available
    train_dataset = LungDataset(train_df, mode="train", cache_images=True)
    val_dataset = LungDataset(val_df, mode="val", cache_images=True)
    test_dataset = LungDataset(test_df, mode="test", cache_images=True)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Training.BATCH_SIZE,
        shuffle=True,
        num_workers=System.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Training.BATCH_SIZE,
        shuffle=False,
        num_workers=System.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Training.BATCH_SIZE,
        shuffle=False,
        num_workers=System.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Optimization
    model = PyramidDualAxisNet()
    model.to(device)

    criterion = LaplaceLogLikelihoodLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Training.LEARNING_RATE,
        weight_decay=Training.WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Training.EPOCHS, eta_min=Training.ETA_MIN
    )

    # 4. Training Loop
    best_metric = -float("inf")
    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Training.EPOCHS} epochs...")

    for epoch in range(Training.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_metric = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Checkpointing (Save best model based on Metric)
        # Note: The metric is negative (higher is better)
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Paths.MODEL_SAVE_PATH)
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Training.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 5. Final Evaluation & Failure Analysis
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Paths.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Compute final metric on full validation set
    _, final_val_metric = evaluate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # --- Failure Analysis ---
    print("\nRunning Failure Analysis on Validation Set...")
    val_errors = []
    val_features = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            axial = data["axial"].to(device)
            coronal = data["coronal"].to(device)
            tabular = data["tabular"].to(device)
            time_delta = data["time_delta"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)
            target = data["target"].to(device)

            # Predict
            pred_fvc, pred_sigma = model(
                axial, coronal, tabular, time_delta, baseline_fvc
            )

            # Calculate Absolute Error
            abs_error = torch.abs(target - pred_fvc).cpu().numpy().flatten()

            # Collect features for correlation analysis
            # Tabular features in dataset are normalized, so we use the raw dataframe values
            # corresponding to this batch would be complex to map back directly via loader order.
            # Instead, we can approximate using the tensors or just rely on the error distribution.
            # To be precise, let's extract the normalized tabular features from the tensor.
            # tabular tensor: [age_norm, sex, smoke, percent_norm]
            feats = tabular.cpu().numpy()
            deltas = time_delta.cpu().numpy()

            batch_errors = pd.DataFrame(
                {
                    "Error": abs_error,
                    "Norm_Age": feats[:, 0],
                    "Sex": feats[:, 1],
                    "Smoking": feats[:, 2],
                    "Norm_Percent": feats[:, 3],
                    "Time_Delta": deltas.flatten(),
                }
            )
            val_errors.append(batch_errors)

    if val_errors:
        analysis_df = pd.concat(val_errors, ignore_index=True)
        correlations = analysis_df.corr()["Error"].sort_values(ascending=False)
        print("Correlation of Absolute Error with Input Features:")
        print(correlations.drop("Error"))

    # 6. Submission Generation
    # Threshold check
    THRESHOLD = -6.510164260864258

    if final_val_metric > THRESHOLD:
        print(
            f"\nMetric ({final_val_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        submission_rows = []

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                axial = data["axial"].to(device)
                coronal = data["coronal"].to(device)
                tabular = data["tabular"].to(device)
                time_delta = data["time_delta"].to(device)
                baseline_fvc = data["baseline_fvc"].to(device)
                patient_weeks = data["patient_week"]  # List of strings

                # Predict
                pred_fvc, pred_sigma = model(
                    axial, coronal, tabular, time_delta, baseline_fvc
                )

                # Move to CPU
                pred_fvc_np = pred_fvc.cpu().numpy().flatten()
                pred_sigma_np = pred_sigma.cpu().numpy().flatten()

                # Clip confidence as per metric requirement (min 70)
                # Although the loss function does this, the submission file needs explicit values.
                # The metric definition says sigma_clipped = max(sigma, 70).
                pred_sigma_np = np.maximum(pred_sigma_np, 70)

                for pw, fvc, conf in zip(patient_weeks, pred_fvc_np, pred_sigma_np):
                    submission_rows.append(
                        {"Patient_Week": pw, "FVC": fvc, "Confidence": conf}
                    )

        submission_df = pd.DataFrame(submission_rows)

        # Ensure correct column order
        submission_df = submission_df[["Patient_Week", "FVC", "Confidence"]]

        # Save
        submission_df.to_csv(Paths.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Paths.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_val_metric}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
