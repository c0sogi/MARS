import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.utils import seed_everything, load_checkpoint, laplace_log_likelihood
from library.data import LungDataset, get_transforms
from library.model import DCSLNet
from library.train import train_model


def run():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Set a limited number of epochs for a fast baseline execution
    # 15 epochs is sufficient for the model to converge on this small dataset
    # while keeping runtime very short (< 30 mins).
    FAST_RUN_EPOCHS = 15

    print(f"Starting execution with {FAST_RUN_EPOCHS} epochs...")

    # 2. Train the Model
    # We use debug=False to train on the full training set (which is small, ~1100 samples)
    # but limit the epochs to ensure quick execution.
    train_model(debug=False, epochs=FAST_RUN_EPOCHS)

    # 3. Validation Assessment
    print("Loading best model for validation...")
    device = Config.DEVICE
    model = DCSLNet().to(device)

    # Load the best checkpoint saved during training
    load_checkpoint(model, filename="best_model.pth", device=device)
    model.eval()

    # Prepare Validation Loader
    val_dataset = LungDataset(
        Config.VAL_CSV, mode="val", transform=get_transforms("val")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run Inference on Validation Set
    all_true = []
    all_pred = []
    all_sigma = []

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tab = batch["tabular"].to(device)

            meta_dt = batch["meta_dt"].to(device)
            meta_base = batch["meta_base_fvc"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            preds = model(img_ax, img_cor, tab)

            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            # Reconstruct Trajectory
            fvc_pred = meta_base + alpha * meta_dt
            sigma_pred = sigma_base + sigma_growth * torch.abs(meta_dt)

            # Collect results
            all_true.extend(target.cpu().numpy())
            all_pred.extend(fvc_pred.cpu().numpy())
            all_sigma.extend(sigma_pred.cpu().numpy())

    # Convert to numpy arrays
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_sigma = np.array(all_sigma)

    # Compute Metric
    final_metric = laplace_log_likelihood(all_true, all_pred, all_sigma)

    # REQUIRED OUTPUT: Print Final Validation Metric with full precision
    print(f"Final Validation Metric: {final_metric:.16f}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    val_df = val_dataset.df.copy()

    # Calculate Absolute Error
    val_df["pred_fvc"] = all_pred
    val_df["abs_error"] = np.abs(val_df["FVC"] - val_df["pred_fvc"])

    # Prepare features for correlation analysis
    # Convert categorical features to codes
    val_df["Sex_Code"] = val_df["Sex"].astype("category").cat.codes
    val_df["Smoking_Code"] = val_df["SmokingStatus"].astype("category").cat.codes

    features_to_analyze = ["Weeks", "Percent", "Age", "Sex_Code", "Smoking_Code"]

    # Calculate correlation
    correlations = val_df[features_to_analyze].corrwith(val_df["abs_error"])

    print("Correlation between Absolute Error and Input Features:")
    print(correlations)
    print("========================\n")

    # 5. Submission Generation
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric:.4f}) meets threshold ({THRESHOLD:.4f}). Generating submission..."
        )
        generate_submission(model, device)
    else:
        print(
            f"Metric ({final_metric:.4f}) does not meet threshold ({THRESHOLD:.4f}). Submission skipped."
        )


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves submission.csv
    """
    # Prepare Test Loader
    test_dataset = LungDataset(
        Config.TEST_CSV, mode="test", transform=get_transforms("val")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    results_fvc = []
    results_sigma = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tab = batch["tabular"].to(device)

            meta_dt = batch["meta_dt"].to(device)
            meta_base = batch["meta_base_fvc"].to(device)

            # Forward pass
            preds = model(img_ax, img_cor, tab)

            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            # Reconstruct Trajectory
            fvc_pred = meta_base + alpha * meta_dt
            sigma_pred = sigma_base + sigma_growth * torch.abs(meta_dt)

            results_fvc.extend(fvc_pred.cpu().numpy())
            results_sigma.extend(sigma_pred.cpu().numpy())

    # Load test metadata to ensure correct ID mapping
    test_df = pd.read_csv(Config.TEST_CSV)

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": results_fvc,
            "Confidence": results_sigma,
        }
    )

    # Save to file
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    run()
