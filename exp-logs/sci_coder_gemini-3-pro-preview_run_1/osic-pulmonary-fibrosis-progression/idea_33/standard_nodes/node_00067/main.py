import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
import library.data
import importlib

importlib.reload(library.data)
from library.data import LungDataset, get_transforms
from library.model import GTVRNet
from library.train import train_epoch, valid_epoch, LaplaceLikelihoodLoss


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline
    # Reducing epochs to ensure completion within strict time limits
    EPOCHS = 15
    BATCH_SIZE = Config.BATCH_SIZE

    # Create submission directory
    os.makedirs("./submission", exist_ok=True)
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)

    print(f"Running on device: {device}")

    # 2. Data Preparation
    print("Initializing Datasets...")
    train_dataset = LungDataset(
        csv_path=Config.TRAIN_CSV,
        mode="train",
        transform=get_transforms("train"),
        load_cached_data=True,
    )

    val_dataset = LungDataset(
        csv_path=Config.VAL_CSV,
        mode="val",
        transform=get_transforms("val"),
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = GTVRNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLikelihoodLoss()

    # 4. Training Loop
    print("Starting Training...")
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.MODEL_SAVE_DIR, "best_model.pth")

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metric = valid_epoch(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Metric: {val_metric:.4f}"
        )

        # Save Best
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)

    print(f"Training finished. Best Metric: {best_metric:.4f}")

    # 5. Validation & Failure Analysis
    print("Performing Validation and Failure Analysis...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_preds_fvc = []
    val_preds_sigma = []
    val_targets = []

    # Store features for failure analysis
    # We need to extract them from the dataset/loader again or accumulate during inference
    # Accumulating during inference loop:
    meta_features = {
        "Baseline_FVC": [],
        "Delta_Week": [],
        "Age_norm": [],
        "BasePercent_norm": [],
    }

    with torch.no_grad():
        for batch in val_loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)

            # Forward
            fvc_pred, sigma_pred = model(img_axial, img_coronal, tabular, meta)

            # Collect results
            val_preds_fvc.extend(fvc_pred.cpu().numpy().flatten())
            val_preds_sigma.extend(sigma_pred.cpu().numpy().flatten())
            val_targets.extend(target.cpu().numpy().flatten())

            # Collect features (meta: [Baseline_FVC, Delta_Week], tabular: [Age, Sex, Smoke..., Percent])
            # Tabular index 0 is Age_norm, index 5 is BasePercent_norm
            meta_features["Baseline_FVC"].extend(meta[:, 0].cpu().numpy().flatten())
            meta_features["Delta_Week"].extend(meta[:, 1].cpu().numpy().flatten())
            meta_features["Age_norm"].extend(tabular[:, 0].cpu().numpy().flatten())
            meta_features["BasePercent_norm"].extend(
                tabular[:, 5].cpu().numpy().flatten()
            )

    # Convert to numpy
    y_true = np.array(val_targets)
    y_pred = np.array(val_preds_fvc)
    sigma_pred = np.array(val_preds_sigma)

    # Compute Final Metric
    final_metric = laplace_log_likelihood_metric(y_true, y_pred, sigma_pred)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Error Magnitude with Features
    error_magnitude = np.abs(y_true - y_pred)

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    analysis_df = pd.DataFrame(
        {
            "Error": error_magnitude,
            "Baseline_FVC": meta_features["Baseline_FVC"],
            "Delta_Week": meta_features["Delta_Week"],
            "Age_norm": meta_features["Age_norm"],
            "BasePercent_norm": meta_features["BasePercent_norm"],
        }
    )

    correlations = analysis_df.corr()["Error"].drop("Error")
    print(correlations)

    # 6. Submission
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        test_dataset = LungDataset(
            csv_path=Config.TEST_CSV,
            mode="test",
            transform=get_transforms("val"),  # No augmentation for test
            load_cached_data=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_rows = []

        with torch.no_grad():
            for batch in test_loader:
                img_axial = batch["img_axial"].to(device)
                img_coronal = batch["img_coronal"].to(device)
                tabular = batch["tabular"].to(device)
                meta = batch["meta"].to(device)
                patient_weeks = batch["patient_week"]  # List of strings

                # Forward
                fvc_pred, sigma_pred = model(img_axial, img_coronal, tabular, meta)

                fvc_vals = fvc_pred.cpu().numpy().flatten()
                sigma_vals = sigma_pred.cpu().numpy().flatten()

                for pw, fvc, conf in zip(patient_weeks, fvc_vals, sigma_vals):
                    # Clip confidence as per metric requirement (min 70)
                    conf_clipped = max(conf, 70.0)
                    submission_rows.append(
                        {"Patient_Week": pw, "FVC": fvc, "Confidence": conf_clipped}
                    )

        # Create DataFrame
        sub_df = pd.DataFrame(submission_rows)

        # Ensure correct column order
        sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

        # Save
        save_path = "./submission/submission.csv"
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
