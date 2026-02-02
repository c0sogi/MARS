import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library modules
from library.config import Config
from library.utils import seed_everything, AverageMeter, compute_metric
from library.dataset import get_dataloaders
from library.model import DSPRNet
from library.loss import LaplaceNLLLoss
from library.engine import train_one_epoch, evaluate


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # load_cached_data=True ensures we use the preprocessed artifacts if available
    train_loader, val_loader, test_loader, target_stats = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing DSPRNet model...")
    model = DSPRNet().to(device)

    # 4. Optimization Setup
    # Differential Learning Rates: Lower for backbone, higher for new heads
    backbone_params = list(model.backbone.parameters())

    # Collect parameters for the new heads/streams
    head_params = (
        list(model.img_projector.parameters())
        + list(model.stream_a.parameters())
        + list(model.stream_b.parameters())
        + list(model.head.parameters())
    )

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEADS},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    loss_fn = LaplaceNLLLoss()

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        # Train Step
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, loss_fn, device
        )

        # Validation Step
        val_loss, val_metric = evaluate(
            model, val_loader, loss_fn, device, target_stats
        )

        # Update Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Metric: {val_metric:.6f}"
        )

        # Save Best Model
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best metric! Model saved to {best_model_path}")

    print(f"Training complete.")
    print(f"Final Validation Metric: {best_metric}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis on Validation Set...")
    # Load best model for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # We reconstruct the validation predictions to analyze errors
    val_df = val_loader.dataset.df.copy()

    all_preds_mu = []
    all_targets = []

    with torch.no_grad():
        for images, clinical_features, targets in val_loader:
            images = images.to(device)
            clinical_features = clinical_features.to(device)

            mu, _ = model(images, clinical_features)

            all_preds_mu.append(mu.cpu().numpy())
            all_targets.append(targets.view(-1).numpy())

    all_preds_mu = np.concatenate(all_preds_mu)
    all_targets = np.concatenate(all_targets)

    # Inverse transform to get real FVC values
    global_mean = target_stats["mean"]
    global_std = target_stats["std"]

    pred_fvc_ml = all_preds_mu * global_std + global_mean
    true_fvc_ml = all_targets * global_std + global_mean

    # Calculate Absolute Error
    abs_errors = np.abs(true_fvc_ml - pred_fvc_ml)

    # Verify alignment and compute correlations
    if len(val_df) == len(abs_errors):
        val_df["AbsError"] = abs_errors

        # Select relevant columns for correlation analysis
        # 'Baseline_FVC' is created in prepare_tabular
        cols_of_interest = ["Baseline_FVC", "Age", "Percent", "Weeks", "AbsError"]
        available_cols = [c for c in cols_of_interest if c in val_df.columns]

        if "AbsError" in available_cols:
            corr_matrix = val_df[available_cols].corr()
            print("Correlation between Input Features and Absolute Error:")
            print(corr_matrix["AbsError"].sort_values(ascending=False))
    else:
        print(
            "Warning: Validation dataframe length mismatch. Skipping detailed correlation analysis."
        )

    # 7. Submission Generation
    THRESHOLD = -6.573619738753321
    if best_metric > THRESHOLD:
        print("\nMetric threshold met. Generating submission...")
        generate_submission(model, test_loader, device, target_stats)
    else:
        print(
            f"\nBest metric ({best_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


def generate_submission(model, test_loader, device, target_stats):
    """
    Generates the submission file by predicting FVC for all requested weeks.
    """
    model.eval()

    # 1. Load Sample Submission to get requested Patient_Weeks
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    sample_df = pd.read_csv(sample_sub_path)

    # Parse Patient ID and Week from the ID string (e.g., ID000..._12)
    sample_df["Patient"] = sample_df["Patient_Week"].apply(
        lambda x: x.rsplit("_", 1)[0]
    )
    sample_df["Weeks"] = (
        sample_df["Patient_Week"].apply(lambda x: x.rsplit("_", 1)[1]).astype(int)
    )

    # 2. Map Test Data
    # Create a dictionary mapping PatientID -> (Image Tensor, Static Clinical Features)
    # The test_loader yields one batch per patient (baseline visit) in the order of test_df
    test_meta_df = test_loader.dataset.df
    patient_data_map = {}

    current_idx = 0
    with torch.no_grad():
        for images, clinical_features in test_loader:
            images = images.to(device)
            clinical_features = clinical_features.to(device)

            batch_size = images.size(0)
            for i in range(batch_size):
                pid = test_meta_df.iloc[current_idx + i]["Patient"]
                patient_data_map[pid] = {
                    "image": images[i],  # (3, 260, 260)
                    "clinical": clinical_features[
                        i
                    ],  # (5,) [BaseFVC, RelTime, Age, Sex, Smoke]
                }
            current_idx += batch_size

    # 3. Generate Predictions
    predictions = {}  # Key: Patient_Week, Value: (FVC, Confidence)
    unique_patients = sample_df["Patient"].unique()

    global_mean = target_stats["mean"]
    global_std = target_stats["std"]

    print(f"Predicting for {len(unique_patients)} unique patients in test set...")

    with torch.no_grad():
        for pid in unique_patients:
            if pid not in patient_data_map:
                continue

            # Retrieve static data for this patient
            p_data = patient_data_map[pid]
            img_tensor = p_data["image"]
            static_clinical = p_data["clinical"]

            # Get baseline week for relative time calculation
            baseline_week = test_meta_df[test_meta_df["Patient"] == pid][
                "Baseline_Week"
            ].values[0]

            # Identify all target weeks for this patient
            target_weeks = sample_df[sample_df["Patient"] == pid]["Weeks"].values
            n_weeks = len(target_weeks)

            if n_weeks == 0:
                continue

            # Create Batch
            # Repeat image: (N, 3, 260, 260)
            batch_imgs = img_tensor.unsqueeze(0).repeat(n_weeks, 1, 1, 1)

            # Repeat clinical features: (N, 5)
            batch_clinical = static_clinical.unsqueeze(0).repeat(n_weeks, 1)

            # Update Relative Time feature (Index 1)
            # Formula: (Target_Week - Baseline_Week) * 0.01
            rel_times = (target_weeks - baseline_week) * 0.01
            batch_clinical[:, 1] = torch.tensor(rel_times, dtype=torch.float32).to(
                device
            )

            # Forward Pass
            mu, sigma = model(batch_imgs, batch_clinical)

            # Inverse Transform
            mu_np = mu.cpu().numpy()
            sigma_np = sigma.cpu().numpy()

            pred_fvc = mu_np * global_std + global_mean
            pred_sigma = sigma_np * global_std

            # Apply Submission Clipping for Confidence
            pred_sigma = np.maximum(pred_sigma, Config.SIGMA_MIN_CLIP)

            # Store results
            for w, fvc, conf in zip(target_weeks, pred_fvc, pred_sigma):
                key = f"{pid}_{w}"
                predictions[key] = (fvc, conf)

    # 4. Construct Final DataFrame
    submission_rows = []
    for idx, row in sample_df.iterrows():
        pw = row["Patient_Week"]
        if pw in predictions:
            fvc, conf = predictions[pw]
        else:
            # Fallback (should not happen for valid patients)
            fvc = 2000
            conf = 100
        submission_rows.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    submission_df = pd.DataFrame(submission_rows)

    # Save
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


if __name__ == "__main__":
    main()
