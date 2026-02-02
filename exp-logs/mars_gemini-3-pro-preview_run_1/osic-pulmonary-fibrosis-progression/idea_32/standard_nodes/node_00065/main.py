import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, score
from library.data import get_dataloaders, get_test_dataloader
from library.model import MPVERNet
from library.train import Trainer


def main():
    # 1. Setup and Configuration
    Config.setup()

    # Override Config for Fast Baseline Execution
    Config.EPOCHS = 20  # Reduced epochs for quick turnaround

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader = get_dataloaders()

    # 3. Model Initialization
    print("Initializing Model...")
    model = MPVERNet()
    model.to(device)

    # 4. Training
    print("Starting Training...")
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    # 5. Validation Assessment
    print("\nPerforming Final Validation Assessment...")
    # Load best model weights
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    model.eval()

    val_preds = []
    val_sigmas = []
    val_targets = []
    val_patient_weeks = []

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab_norm = batch["tabular_norm"].to(device)
            tab_raw = batch["tabular_raw"].to(device)
            time_delta = batch["time_delta"].to(device)
            target = batch["target"].to(device)

            # Inference
            pred_fvc, pred_sigma = model(img_ax, img_cor, tab_norm, tab_raw, time_delta)

            # Collect results
            val_preds.append(pred_fvc.cpu().numpy())
            val_sigmas.append(pred_sigma.cpu().numpy())
            val_targets.append(target.cpu().numpy())
            val_patient_weeks.extend(batch["patient_week"])

    # Concatenate results
    val_preds = np.concatenate(val_preds, axis=0).flatten()
    val_sigmas = np.concatenate(val_sigmas, axis=0).flatten()
    val_targets = np.concatenate(val_targets, axis=0).flatten()

    # Compute Metric
    final_metric = score(val_targets, val_preds, val_sigmas)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_preds)

    # Get validation DataFrame to access features
    # Note: val_loader is not shuffled (shuffle=False), so order matches
    val_df = val_loader.dataset.df.copy()

    # Ensure lengths match (just in case of drop_last, though val loader shouldn't drop last)
    if len(val_df) != len(errors):
        # Fallback: align by index if necessary, but typically len matches
        val_df = val_df.iloc[: len(errors)]

    val_df["Error_Magnitude"] = errors

    # Features to correlate with Error
    analysis_features = ["Age", "Percent", "Weeks", "Baseline_FVC", "Baseline_Percent"]

    print("Correlation between Error Magnitude and Input Features:")
    for feat in analysis_features:
        if feat in val_df.columns:
            corr = val_df[feat].corr(val_df["Error_Magnitude"])
            print(f"  {feat}: {corr:.4f}")

    # 7. Submission Generation
    THRESHOLD_METRIC = -6.510164260864258

    if final_metric > THRESHOLD_METRIC:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD_METRIC}). Generating submission..."
        )

        test_loader = get_test_dataloader()

        test_ids = []
        test_fvcs = []
        test_confs = []

        with torch.no_grad():
            for batch in test_loader:
                img_ax = batch["image_axial"].to(device)
                img_cor = batch["image_coronal"].to(device)
                tab_norm = batch["tabular_norm"].to(device)
                tab_raw = batch["tabular_raw"].to(device)
                time_delta = batch["time_delta"].to(device)

                # Inference
                p_fvc, p_sigma = model(img_ax, img_cor, tab_norm, tab_raw, time_delta)

                test_ids.extend(batch["patient_week"])
                test_fvcs.extend(p_fvc.cpu().numpy().flatten())
                test_confs.extend(p_sigma.cpu().numpy().flatten())

        # Create Submission DataFrame
        sub_df = pd.DataFrame(
            {"Patient_Week": test_ids, "FVC": test_fvcs, "Confidence": test_confs}
        )

        # Save
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    main()
