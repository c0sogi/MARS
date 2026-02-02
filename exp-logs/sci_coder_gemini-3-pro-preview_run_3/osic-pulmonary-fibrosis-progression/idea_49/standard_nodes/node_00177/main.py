import os
import sys
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders
from library.train import Trainer
from library.model import SCARNet


def main():
    # --- 1. Configuration & Setup ---
    # Adjust epochs for a fast baseline execution as requested
    Config.EPOCHS = 25
    Config.T_MAX = 25

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print("Initializing SCAR-Net Pipeline...")
    Config.print_config()

    # --- 2. Training ---
    print("\n--- Phase 1: Training ---")
    train_loader, val_loader, test_loader = get_dataloaders()

    trainer = Trainer(train_loader, val_loader)
    trainer.fit()

    # --- 3. Validation & Failure Analysis ---
    print("\n--- Phase 2: Validation & Failure Analysis ---")

    # Load the best model
    device = Config.DEVICE
    model = SCARNet()
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found. Training may have failed.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Containers for analysis
    val_preds_mu = []
    val_preds_sigma = []
    val_targets = []
    val_meta_rows = []

    # Inference on Validation Set
    with torch.no_grad():
        for images, tabular, targets in val_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            # Forward pass
            preds = model(images, tabular)

            # Extract scaled predictions
            mu_scaled = preds[:, 0].cpu().numpy()
            sigma_scaled = preds[:, 1].cpu().numpy()

            # Inverse Transform to original scale (ml)
            mu_final = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_final = sigma_scaled * Config.TARGET_STD

            # Inverse Transform targets
            t_scaled = targets.cpu().numpy().flatten()
            t_final = t_scaled * Config.TARGET_STD + Config.TARGET_MEAN

            # Store
            val_preds_mu.extend(mu_final)
            val_preds_sigma.extend(sigma_final)
            val_targets.extend(t_final)
            val_meta_rows.append(tabular.cpu().numpy())

    # Convert to arrays
    val_preds_mu = np.array(val_preds_mu)
    val_preds_sigma = np.array(val_preds_sigma)
    val_targets = np.array(val_targets)
    val_meta = np.vstack(val_meta_rows)

    # Compute Final Metric
    final_metric = score_function(val_targets, val_preds_mu, val_preds_sigma)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error Magnitude and Features
    errors = np.abs(val_targets - val_preds_mu)

    # Feature indices in 'tabular':
    # 0: Baseline_FVC_Scaled, 1: Time_Scaled, 2: Age_Scaled, 3: Sex_Code, 4: Smoking_Code
    feature_map = {
        0: "Baseline_FVC",
        1: "Relative_Time",
        2: "Age",
        3: "Sex",
        4: "SmokingStatus",
    }

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    for idx, name in feature_map.items():
        feat_values = val_meta[:, idx]
        # Handle constant features (std=0) to avoid NaN correlation
        if np.std(feat_values) > 1e-9:
            corr = np.corrcoef(errors, feat_values)[0, 1]
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: N/A (Constant)")

    # --- 4. Submission ---
    print("\n--- Phase 3: Submission Generation ---")
    threshold = -6.573619738753321

    if final_metric > threshold:
        print(
            f"Metric ({final_metric:.6f}) > Threshold ({threshold:.6f}). Generating submission..."
        )

        if test_loader is None:
            print(
                "Warning: Test loader is None (sample_submission.csv missing?). Skipping submission."
            )
            return

        sub_patient_weeks = []
        sub_fvc = []
        sub_conf = []

        with torch.no_grad():
            for images, tabular, patient_ids, weeks in test_loader:
                images = images.to(device)
                tabular = tabular.to(device)

                preds = model(images, tabular)

                # Inverse Transform
                mu_scaled = preds[:, 0].cpu().numpy()
                sigma_scaled = preds[:, 1].cpu().numpy()

                mu_final = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN
                sigma_final = sigma_scaled * Config.TARGET_STD

                # Apply Confidence Floor (70ml)
                sigma_final = np.maximum(sigma_final, 70)

                # Collect results
                for i in range(len(patient_ids)):
                    pw = f"{patient_ids[i]}_{weeks[i].item()}"
                    sub_patient_weeks.append(pw)
                    sub_fvc.append(mu_final[i])
                    sub_conf.append(sigma_final[i])

        # Create DataFrame
        submission_df = pd.DataFrame(
            {"Patient_Week": sub_patient_weeks, "FVC": sub_fvc, "Confidence": sub_conf}
        )

        # Ensure correct path
        save_path = Config.SUBMISSION_FILE
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        print(f"Submission shape: {submission_df.shape}")

    else:
        print(
            f"Metric ({final_metric:.6f}) <= Threshold ({threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
