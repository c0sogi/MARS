import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from provided library
from library.config import Config
from library.train import run_training
from library.model import OSICModel
from library.data import get_scalers, process_patient_image, OSICDataset
from library.utils import seed_everything, laplace_log_likelihood_metric


# -------------------------------------------------------------------------
# Custom Dataset for Test Inference
# -------------------------------------------------------------------------
class TestDataset(Dataset):
    """
    Custom dataset for test inference that correctly maps baseline data
    from test.csv to the requested weeks in sample_submission.csv.
    """

    def __init__(self, test_df, submission_df, scalers):
        self.test_df = test_df.set_index("Patient")
        self.sub_df = submission_df.copy()
        self.scalers = scalers

        # Standard Image Transforms
        self.transform = A.Compose(
            [A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD), ToTensorV2()]
        )

    def __len__(self):
        return len(self.sub_df)

    def __getitem__(self, idx):
        row = self.sub_df.iloc[idx]
        patient_id = row["Patient"]
        target_week = row["Weeks"]

        # Retrieve baseline info from test.csv
        # Note: test.csv contains the baseline measurement for each patient
        base_data = self.test_df.loc[patient_id]
        base_week = base_data["Weeks"]
        base_fvc = base_data["FVC"]
        age = base_data["Age"]
        sex = base_data["Sex"]
        smoke = base_data["SmokingStatus"]

        # --- Feature Engineering & Scaling ---
        s = self.scalers

        # Continuous Features
        age_sc = (age - s["age_mean"]) / s["age_std"]
        base_fvc_sc = (base_fvc - s["base_fvc_mean"]) / s["base_fvc_std"]
        # Relative week: Target Week - Baseline Week
        rel_week_sc = (target_week - base_week - s["rel_weeks_mean"]) / s[
            "rel_weeks_std"
        ]

        # Categorical Features (One-Hot)
        sex_m = 1.0 if sex == "Male" else 0.0
        sex_f = 1.0 if sex == "Female" else 0.0

        smoke_ex = 1.0 if smoke == "Ex-smoker" else 0.0
        smoke_never = 1.0 if smoke == "Never smoked" else 0.0
        smoke_cur = 1.0 if smoke == "Currently smokes" else 0.0

        # Construct Tabular Vector
        tab_vector = np.array(
            [
                age_sc,
                base_fvc_sc,
                rel_week_sc,
                sex_m,
                sex_f,
                smoke_ex,
                smoke_never,
                smoke_cur,
            ],
            dtype=np.float32,
        )

        # --- Image Loading ---
        # split_type='test' ensures we look in the test DICOM directory
        img_numpy = process_patient_image(
            patient_id, split_type="test", load_cached_data=True
        )
        img_tensor = self.transform(image=img_numpy)["image"]

        return {
            "image": img_tensor,
            "tabular": torch.from_numpy(tab_vector),
            "patient_week": row["Patient_Week"],
        }


# -------------------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------------------
def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    device = Config.DEVICE

    print("=== Starting Pipeline ===")

    # 2. Training
    # We use a reduced number of epochs for the fast baseline requirement
    print("\n[Step 1] Running Training...")
    run_training(epochs=15, load_cached_data=True)

    # 3. Load Metadata & Scalers
    print("\n[Step 2] Loading Metadata and Scalers...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    scalers = get_scalers(train_df)

    # 4. Validation Inference
    print("\n[Step 3] Performing Validation...")
    val_dataset = OSICDataset(
        val_df, split_type="val", scalers=scalers, load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Best Model
    model = OSICModel().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Inference Loop
    all_preds_mu = []
    all_preds_sigma = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)

            preds = model(images, tabular)

            # Unscale Predictions
            # Mu: scale * std + mean
            mu_scaled = preds[:, 0]
            mu_ml = mu_scaled * scalers["fvc_std"] + scalers["fvc_mean"]

            # Sigma: softplus(scale) * std (No mean addition for uncertainty)
            sigma_scaled = F.softplus(preds[:, 1])
            sigma_ml = sigma_scaled * scalers["fvc_std"]

            # Unscale Targets
            targets_ml = targets.squeeze() * scalers["fvc_std"] + scalers["fvc_mean"]

            all_preds_mu.append(mu_ml.cpu())
            all_preds_sigma.append(sigma_ml.cpu())
            all_targets.append(targets_ml.cpu())

    # Concatenate results
    y_pred_mu = torch.cat(all_preds_mu)
    y_pred_sigma = torch.cat(all_preds_sigma)
    y_true = torch.cat(all_targets)

    # 5. Compute Metric
    final_metric = laplace_log_likelihood_metric(y_true, y_pred_mu, y_pred_sigma)
    print(f"Final Validation Metric: {final_metric.item()}")

    # 6. Failure Analysis
    print("\n[Step 4] Failure Analysis...")
    # Calculate absolute error
    errors = torch.abs(y_true - y_pred_mu).numpy()

    # We need to align errors with original features for correlation
    # Since val_loader was shuffle=False, we can align with val_df
    analysis_df = val_df.copy()

    # Re-merge baseline info to get Base_FVC and Base_Week for correlation
    # (Logic borrowed from OSICDataset to ensure consistency)
    temp_df = train_df.sort_values(["Patient", "Weeks"])
    baseline_map = (
        temp_df.groupby("Patient")
        .first()[["FVC", "Weeks"]]
        .rename(columns={"FVC": "Base_FVC", "Weeks": "Base_Week"})
    )

    analysis_df = analysis_df.merge(baseline_map, on="Patient", how="left")
    analysis_df["Rel_Weeks"] = analysis_df["Weeks"] - analysis_df["Base_Week"]
    analysis_df["Error"] = errors

    # Compute Correlations
    correlations = analysis_df[
        ["Age", "Base_FVC", "Rel_Weeks", "Percent", "Error"]
    ].corr()["Error"]
    print("Correlation between Error Magnitude and Features:")
    print(correlations.drop("Error").sort_values(ascending=False))

    # 7. Submission Generation
    THRESHOLD = -6.7377055487078
    if final_metric.item() > THRESHOLD:
        print("\n[Step 5] Metric passed threshold. Generating Submission...")

        # Load Test Data
        test_df = pd.read_csv(Config.TEST_CSV)
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

        # Parse Patient and Weeks from sample_submission
        # Format: ID..._Week
        # We split on the last underscore
        sample_sub["Patient"] = sample_sub["Patient_Week"].apply(
            lambda x: "_".join(x.split("_")[:-1])
        )
        sample_sub["Weeks"] = sample_sub["Patient_Week"].apply(
            lambda x: int(x.split("_")[-1])
        )

        # Create Test Dataset
        test_dataset = TestDataset(test_df, sample_sub, scalers)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        submission_rows = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                tabular = batch["tabular"].to(device)
                patient_weeks = batch["patient_week"]

                preds = model(images, tabular)

                # Unscale
                mu_scaled = preds[:, 0]
                sigma_scaled = F.softplus(preds[:, 1])

                mu_ml = mu_scaled * scalers["fvc_std"] + scalers["fvc_mean"]
                sigma_ml = sigma_scaled * scalers["fvc_std"]

                # Clip Confidence as per metric requirement (min 70)
                sigma_ml = torch.clamp(sigma_ml, min=70)

                # Store
                mu_np = mu_ml.cpu().numpy()
                sigma_np = sigma_ml.cpu().numpy()

                for pw, fvc, conf in zip(patient_weeks, mu_np, sigma_np):
                    submission_rows.append(
                        {"Patient_Week": pw, "FVC": fvc, "Confidence": conf}
                    )

        # Save Submission
        sub_df = pd.DataFrame(submission_rows)
        sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]  # Ensure column order
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\n[Step 5] Metric {final_metric.item():.4f} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
