import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Import library modules
from library.config import Config, seed_everything
from library.data import get_dataloaders, TabularPreprocessor, LungDataset
from library.model import SPPDSNet
from library.train import train_model
from library.utils import save_submission, calculate_metric


def main():
    # 1. Setup and Training
    seed_everything(Config.SEED)
    print(f"Running SPP-DS Net Pipeline on {Config.DEVICE}")

    # Train the model (returns best validation score, but we will re-evaluate for analysis)
    # This step saves the best model to Config.BEST_MODEL_PATH
    print("\n--- Starting Training ---")
    _ = train_model(debug=Config.DEBUG)

    # 2. Load Best Model
    print("\n--- Loading Best Model ---")
    device = Config.DEVICE
    model = SPPDSNet().to(device)

    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
        )

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # 3. Validation & Failure Analysis
    print("\n--- Running Validation & Failure Analysis ---")
    # Get dataloaders to access the validation set and preprocessor
    _, val_loader, test_ds_base_loader, preprocessor = get_dataloaders(
        debug=Config.DEBUG
    )
    # Note: test_ds_base_loader is the loader for the provided test.csv (baseline only),
    # we will use its dataset later for image loading logic.
    test_ds_base = test_ds_base_loader.dataset

    val_results = []
    fvc_scale = preprocessor.target_scaler.scale_[0]

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            tabs = batch["tabular"].to(device)
            targets_raw = batch["target"].numpy()
            p_weeks = batch["patient_week"]

            # Inference
            mu_scaled, sigma_scaled = model(imgs, tabs)

            # Move to CPU
            mu_scaled = mu_scaled.cpu().numpy()
            sigma_scaled = sigma_scaled.cpu().numpy()

            # Inverse Transform
            # Handle shapes: inverse_transform_target expects (N, 1) or similar, returns (N, 1)
            mu_raw = preprocessor.inverse_transform_target(mu_scaled).flatten()

            # Sigma scaling: sigma_raw = sigma_scaled * std_dev
            sigma_raw = (sigma_scaled * fvc_scale).flatten()

            # Store results
            for i in range(len(targets_raw)):
                true_val = targets_raw[i]
                pred_val = mu_raw[i]
                sigma_val = sigma_raw[i]

                val_results.append(
                    {
                        "Patient_Week": p_weeks[i],
                        "True_FVC": true_val,
                        "Pred_FVC": pred_val,
                        "Pred_Sigma": sigma_val,
                        "Abs_Error": np.abs(true_val - pred_val),
                    }
                )

    val_df = pd.DataFrame(val_results)

    # Calculate Metric
    metric = calculate_metric(
        val_df["True_FVC"].values,
        val_df["Pred_FVC"].values,
        val_df["Pred_Sigma"].values,
    )
    print(f"Final Validation Metric: {metric}")

    # Failure Analysis: Correlation with features
    # Parse Patient and Week
    val_df["Patient"] = val_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    val_df["Weeks"] = val_df["Patient_Week"].apply(
        lambda x: int(x.split("_")[1])
    )  # Using 'Weeks' to match metadata

    # Load validation metadata to get original features for correlation
    val_meta = pd.read_csv(Config.VAL_CSV)

    # Merge metadata onto results
    # val_meta has ['Patient', 'Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus', ...]
    # We rename FVC in meta to avoid collision or just use suffixes
    analysis_df = val_df.merge(
        val_meta[["Patient", "Weeks", "Percent", "Age"]],
        on=["Patient", "Weeks"],
        how="left",
    )

    print("\nCorrelation between Absolute Error and Features:")
    features_to_check = ["Weeks", "Percent", "Age", "True_FVC"]
    for feat in features_to_check:
        if feat in analysis_df.columns:
            corr = analysis_df["Abs_Error"].corr(analysis_df[feat])
            print(f"  {feat}: {corr:.4f}")

    # 4. Submission
    THRESHOLD = -6.573619738753321
    if metric > THRESHOLD:
        print(
            f"\nMetric ({metric:.4f}) > Threshold ({THRESHOLD:.4f}). Generating submission..."
        )
        generate_submission(model, preprocessor, test_ds_base, device)
    else:
        print(
            f"\nMetric ({metric:.4f}) <= Threshold ({THRESHOLD:.4f}). Skipping submission."
        )


def generate_submission(model, preprocessor, base_dataset, device):
    """
    Generates submission file for all Patient_Weeks in sample_submission.csv.
    """
    # Load necessary files
    sample_sub = pd.read_csv(os.path.join(Config.INPUT_DIR, "sample_submission.csv"))
    test_meta = pd.read_csv(Config.TEST_CSV)

    # Create a map for fast lookup of baseline patient info
    test_patient_map = test_meta.set_index("Patient").to_dict("index")

    # Prepare data arrays
    X_list = []
    patients_list = []
    weeks_list = []
    patient_week_ids = []

    # Iterate over required predictions
    for idx, row in sample_sub.iterrows():
        pw = row["Patient_Week"]
        parts = pw.split("_")
        patient = parts[0]
        target_week = int(parts[1])

        if patient not in test_patient_map:
            continue

        base_info = test_patient_map[patient]

        # Extract raw baseline features
        base_fvc = base_info["FVC"]
        base_pct = base_info["Percent"]
        base_week = base_info["Weeks"]
        age = base_info["Age"]
        sex = base_info["Sex"]
        smoke = base_info["SmokingStatus"]

        # 1. Scale Numerical Features: [Base_FVC, Base_Percent, Age]
        # preprocessor.scaler was fitted on these 3 columns
        nums = np.array([[base_fvc, base_pct, age]])
        nums_scaled = preprocessor.scaler.transform(nums)[0]  # Shape (3,)

        # 2. Calculate Relative Week
        rel_week = (target_week - base_week) * Config.TIME_SCALE

        # 3. Encode Categoricals
        sex_code = preprocessor.sex_map.get(sex, 0)
        smoke_code = preprocessor.smoke_map.get(smoke, 0)

        # 4. Construct Feature Vector
        # Order must match TabularPreprocessor.transform:
        # [Base_FVC, Base_Percent, Rel_Week, Age, Sex, Smoking]
        feature_vec = np.array(
            [
                nums_scaled[0],  # Base_FVC
                nums_scaled[1],  # Base_Percent
                rel_week,  # Rel_Week
                nums_scaled[2],  # Age
                sex_code,  # Sex
                smoke_code,  # Smoking
            ],
            dtype=np.float32,
        )

        X_list.append(feature_vec)
        patients_list.append(patient)
        weeks_list.append(target_week)
        patient_week_ids.append(pw)

    X_submission = np.stack(X_list)

    # Custom Dataset for Submission
    class SubmissionDataset(Dataset):
        def __init__(self, X, patients, weeks, base_ds):
            self.X = X
            self.patients = patients
            self.weeks = weeks
            self.base_ds = base_ds

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            patient = self.patients[idx]
            # Reuse image loading logic from the base dataset (LungDataset)
            # This handles caching and DICOM processing
            img = self.base_ds._load_image(patient)

            return {
                "image": torch.tensor(img, dtype=torch.float32),
                "tabular": torch.tensor(self.X[idx], dtype=torch.float32),
                "patient_week": f"{patient}_{self.weeks[idx]}",  # Just for verification
            }

    # Create Loader
    sub_ds = SubmissionDataset(X_submission, patients_list, weeks_list, base_dataset)
    sub_loader = DataLoader(
        sub_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Inference
    all_preds_fvc = []
    all_preds_conf = []

    fvc_scale = preprocessor.target_scaler.scale_[0]

    with torch.no_grad():
        for batch in sub_loader:
            imgs = batch["image"].to(device)
            tabs = batch["tabular"].to(device)

            mu_scaled, sigma_scaled = model(imgs, tabs)

            mu_scaled = mu_scaled.cpu().numpy()
            sigma_scaled = sigma_scaled.cpu().numpy()

            # Inverse Transform
            mu_raw = preprocessor.inverse_transform_target(mu_scaled).flatten()
            sigma_raw = (sigma_scaled * fvc_scale).flatten()

            # Clip Confidence
            sigma_raw = np.maximum(sigma_raw, Config.SIGMA_CLIP)

            all_preds_fvc.extend(mu_raw)
            all_preds_conf.extend(sigma_raw)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "Patient_Week": patient_week_ids,
            "FVC": all_preds_fvc,
            "Confidence": all_preds_conf,
        }
    )

    # Save
    save_submission(submission_df, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
