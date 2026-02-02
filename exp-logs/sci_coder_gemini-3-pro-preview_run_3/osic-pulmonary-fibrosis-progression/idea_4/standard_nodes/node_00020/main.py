import sys
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Ensure the working directory is in the path
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.trainer import Trainer
from library.dataset import LungFVCDataset
from library.utils import unscale_data, laplace_log_likelihood_metric
from library.model import DualPathTransformer


def run_failure_analysis(model, val_loader, device):
    """
    Runs inference on validation set, computes the official metric,
    and calculates correlations between absolute error and input features.
    """
    model.eval()

    predictions = []
    sigmas = []
    targets = []

    # Feature collectors
    ages = []
    weeks = []
    sexes = []
    smokings = []

    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            images = batch["images"].to(device)
            meta_age = batch["meta_age"].to(device)
            meta_sex = batch["meta_sex"].to(device)
            meta_smoke = batch["meta_smoke"].to(device)
            linear_features = batch["linear_features"].to(device)
            target_std = batch["target"].to(device)

            # Forward pass
            fvc_pred_std, sigma_pred_std = model(
                images, meta_age, meta_sex, meta_smoke, linear_features
            )

            # Unscale predictions
            fvc_pred, sigma_pred = unscale_data(fvc_pred_std, sigma_pred_std)
            fvc_true, _ = unscale_data(target_std, torch.zeros_like(target_std))

            # Store results
            predictions.extend(fvc_pred.cpu().numpy())
            sigmas.extend(sigma_pred.cpu().numpy())
            targets.extend(fvc_true.cpu().numpy())

            # Store features (convert from tensor to numpy)
            # meta_age is standardized, but correlation works fine with linear transforms
            ages.extend(batch["meta_age"].cpu().numpy())
            weeks.extend(batch["weeks"].numpy())
            sexes.extend(batch["meta_sex"].cpu().numpy())
            smokings.extend(batch["meta_smoke"].cpu().numpy())

    predictions = np.array(predictions)
    sigmas = np.array(sigmas)
    targets = np.array(targets)

    # Compute Metric
    metric = laplace_log_likelihood_metric(targets, predictions, sigmas)
    print(f"Final Validation Metric: {metric}")

    # Compute Failure Analysis (Correlations)
    abs_errors = np.abs(predictions - targets)

    print("\n--- Failure Analysis (Correlation with Absolute Error) ---")

    # Calculate Pearson correlations
    # Handle potential constant arrays (though unlikely)
    def safe_corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return pearsonr(x, y)[0]

    corr_age = safe_corr(abs_errors, ages)
    corr_weeks = safe_corr(abs_errors, weeks)
    corr_sex = safe_corr(abs_errors, sexes)
    corr_smoke = safe_corr(abs_errors, smokings)

    print(f"Age: {corr_age:.4f}")
    print(f"Weeks: {corr_weeks:.4f}")
    print(f"Sex: {corr_sex:.4f}")
    print(f"SmokingStatus: {corr_smoke:.4f}")

    return metric


def generate_submission(model, device):
    """
    Generates the submission.csv file for the test set.
    """
    print("\nGenerating submission...")

    # Load metadata
    test_df = pd.read_csv(Config.TEST_META_PATH)
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Parse Patient and Weeks from Patient_Week column
    sub_df[["Patient", "Weeks_Str"]] = sub_df["Patient_Week"].str.split(
        "_", expand=True
    )
    sub_df["Weeks"] = sub_df["Weeks_Str"].astype(int)

    # Map static metadata and Baseline FVC from test_df to submission rows
    # 1. Create mapping dictionaries
    # Ensure image_path exists
    if "image_path" not in test_df.columns:
        test_df["image_path"] = test_df["Patient"].apply(
            lambda x: os.path.join("test", x)
        )

    patient_to_age = test_df.set_index("Patient")["Age"].to_dict()
    patient_to_sex = test_df.set_index("Patient")["Sex"].to_dict()
    patient_to_smoke = test_df.set_index("Patient")["SmokingStatus"].to_dict()
    patient_to_img = test_df.set_index("Patient")["image_path"].to_dict()
    patient_to_fvc = test_df.set_index("Patient")["FVC"].to_dict()

    # 2. Apply mappings
    sub_df["Age"] = sub_df["Patient"].map(patient_to_age)
    sub_df["Sex"] = sub_df["Patient"].map(patient_to_sex)
    sub_df["SmokingStatus"] = sub_df["Patient"].map(patient_to_smoke)
    sub_df["image_path"] = sub_df["Patient"].map(patient_to_img)

    # CRITICAL: Overwrite FVC in submission rows with the Baseline FVC from test_df.
    # This ensures the Dataset class uses the correct value for the "Baseline FVC" feature
    # regardless of which row it picks as the reference.
    sub_df["FVC"] = sub_df["Patient"].map(patient_to_fvc)

    # 3. Create Dataset
    # We pass sub_df directly. Dataset will compute baselines from the 'FVC' column we just populated.
    dataset = LungFVCDataset(sub_df, mode="test")

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Inference
    model.eval()
    all_fvc = []
    all_sigma = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            meta_age = batch["meta_age"].to(device)
            meta_sex = batch["meta_sex"].to(device)
            meta_smoke = batch["meta_smoke"].to(device)
            linear_features = batch["linear_features"].to(device)

            fvc_pred_std, sigma_pred_std = model(
                images, meta_age, meta_sex, meta_smoke, linear_features
            )

            fvc_pred, sigma_pred = unscale_data(fvc_pred_std, sigma_pred_std)

            all_fvc.extend(fvc_pred.cpu().numpy())
            all_sigma.extend(sigma_pred.cpu().numpy())

    # 5. Format Submission
    sub_df["FVC_Pred"] = all_fvc
    sub_df["Sigma_Pred"] = all_sigma

    # Apply post-processing
    # Clip confidence to 70 as per metric
    sub_df["Confidence"] = np.maximum(sub_df["Sigma_Pred"], 70.0)
    sub_df["FVC"] = sub_df["FVC_Pred"]

    # Select final columns
    submission = sub_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Initialize Trainer
    # debug=False uses the full dataset as per Config
    trainer = Trainer(debug=Config.DEBUG)

    # Train Model
    # Using Config.EPOCHS (35)
    trainer.fit(epochs=Config.EPOCHS)

    # Load Best Model for Validation
    print("Loading best model...")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    model = DualPathTransformer().to(trainer.device)
    model.load_state_dict(torch.load(best_model_path, map_location=trainer.device))

    # Run Validation and Failure Analysis
    _, val_loader = trainer.get_dataloaders()
    metric = run_failure_analysis(model, val_loader, trainer.device)

    # Generate Submission if Metric is good
    THRESHOLD = -6.6997912217
    if metric > THRESHOLD:
        generate_submission(model, trainer.device)
    else:
        print(f"Metric {metric} is not greater than {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
