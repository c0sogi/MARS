import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch

# Import from the provided library files
from library.config import Config
from library.cnn_model import generate_embeddings
from library.data_processor import DataProcessor
from library.regressors import QuantileModel, ResidualModel, laplace_log_likelihood


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_mini_dataset(input_path, output_path, n_patients):
    """
    Creates a smaller dataset by sampling a subset of patients.
    Ensures all records for selected patients are kept.
    """
    df = pd.read_csv(input_path)

    # Identify unique patients
    if "Patient" in df.columns:
        patient_col = "Patient"
    else:
        # For submission file/test_metadata where Patient is derived or separate
        patient_col = "Patient"

    unique_patients = df[patient_col].unique()

    # Select subset
    if len(unique_patients) > n_patients:
        selected_patients = unique_patients[:n_patients]
        df_subset = df[df[patient_col].isin(selected_patients)].copy()
    else:
        df_subset = df.copy()

    df_subset.to_csv(output_path, index=False)
    return df_subset


def run_demo():
    # 1. Setup
    print("=== Starting 2.5D Context-Aware Pipeline Demo ===")
    warnings.filterwarnings("ignore")
    set_seed(Config.SEED)

    # Override Cache Directory for Demo to avoid conflicts
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_run", "cache")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define paths for mini datasets
    mini_train_path = os.path.join(Config.WORKING_DIR, "demo_run", "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "demo_run", "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "demo_run", "mini_test.csv")

    # 2. Create Mini Datasets (Subset of Patients)
    # We need enough training samples for PCA (n_components=30)
    print("Creating mini datasets...")
    df_train_mini = create_mini_dataset(
        Config.TRAIN_METADATA_PATH, mini_train_path, n_patients=40
    )
    df_val_mini = create_mini_dataset(
        Config.VAL_METADATA_PATH, mini_val_path, n_patients=10
    )
    df_test_mini = create_mini_dataset(
        Config.TEST_METADATA_PATH, mini_test_path, n_patients=5
    )

    print(f"Mini Train Shape: {df_train_mini.shape}")
    print(f"Mini Val Shape: {df_val_mini.shape}")
    print(f"Mini Test Shape: {df_test_mini.shape}")

    # 3. Generate Image Embeddings (CNN + Radiomics)
    # We pass the mini dataframes directly.
    # load_cached_data=False forces the pipeline to run image processing.
    print("\n=== Generating Image Embeddings ===")

    train_feats = generate_embeddings(
        df_train_mini, "train_mini", load_cached_data=False
    )
    val_feats = generate_embeddings(df_val_mini, "val_mini", load_cached_data=False)
    test_feats = generate_embeddings(df_test_mini, "test_mini", load_cached_data=False)

    # Verify Embedding Shapes
    # Embeddings: (N_patients, 1280) for EfficientNet-B0
    # Radiomics: (N_patients, 4)
    assert train_feats["embeddings"].shape[1] == 1280, "Unexpected embedding dimension"
    assert train_feats["radiomics"].shape[1] == 4, "Unexpected radiomics dimension"
    print("Embeddings generated successfully.")

    # 4. Process Data (Merge Tabular + Image, PCA, Scaling)
    print("\n=== Processing Tabular & Image Data ===")
    processor = DataProcessor()

    # Note: We pass the paths to the mini CSVs we created
    data_matrices = processor.process_data(
        mini_train_path,
        mini_val_path,
        mini_test_path,
        train_feats,
        val_feats,
        test_feats,
        load_cached_data=False,
    )

    # Unpack matrices
    X_train_fvc = data_matrices["train"]["X_fvc"]
    y_train = data_matrices["train"]["y"]
    X_train_unc = data_matrices["train"]["X_unc"]

    X_val_fvc = data_matrices["val"]["X_fvc"]
    y_val = data_matrices["val"]["y"]
    X_val_unc = data_matrices["val"]["X_unc"]

    X_test_fvc = data_matrices["test"]["X_fvc"]
    X_test_unc = data_matrices["test"]["X_unc"]
    test_patient_weeks = data_matrices["test"]["patient_weeks"]

    print(f"Train Feature Matrix (FVC): {X_train_fvc.shape}")
    print(f"Train Feature Matrix (Uncertainty): {X_train_unc.shape}")

    # 5. Model Training
    print("\n=== Training Models ===")

    # A. FVC Model (Quantile Regression)
    print("Training Quantile Model (FVC)...")
    fvc_model = QuantileModel(quantile=Config.QUANTILE)
    fvc_model.fit(X_train_fvc, y_train)

    # Predict on Train to get residuals for Uncertainty Model
    train_preds = fvc_model.predict(X_train_fvc)
    train_residuals = np.abs(y_train - train_preds)

    # B. Uncertainty Model (ElasticNet on Residuals)
    print("Training Residual Model (Uncertainty)...")
    unc_model = ResidualModel()
    unc_model.fit(X_train_unc, train_residuals)

    # 6. Validation & Evaluation
    print("\n=== Validation ===")
    val_preds_fvc = fvc_model.predict(X_val_fvc)
    val_preds_sigma = unc_model.predict(X_val_unc)

    # Compute Metric
    metric_score = laplace_log_likelihood(y_val, val_preds_fvc, val_preds_sigma)
    print(f"Validation Laplace Log Likelihood: {metric_score:.4f}")

    # Assertions to ensure logic is correct
    assert len(val_preds_fvc) == len(y_val), "Prediction length mismatch"
    assert np.all(val_preds_sigma >= 0), "Negative uncertainty values detected"

    # 7. Test Inference & Submission Construction
    print("\n=== Test Inference ===")
    test_preds_fvc = fvc_model.predict(X_test_fvc)
    test_preds_sigma = unc_model.predict(X_test_unc)

    # Construct Submission DataFrame
    submission = pd.DataFrame(
        {
            "Patient_Week": test_patient_weeks,
            "FVC": test_preds_fvc,
            "Confidence": test_preds_sigma,
        }
    )

    # Verify Submission Format
    print("Sample Submission Rows:")
    print(submission.head())

    assert "Patient_Week" in submission.columns
    assert "FVC" in submission.columns
    assert "Confidence" in submission.columns
    assert len(submission) == len(df_test_mini), "Submission row count mismatch"

    # Save submission (optional for demo, but good practice)
    submission_path = os.path.join(Config.WORKING_DIR, "demo_run", "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
