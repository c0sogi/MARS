import os
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.image_processing import ImageProcessor
from library.feature_engineering import FeatureProcessor
from library.models import QuantileLinearModel, ResidualElasticModel
from library.train_eval import Trainer


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("=== Setting up Configuration ===")

    # Override Config for the demo to ensure speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5  # Process only 5 patients per split
    Config.CACHE_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.N_JOBS = 2  # Limit threads for the demo

    # Create necessary directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Cache Directory: {Config.CACHE_DIR}")

    # ==========================================
    # 2. Demonstrate Utils (Metric)
    # ==========================================
    print("\n=== Demonstrating Utils (Metric) ===")

    # Test Case: Perfect prediction
    # FVC_true=2000, FVC_pred=2000, Sigma=100
    # Delta = 0
    # Sigma_clipped = max(100, 70) = 100
    # Metric = - (sqrt(2)*0)/100 - ln(sqrt(2)*100)
    #        = - ln(141.421356) ≈ -4.9517

    y_true = np.array([2000])
    y_pred = np.array([2000])
    sigma = np.array([100])

    score = laplace_log_likelihood(y_true, y_pred, sigma)
    print(f"Calculated Score: {score}")

    expected_score = -np.log(np.sqrt(2) * 100)
    assert np.isclose(score, expected_score, atol=1e-4), "Metric calculation mismatch!"
    print("Metric verification passed.")

    # ==========================================
    # 3. Demonstrate Image Processing
    # ==========================================
    print("\n=== Demonstrating Image Processing ===")

    # Get a sample patient from metadata to ensure path validity
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    sample_patient = df_train.iloc[0]
    pid = sample_patient["Patient"]
    dcm_rel_path = sample_patient["dcm_path"]

    print(f"Processing sample patient: {pid}")

    # Initialize Processor (Force CPU for demo reliability if GPU is busy, though Config handles it)
    # We stick to Config.DEVICE for consistency
    img_processor = ImageProcessor()

    # 3.1 Load Scan
    full_path = os.path.join(Config.INPUT_DIR, dcm_rel_path)
    volume = img_processor.load_scan(full_path)
    print(f"Loaded volume shape: {volume.shape}")

    # 3.2 Extract Morphological Features
    morph_feats = img_processor.extract_morphological_profile(volume)
    print(f"Morphological features shape: {morph_feats.shape}")
    # Expecting 8 coefficients (4 area + 4 density)
    assert morph_feats.shape == (8,), f"Expected shape (8,), got {morph_feats.shape}"

    # 3.3 Extract Texture Features
    # Note: This might be slow on CPU, but we only do one patient here
    texture_feats = img_processor.extract_stratified_texture(volume)
    print(f"Texture features shape: {texture_feats.shape}")
    # Expecting 3 slices * 1280 features (EfficientNet-B0) = 3840
    assert texture_feats.shape == (
        3840,
    ), f"Expected shape (3840,), got {texture_feats.shape}"

    print("Image processing verification passed.")

    # ==========================================
    # 4. Demonstrate Feature Engineering
    # ==========================================
    print("\n=== Demonstrating Feature Engineering ===")

    feat_processor = FeatureProcessor()

    # Run the full pipeline (Load -> Extract -> Fit -> Transform)
    # load_cached_data=False forces re-computation for the demo
    train_data, val_data, test_data = feat_processor.process_pipelines(
        load_cached_data=False
    )

    # Verify Data Dictionary Structure
    required_keys = ["X_fvc", "X_unc", "y", "weeks", "patient"]
    for key in required_keys:
        assert key in train_data, f"Missing key {key} in train_data"

    # Verify Shapes
    # X_fvc should have:
    #   30 (PCA) + 8 (Morph) + 2 (Clinical Cont) + 5 (Clinical Cat: Sex=2, Smoking=3) = 45 static features
    #   + 1 (Time) + 45 (Interactions) = 91 features total
    # Note: OneHotEncoder output size depends on unique values in the subset.
    # In full data: Sex(2) + Smoking(3) = 5.
    # In debug subset (5 samples), we might have fewer categories, so exact dimension check is tricky.
    # We check basic consistency.

    n_samples = train_data["X_fvc"].shape[0]
    n_features_fvc = train_data["X_fvc"].shape[1]
    n_features_unc = train_data["X_unc"].shape[1]

    print(f"Train samples: {n_samples}")
    print(f"FVC Model Features: {n_features_fvc}")
    print(f"Uncertainty Model Features: {n_features_unc}")

    assert (
        n_samples == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} samples, got {n_samples}"
    assert n_features_fvc > 10, "Feature matrix seems too small."
    assert train_data["y"].shape[0] == n_samples, "Target mismatch."

    print("Feature engineering verification passed.")

    # ==========================================
    # 5. Demonstrate Models
    # ==========================================
    print("\n=== Demonstrating Models ===")

    X_train_fvc = train_data["X_fvc"]
    X_train_unc = train_data["X_unc"]
    y_train = train_data["y"]

    # 5.1 Quantile Model (FVC)
    print("Training QuantileLinearModel...")
    fvc_model = QuantileLinearModel(quantile=0.5, alpha=0.1)
    fvc_model.fit(X_train_fvc, y_train)

    fvc_pred = fvc_model.predict(X_train_fvc)
    assert fvc_pred.shape == y_train.shape, "Prediction shape mismatch"
    print(f"FVC Prediction Mean: {np.mean(fvc_pred):.2f}")

    # 5.2 Residual Model (Uncertainty)
    print("Training ResidualElasticModel...")
    # Compute residuals from FVC model
    residuals = np.abs(y_train - fvc_pred)

    unc_model = ResidualElasticModel(alpha=0.1, l1_ratio=0.5)
    unc_model.fit(X_train_unc, residuals)

    unc_pred = unc_model.predict(X_train_unc)
    assert np.all(unc_pred >= 0), "Uncertainty predictions must be non-negative"
    print(f"Uncertainty Prediction Mean: {np.mean(unc_pred):.2f}")

    print("Model verification passed.")

    # ==========================================
    # 6. Demonstrate Full Training Pipeline
    # ==========================================
    print("\n=== Demonstrating Full Trainer Pipeline ===")

    # Instantiate Trainer
    trainer = Trainer()

    # We can inject the data we already processed to save time,
    # but the Trainer usually loads it internally.
    # For demonstration of the class methods, we'll follow the standard flow.

    # 6.1 Train
    # We pass the dictionaries we created in Step 4
    trainer.train(train_data)

    # 6.2 Evaluate
    score = trainer.evaluate(val_data)
    print(f"Evaluation Score: {score}")

    # 6.3 Generate Submission
    trainer.generate_submission(test_data)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(sub_df.head())

    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in sub_df.columns for col in expected_cols
    ), "Missing columns in submission."
    assert len(sub_df) == len(
        test_data["patient_week"]
    ), "Submission row count mismatch."

    # 6.4 Save Models
    trainer.save_models()
    assert os.path.exists(
        os.path.join(Config.CACHE_DIR, "fvc_model.joblib")
    ), "FVC model not saved."
    assert os.path.exists(
        os.path.join(Config.CACHE_DIR, "unc_model.joblib")
    ), "Uncertainty model not saved."

    print("Full pipeline verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
