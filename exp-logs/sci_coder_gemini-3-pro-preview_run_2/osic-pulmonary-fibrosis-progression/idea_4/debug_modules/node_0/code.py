import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch

# Import from the provided library
from library.config import Config, process_tabular_data
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.image_processing import process_patient_scan
from library.feature_extraction import CNNFeatureExtractor
from library.data_manager import create_interaction_features, get_static_features
from library.modeling import FVCPredictor, UncertaintyPredictor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Library Usage Demonstration ===\n")

    # 1. Setup and Reproducibility
    print("[1] Setting up environment...")
    seed_everything(Config.SEED)

    # Ensure working directory for cache exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    print(f"    Cache Directory: {Config.CACHE_DIR}")
    print(f"    Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # 2. Verify Metric Logic
    print("\n[2] Verifying Metric Function...")
    # Case 1: Perfect prediction
    # metric = - (sqrt(2) * 0) / 70 - ln(sqrt(2) * 70)
    #        = 0 - ln(98.99) ≈ -4.595
    y_true = np.array([2000])
    y_pred = np.array([2000])
    sigma = np.array([70])  # Minimum clipped value
    score = laplace_log_likelihood_metric(y_true, y_pred, sigma)

    expected_score = -np.log(np.sqrt(2) * 70)
    print(f"    Perfect Prediction Score: {score:.4f} (Expected: {expected_score:.4f})")
    assert np.isclose(
        score, expected_score, atol=1e-4
    ), "Metric calculation mismatch for perfect prediction"

    # Case 2: Large Error (clipped at 1000)
    y_pred_bad = np.array([4000])  # Delta = 2000 -> clipped to 1000
    score_bad = laplace_log_likelihood_metric(y_true, y_pred_bad, sigma)
    # metric = - (sqrt(2) * 1000) / 70 - ln(sqrt(2) * 70)
    #        = -20.203 - 4.595 = -24.798
    expected_bad = -(np.sqrt(2) * 1000) / 70 - np.log(np.sqrt(2) * 70)
    print(
        f"    Clipped Error Score:      {score_bad:.4f} (Expected: {expected_bad:.4f})"
    )
    assert np.isclose(
        score_bad, expected_bad, atol=1e-4
    ), "Metric calculation mismatch for clipped error"
    print("    Metric verification passed.")

    # 3. Image Processing Demonstration
    print("\n[3] Demonstrating Image Processing...")
    # Load training metadata to get a valid patient path
    train_meta_path = os.path.join(Config.METADATA_DIR, "train_metadata.csv")
    df_train = pd.read_csv(train_meta_path)

    # Select the first patient
    sample_patient = df_train.iloc[0]
    patient_id = sample_patient["Patient"]
    rel_path = sample_patient["dcm_path"]
    full_path = os.path.join(Config.INPUT_DIR, rel_path)

    print(f"    Processing scan for patient: {patient_id}")
    # Process scan (Load -> HU -> Select Variance Slices -> Preprocess)
    # We disable loading from cache to demonstrate the actual processing logic
    volume = process_patient_scan(
        patient_id=patient_id,
        dcm_path=full_path,
        load_cached_data=False,
        n_slices=Config.SLICES_PER_PATIENT,
        img_size=Config.IMG_SIZE,
    )

    print(f"    Output Volume Shape: {volume.shape}")

    # Validation
    expected_shape = (Config.SLICES_PER_PATIENT, Config.IMG_SIZE, Config.IMG_SIZE, 3)
    assert (
        volume.shape == expected_shape
    ), f"Volume shape mismatch. Expected {expected_shape}, got {volume.shape}"
    assert volume.dtype == np.float32, "Volume should be float32"
    assert (
        volume.min() >= 0.0 and volume.max() <= 1.0
    ), "Pixel values should be normalized to [0, 1]"
    print("    Image processing verification passed.")

    # 4. Feature Extraction Demonstration (Subset)
    print("\n[4] Demonstrating Feature Extraction (Subset)...")
    # We will use a tiny subset of 3 patients to keep execution fast
    subset_df = df_train.head(3).copy()
    print(f"    Extracting features for {len(subset_df)} patients...")

    extractor = CNNFeatureExtractor(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    features_list = []
    for _, row in subset_df.iterrows():
        # Re-process scan (using cache if available from step 3 for the first one)
        full_dcm_path = os.path.join(Config.INPUT_DIR, row["dcm_path"])
        vol = process_patient_scan(
            patient_id=row["Patient"], dcm_path=full_dcm_path, load_cached_data=True
        )
        # Extract embedding
        emb = extractor.extract(vol)
        features_list.append(emb)

    raw_features = np.vstack(features_list)
    print(f"    Extracted Features Shape: {raw_features.shape}")

    # Validation
    # EfficientNet-B0 usually outputs 1280 dim features
    assert len(raw_features) == 3
    assert (
        raw_features.shape[1] == 1280
    ), f"Expected 1280 features, got {raw_features.shape[1]}"
    print("    Feature extraction verification passed.")

    # 5. Data Management & Feature Construction
    print("\n[5] Demonstrating Data Management & Feature Construction...")

    # Simulate PCA reduction (since we have too few samples to run actual PCA meaningfully)
    # In a real run, we would fit PCA on the whole training set.
    # Here we just project to random components for structural demonstration.
    pca_projection = np.random.randn(1280, Config.PCA_COMPONENTS)
    img_pca_features = np.dot(raw_features, pca_projection)

    # Process Tabular Data
    # This adds 'Relative_Weeks', 'Sex_Male', 'Smoking_Ex', etc.
    df_processed = process_tabular_data(subset_df, mode="train_subset")

    # Extract Static Features defined in Config
    X_base = df_processed[Config.STATIC_COLS].values.astype(np.float32)

    # Merge Static + Image
    X_static = np.hstack([X_base, img_pca_features])

    # Get Time Variable
    weeks = df_processed["Relative_Weeks"].values.astype(np.float32)

    # Create Interaction Features for the Varying-Coefficient Model
    # [Static, Time, Static * Time]
    X_full = create_interaction_features(X_static, weeks)

    y_subset = df_processed["FVC"].values

    print(f"    Static Features Shape: {X_static.shape}")
    print(f"    Full Features Shape:   {X_full.shape}")

    # Validation
    # Static cols (6) + PCA (30) = 36
    n_static = len(Config.STATIC_COLS) + Config.PCA_COMPONENTS
    assert X_static.shape[1] == n_static, f"Expected {n_static} static features"

    # Full: Static (36) + Time (1) + Interaction (36) = 73
    n_full = n_static + 1 + n_static
    assert X_full.shape[1] == n_full, f"Expected {n_full} full features"
    print("    Feature construction verification passed.")

    # 6. Modeling Demonstration
    print("\n[6] Demonstrating Model Training...")

    # --- Stage 1: FVC Predictor ---
    print("    Training FVC Predictor (Quantile Regression)...")
    fvc_model = FVCPredictor(quantile=0.5, alpha=1.0)
    fvc_model.fit(X_full, y_subset)

    # Predict
    y_pred = fvc_model.predict(X_full)
    mae = np.mean(np.abs(y_subset - y_pred))
    print(f"    FVC Training MAE: {mae:.2f}")

    assert y_pred.shape == y_subset.shape

    # --- Stage 2: Uncertainty Predictor ---
    print("    Training Uncertainty Predictor (ElasticNet)...")
    residuals = np.abs(y_subset - y_pred)

    # Use static features only for uncertainty
    unc_model = UncertaintyPredictor(alpha=0.1)
    unc_model.fit(X_static, residuals)

    # Predict MAD
    mad_pred = unc_model.predict(X_static)

    # Convert to Sigma
    sigma_pred = mad_pred * np.sqrt(2)

    assert sigma_pred.shape == y_subset.shape
    print(f"    Mean Predicted Sigma: {np.mean(sigma_pred):.2f}")
    print("    Modeling verification passed.")

    # 7. Inference Logic
    print("\n[7] Demonstrating Inference Logic...")
    # Simulate a test case
    test_sigma = np.maximum(sigma_pred, 70)  # Clip at 70

    submission_df = pd.DataFrame(
        {
            "Patient_Week": df_processed["Patient"]
            + "_"
            + df_processed["Weeks"].astype(str),
            "FVC": y_pred,
            "Confidence": test_sigma,
        }
    )

    print("    Sample Submission Rows:")
    print(submission_df.head())

    assert "Patient_Week" in submission_df.columns
    assert "FVC" in submission_df.columns
    assert "Confidence" in submission_df.columns
    assert (submission_df["Confidence"] >= 70).all(), "Confidence values must be >= 70"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
