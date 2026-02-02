import os
import numpy as np
import pandas as pd
import scipy.sparse
from library.config import Config
from library.utils import set_seed, JigsawMetrics
from library.data_loader import load_data
from library.features import FeatureExtractor
from library.model import RidgeRegressor


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("[Demo] Setting up configuration for fast execution...")

    # Override Config for speed and demonstration purposes
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5000  # Small subset for quick runtime

    # Reduce dimensionality for faster vectorization
    Config.WORD_MAX_FEATURES = 1000
    Config.CHAR_MAX_FEATURES = 1000

    # Set output paths for this run
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist (Config __init__ usually does this, but we changed paths)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(Config.SEED)

    # ==========================================
    # 2. Data Loading & Bias Resampling
    # ==========================================
    print("\n[Demo] Loading and resampling data...")
    # We force load_cached_data=False to demonstrate the raw processing logic
    train_df, val_df, test_df = load_data(load_cached_data=False, debug=True)

    # Verification: Check Data Loading
    print("[Verification] Verifying data shapes...")
    assert len(train_df) > 0, "Training data should not be empty."
    assert len(val_df) > 0, "Validation data should not be empty."
    assert len(test_df) > 0, "Test data should not be empty."

    # Verification: Check Resampling Logic
    # In debug mode, we sample DEBUG_SAMPLE_SIZE first.
    # If resampling is active (weight > 1), train_df should be larger than the unique rows if identities exist.
    # However, if the random sample happens to have no identities, it might remain same size.
    # We check if 'target' column exists in train/val and NOT in test.
    assert "target" in train_df.columns
    assert "target" in val_df.columns
    assert "target" not in test_df.columns
    print("Data loading verified successfully.")

    # ==========================================
    # 3. Feature Extraction
    # ==========================================
    print("\n[Demo] Extracting TF-IDF features...")
    extractor = FeatureExtractor()

    # Extract features (this handles fitting on train and transforming all)
    X_train, X_val, X_test = extractor.extract_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verification: Check Feature Matrix Dimensions
    print("[Verification] Verifying feature matrices...")
    n_samples_train = train_df.shape[0]
    n_samples_val = val_df.shape[0]
    n_samples_test = test_df.shape[0]

    assert (
        X_train.shape[0] == n_samples_train
    ), f"X_train rows {X_train.shape[0]} != df rows {n_samples_train}"
    assert (
        X_val.shape[0] == n_samples_val
    ), f"X_val rows {X_val.shape[0]} != df rows {n_samples_val}"
    assert (
        X_test.shape[0] == n_samples_test
    ), f"X_test rows {X_test.shape[0]} != df rows {n_samples_test}"

    # Check feature count (Word + Char)
    # Note: TfidfVectorizer might produce fewer features than MAX_FEATURES if vocab is small
    expected_max_cols = Config.WORD_MAX_FEATURES + Config.CHAR_MAX_FEATURES
    assert (
        X_train.shape[1] <= expected_max_cols
    ), "Feature count exceeds configured maximum."
    print(
        f"Feature extraction verified. Dimensions: Train={X_train.shape}, Val={X_val.shape}"
    )

    # ==========================================
    # 4. Model Training
    # ==========================================
    print("\n[Demo] Training Ridge Regression model...")
    model = RidgeRegressor(alpha=1.0)

    # Train on the resampled training set
    model.train(X_train, train_df["target"].values)

    # Verification: Basic Prediction Check
    print("[Verification] Checking prediction output range...")
    sample_preds = model.predict(X_val[:10])
    assert np.all(sample_preds >= 0.0) and np.all(
        sample_preds <= 1.0
    ), "Predictions must be clipped to [0, 1]."
    print("Model training and prediction verified.")

    # ==========================================
    # 5. Evaluation (Bias Metrics)
    # ==========================================
    print("\n[Demo] Evaluating model performance...")
    metrics_results = model.evaluate(X_val, val_df, target_col="target")

    # Verification: Check Metrics Structure
    print("[Verification] Verifying metric outputs...")
    required_keys = ["score", "overall_auc", "subgroup_auc", "bpsn_auc", "bnsp_auc"]
    for key in required_keys:
        assert key in metrics_results, f"Missing metric key: {key}"
        assert isinstance(
            metrics_results[key], float
        ), f"Metric {key} should be a float."

    print(f"Evaluation verified. Score: {metrics_results['score']:.4f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n[Demo] Generating submission file...")
    test_preds = model.predict(X_test)
    model.save_submission(test_df["id"], test_preds)

    # Verification: Check File Artifact
    print("[Verification] Checking submission file...")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert sub_df.shape == (len(test_df), 2), "Submission shape mismatch."
    assert list(sub_df.columns) == ["id", "prediction"], "Submission columns mismatch."
    print("Submission generation verified.")

    # ==========================================
    # 7. Logic Verification: Synthetic Metrics Test
    # ==========================================
    print("\n[Demo] Running synthetic logic verification for JigsawMetrics...")
    # Create a controlled scenario to ensure the complex metric logic works as expected
    # Scenario:
    # - 4 samples
    # - Identity 'male' present in first 2
    # - Target: [1, 0, 1, 0] (Toxic, Non-Toxic, Toxic, Non-Toxic)
    # - Preds:  [0.9, 0.2, 0.8, 0.1] (Good model)

    synthetic_df = pd.DataFrame(
        {
            "target": [1.0, 0.0, 1.0, 0.0],
            "male": [1.0, 1.0, 0.0, 0.0],  # Identity present in first two
            "female": [0.0, 0.0, 0.0, 0.0],  # Other identities empty
        }
    )
    synthetic_preds = np.array([0.9, 0.2, 0.8, 0.1])

    metric_helper = JigsawMetrics()
    # We only care about 'male' for this specific check, others will be 0.5 (default for empty/single class)
    results = metric_helper.compute_bias_metrics(synthetic_df, synthetic_preds)

    # Check Subgroup AUC for 'male'
    # Subgroup: indices [0, 1]. Targets: [1, 0]. Preds: [0.9, 0.2]. Perfect separation -> AUC 1.0
    male_subgroup_auc = results["per_subgroup_auc"]["male"]

    # Check BPSN for 'male' (Background Positive, Subgroup Negative)
    # Background Positive: Toxic + No Identity (index 2)
    # Subgroup Negative: Non-Toxic + Identity (index 1)
    # Set: indices [2, 1]. Targets: [1, 0]. Preds: [0.8, 0.2]. Perfect separation -> AUC 1.0
    male_bpsn_auc = results["per_bpsn_auc"]["male"]

    # Check BNSP for 'male' (Background Negative, Subgroup Positive)
    # Background Negative: Non-Toxic + No Identity (index 3)
    # Subgroup Positive: Toxic + Identity (index 0)
    # Set: indices [3, 0]. Targets: [0, 1]. Preds: [0.1, 0.9]. Perfect separation -> AUC 1.0
    male_bnsp_auc = results["per_bnsp_auc"]["male"]

    assert (
        male_subgroup_auc == 1.0
    ), f"Synthetic Subgroup AUC failed. Got {male_subgroup_auc}"
    assert male_bpsn_auc == 1.0, f"Synthetic BPSN AUC failed. Got {male_bpsn_auc}"
    assert male_bnsp_auc == 1.0, f"Synthetic BNSP AUC failed. Got {male_bnsp_auc}"

    print("Synthetic metric logic verification passed.")
    print("\n[Demo] All steps completed successfully.")


if __name__ == "__main__":
    run_demo()
