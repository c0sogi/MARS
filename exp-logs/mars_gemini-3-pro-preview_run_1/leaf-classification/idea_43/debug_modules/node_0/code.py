import os
import sys
import numpy as np
import pandas as pd
import warnings
import random
from sklearn.metrics import log_loss, accuracy_score

# Import provided library components
from library.config import (
    SEED,
    VISUAL_FEATURES,
    SUBMISSION_DIR,
    METADATA_DIR,
    INPUT_DIR,
)
from library.feature_extraction import extract_single_image_features
from library.data_manager import LeafDataManager
from library.preprocessor import HighPrecisionPreprocessor
from library.classifier import OASDiscriminant

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demo_feature_extraction():
    print("\n--- 1. Demonstrating Feature Extraction ---")
    # Load train metadata to get a valid image path
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    df_train = pd.read_csv(train_meta_path)

    # Select the first image
    sample_row = df_train.iloc[0]
    rel_path = sample_row["file_path"]
    full_path = os.path.join(INPUT_DIR, rel_path)

    print(f"Extracting features for image: {rel_path}")

    # Extract features
    features = extract_single_image_features(full_path)

    # Validation
    assert isinstance(features, dict), "Output must be a dictionary"
    assert all(
        k in features for k in VISUAL_FEATURES
    ), "Missing keys in feature dictionary"
    assert all(
        isinstance(v, float) for v in features.values()
    ), "All feature values must be floats"

    print("Feature extraction successful.")
    print(
        f"Sample Features: Area={features['area']:.2f}, Solidity={features['solidity']:.4f}"
    )


def demo_preprocessor():
    print("\n--- 2. Demonstrating HighPrecisionPreprocessor ---")
    # Create synthetic data: 10 samples, 5 features
    X_synth = np.random.rand(10, 5).astype(np.float32)  # Start with float32

    # Instantiate preprocessor
    preprocessor = HighPrecisionPreprocessor(use_yeo_johnson=True, standardize=True)

    # Fit and Transform
    print("Fitting and transforming synthetic data...")
    X_trans = preprocessor.fit_transform(X_synth)

    # Validation
    assert X_trans.dtype == np.float64, "Preprocessor must return float64"
    assert X_trans.shape == X_synth.shape, "Shape mismatch after transformation"

    # Check standardization (mean approx 0, std approx 1)
    means = np.mean(X_trans, axis=0)
    stds = np.std(X_trans, axis=0)

    print(f"Transformed Data Type: {X_trans.dtype}")
    print(f"Mean of features (should be ~0): {means}")
    print(f"Std of features (should be ~1): {stds}")

    assert np.allclose(means, 0, atol=1e-7), "Standardization failed (Mean != 0)"
    assert np.allclose(stds, 1, atol=1e-7), "Standardization failed (Std != 1)"
    print("Preprocessor demonstration passed.")


def run_pipeline():
    print("\n--- 3. Running Full Pipeline (Data Loading -> Model -> Submission) ---")

    # 1. Load Data using DataManager
    # This handles feature extraction, caching, merging, and internal preprocessing
    dm = LeafDataManager()
    print("Loading and processing data...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = dm.load_data()

    # Validation of loaded data
    print(
        f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )
    assert X_train.dtype == np.float64, "X_train must be float64"
    assert len(classes) == 99, f"Expected 99 classes, found {len(classes)}"

    # 2. Train Classifier
    print("\nTraining OAS Discriminant Classifier...")
    clf = OASDiscriminant(
        assume_centered=True
    )  # Data is already standardized by DataManager
    clf.fit(X_train, y_train)

    # 3. Evaluate on Validation Set
    print("Evaluating on Validation Set...")
    val_probs = clf.predict_proba(X_val)
    val_preds = clf.predict(X_val)

    # Calculate Metrics
    # Note: y_val is integer encoded, val_probs is (n_samples, n_classes)
    loss = log_loss(y_val, val_probs)
    acc = accuracy_score(y_val, clf.classes_[np.argmax(val_probs, axis=1)])

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Log Loss: {loss:.4f}")

    # 4. Generate Predictions for Test Set
    print("\nGenerating Test Predictions...")
    test_probs = clf.predict_proba(X_test)

    # 5. Create Submission File
    print("Creating submission file...")

    # Clip probabilities to avoid log loss extremes (as per task description)
    # max(min(p, 1-10^-15), 10^-15)
    test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)

    # Create DataFrame
    # Columns must be: id, <class_names...>
    submission_df = pd.DataFrame(test_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Ensure output directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Save
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Verify submission format
    assert submission_df.shape == (
        99,
        100,
    ), f"Submission shape mismatch: {submission_df.shape}"
    assert "id" in submission_df.columns
    assert not submission_df.isnull().values.any(), "Submission contains NaNs"


if __name__ == "__main__":
    set_seed(SEED)

    # Execute demonstrations
    demo_feature_extraction()
    demo_preprocessor()

    # Execute main pipeline
    run_pipeline()

    print("\nAll tasks completed successfully.")
