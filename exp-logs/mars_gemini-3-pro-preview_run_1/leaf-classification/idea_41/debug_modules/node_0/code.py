import os
import sys
import numpy as np
import pandas as pd
import warnings

# Ensure local imports work
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import (
    setup_directories,
    TRAIN_CSV,
    INPUT_DIR,
    GEOMETRIC_FEATURES,
    SUBMISSION_DIR,
    PRECISION_TYPE,
)
from library.utils import set_seed
from library.feature_engineering import extract_geometric_properties
from library.data_loader import load_and_augment_data
from library.preprocessing import preprocess_features
from library.model import LinearOASDiscriminant, train_and_evaluate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    print("=" * 50)
    print("STARTING LIBRARY DEMONSTRATION")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Environment Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up environment...")
    set_seed(42)
    setup_directories()
    print("Directories validated and random seed set.")

    # ---------------------------------------------------------
    # 2. Feature Engineering Demo
    # ---------------------------------------------------------
    print("\n[2] Demonstrating Feature Engineering (Single Image)...")

    # Load metadata to find a valid image
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Metadata file {TRAIN_CSV} not found.")

    df_train = pd.read_csv(TRAIN_CSV)
    # Pick the first image
    sample_row = df_train.iloc[0]
    # Construct full path. Metadata contains relative path like 'images/123.jpg'
    image_path = os.path.join(INPUT_DIR, sample_row["file_path"])

    print(f"Processing image: {image_path}")

    # Extract features
    features = extract_geometric_properties(image_path)

    # Validate Output
    if not isinstance(features, dict):
        raise TypeError("extract_geometric_properties should return a dictionary.")

    # Check if all expected keys are present
    missing_keys = [k for k in GEOMETRIC_FEATURES if k not in features]
    if missing_keys:
        raise AssertionError(f"Missing geometric features: {missing_keys}")

    # Check value types
    if not all(isinstance(v, float) for v in features.values()):
        raise ValueError("All geometric feature values must be floats.")

    print(f"Successfully extracted {len(features)} geometric features.")
    print(
        f"Example - Area: {features['Area']:.4f}, Convexity: {features['Convexity']:.4f}"
    )

    # ---------------------------------------------------------
    # 3. Data Loading & Augmentation Demo
    # ---------------------------------------------------------
    print("\n[3] Demonstrating Data Loading & Augmentation...")
    print("Loading data (using cache if available)...")

    # This function handles loading CSVs, extracting features for all images,
    # and caching the result as .npy files.
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_and_augment_data(
        load_cached_data=True
    )

    # Validate Data Shapes and Types
    print(f"Train Shape: {X_train.shape}")
    print(f"Val Shape:   {X_val.shape}")
    print(f"Test Shape:  {X_test.shape}")

    if X_train.dtype != PRECISION_TYPE:
        raise ValueError(
            f"X_train dtype mismatch. Expected {PRECISION_TYPE}, got {X_train.dtype}"
        )

    if len(X_train) != len(y_train):
        raise AssertionError("Mismatch between X_train samples and y_train labels.")

    if len(classes) != len(np.unique(y_train)):
        # Note: This might happen if train split misses a class, but with stratified split it shouldn't.
        # We'll just warn or check strict equality based on expectation.
        print(
            f"Note: {len(classes)} classes defined, {len(np.unique(y_train))} present in train."
        )

    # ---------------------------------------------------------
    # 4. Preprocessing Demo
    # ---------------------------------------------------------
    print("\n[4] Demonstrating High-Precision Preprocessing...")
    print("Applying Yeo-Johnson Power Transformation + Standard Scaling...")

    X_train_trans, X_val_trans, X_test_trans = preprocess_features(
        X_train, X_val, X_test, load_cached_data=True
    )

    # Validate Transformation
    if X_train_trans.shape != X_train.shape:
        raise AssertionError("Shape mismatch after preprocessing.")

    if np.isnan(X_train_trans).any():
        raise ValueError("NaN values found in preprocessed training data.")

    # Check statistics on a subset (first feature)
    mean_val = np.mean(X_train_trans[:, 0])
    std_val = np.std(X_train_trans[:, 0])
    print(
        f"Feature 0 Stats after transform -> Mean: {mean_val:.4f}, Std: {std_val:.4f}"
    )
    # We expect mean ~ 0 and std ~ 1
    if abs(mean_val) > 1e-6 or abs(std_val - 1.0) > 1e-6:
        print(
            "Note: Feature 0 is not perfectly 0 mean/1 std (expected if constant or near-constant)."
        )

    # ---------------------------------------------------------
    # 5. Model Training & Inference Demo
    # ---------------------------------------------------------
    print("\n[5] Demonstrating Linear OAS Discriminant Model...")

    model = LinearOASDiscriminant()
    model.fit(X_train_trans, y_train)
    print("Model fitted successfully.")

    # Predict on Validation set
    val_probs = model.predict_proba(X_val_trans)

    # Validate Probabilities
    if val_probs.shape != (len(X_val), len(classes)):
        raise AssertionError(
            f"Probability output shape mismatch. Got {val_probs.shape}"
        )

    # Check if probabilities sum to 1
    row_sums = np.sum(val_probs, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        raise ValueError("Probabilities do not sum to 1.")

    print("Inference successful. Probabilities valid.")

    # ---------------------------------------------------------
    # 6. Full Pipeline Execution
    # ---------------------------------------------------------
    print("\n[6] Running Full Pipeline (train_and_evaluate)...")
    # This function orchestrates the entire flow and generates the submission file
    val_log_loss, val_acc = train_and_evaluate(load_cached_data=True)

    print(f"Pipeline Completed.")
    print(f"Validation Log Loss: {val_log_loss:.5f}")
    print(f"Validation Accuracy: {val_acc:.5f}")

    # ---------------------------------------------------------
    # 7. Submission Verification
    # ---------------------------------------------------------
    print("\n[7] Verifying Submission File...")
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not generated.")

    sub_df = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    # Check required columns
    if "id" not in sub_df.columns:
        raise ValueError("Submission missing 'id' column.")

    # Check if all classes are present as columns
    missing_classes = [c for c in classes if c not in sub_df.columns]
    if missing_classes:
        raise ValueError(f"Submission missing class columns: {missing_classes[:5]}...")

    # Check row count matches test set
    if len(sub_df) != len(test_ids):
        raise AssertionError(
            f"Submission row count {len(sub_df)} does not match test set size {len(test_ids)}."
        )

    print("Submission file format verified.")
    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    run_demonstration()
