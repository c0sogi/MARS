import os
import numpy as np
import pandas as pd
import cv2

# Import from the provided library
from library.config import INPUT_DIR, TRAIN_META_PATH, SUBMISSION_PATH, SEED
from library.utils import set_seed
from library.features import extract_geometric_features
from library.data import get_data
from library.preprocessing import get_preprocessed_data, HighPrecisionTransformer
from library.model import train_model, generate_submission, OASDiscriminant


def run_demo():
    print("--- Starting Library Demo ---")

    # 1. Setup and Reproducibility
    print("\n[1] Setting Random Seed")
    set_seed(SEED)
    print(f"Seed set to {SEED}")

    # 2. Demonstrate Geometric Feature Extraction
    print("\n[2] Testing Feature Extraction on a Single Image")
    # Pick a sample image from the metadata
    if os.path.exists(TRAIN_META_PATH):
        df_meta = pd.read_csv(TRAIN_META_PATH)
        sample_row = df_meta.iloc[0]
        sample_img_path = os.path.join(INPUT_DIR, sample_row["file_path"])

        print(f"Extracting features from: {sample_img_path}")
        features = extract_geometric_features(sample_img_path)

        # Validation
        expected_keys = [
            "Area",
            "Mean_Thickness",
            "Eccentricity",
            "Solidity",
            "Extent",
            "Aspect_Ratio",
        ]
        assert all(k in features for k in expected_keys), "Missing geometric features."
        print(f"Extracted Features: {features}")
    else:
        print("Metadata not found, skipping single image test.")

    # 3. Data Loading
    print("\n[3] Loading Dataset")
    # We use debug_size=None (Full Data) because OASDiscriminant requires
    # at least one sample per class to compute means.
    # Since the dataset is small (~700 train), this is fast.
    data = get_data(load_cached_data=True, debug_size=None)

    X_train_raw = data["X_train"]
    y_train = data["y_train"]

    print(f"Training Data Shape: {X_train_raw.shape}")
    print(f"Training Labels Shape: {y_train.shape}")

    # Validate data types
    assert X_train_raw.dtypes.apply(
        lambda x: np.issubdtype(x, np.number)
    ).all(), "Non-numeric features detected."
    assert not X_train_raw.isnull().values.any(), "NaNs found in raw data."

    # 4. Preprocessing
    print("\n[4] Preprocessing Data (Sanitization -> PowerTransform -> Scaling)")
    # This handles caching automatically
    processed_data = get_preprocessed_data(load_cached_data=True, debug_size=None)

    X_train = processed_data["X_train"]
    X_val = processed_data["X_val"]
    y_val = processed_data["y_val"]

    # Validate Preprocessing
    print(f"Processed Train Shape: {X_train.shape}")
    assert (
        X_train.dtype == np.float64
    ), "Preprocessing did not enforce float64 precision."

    # Check standardization (Mean ~ 0, Std ~ 1)
    # Note: PowerTransformer output isn't perfectly 0/1 but close enough for linear models
    mean_val = np.mean(X_train)
    std_val = np.std(X_train)
    print(f"Global Mean: {mean_val:.4f}, Global Std: {std_val:.4f}")
    assert abs(mean_val) < 0.1, "Data not centered."

    # 5. Model Training and Evaluation
    print("\n[5] Training OAS Discriminant Model")

    # Train and evaluate
    model = train_model(X_train, y_train, X_val, y_val)

    # Validate Model Properties
    assert isinstance(model, OASDiscriminant)
    assert hasattr(model, "W_"), "Model weights not initialized."
    assert hasattr(model, "b_"), "Model bias not initialized."

    # Verify Probabilities
    print("Verifying probability outputs...")
    probs = model.predict_proba(X_val[:5])
    row_sums = np.sum(probs, axis=1)
    print(f"First 5 row sums: {row_sums}")
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1."
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of bounds."

    # 6. Generate Submission
    print("\n[6] Generating Submission")
    X_test = processed_data["X_test"]
    test_ids = processed_data["test_ids"]

    generate_submission(model, X_test, test_ids)

    # Validate Submission File
    if os.path.exists(SUBMISSION_PATH):
        df_sub = pd.read_csv(SUBMISSION_PATH)
        print(f"Submission file created at {SUBMISSION_PATH}")
        print(f"Submission Shape: {df_sub.shape}")

        # Check constraints
        assert df_sub.shape[0] == len(test_ids), "Submission row count mismatch."
        assert (
            df_sub.shape[1] == len(model.classes_) + 1
        ), "Submission column count mismatch (Classes + ID)."
        assert "id" in df_sub.columns, "ID column missing."

        # Check values
        numeric_cols = df_sub.columns.drop("id")
        assert df_sub[numeric_cols].min().min() >= 0, "Negative probabilities found."
        assert df_sub[numeric_cols].max().max() <= 1, "Probabilities > 1 found."
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
