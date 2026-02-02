import os
import sys
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.utils import set_seed, compute_log_loss
from library.geometry import GeometryExtractor
from library.data import load_and_merge_data
from library.pipeline import SanitizedTransformer
from library.model import OASDiscriminant, run_training_process
from library.config import METADATA_DIR, INPUT_DIR, WORKING_DIR, SUBMISSION_FILE


def main():
    print("=== Starting Demonstration and Validation Script ===\n")

    # 1. Setup
    set_seed(42)

    # Define a temporary metadata file for fast testing
    temp_meta_path = os.path.join(WORKING_DIR, "temp_test_meta.csv")

    # 2. Verify Geometry Extraction
    print("--- Verifying GeometryExtractor ---")
    extractor = GeometryExtractor()

    # Load actual metadata to get a valid file path
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    df_train_meta = pd.read_csv(train_meta_path)

    # Test single image extraction
    sample_row = df_train_meta.iloc[0]
    sample_img_path = os.path.join(INPUT_DIR, sample_row["file_path"])

    print(f"Extracting features from: {sample_img_path}")
    features = extractor._extract_single_image(sample_img_path)

    print(f"Extracted feature vector: {features}")
    assert isinstance(features, np.ndarray), "Output must be a numpy array"
    assert features.shape == (
        7,
    ), f"Expected 7 geometric features, got {features.shape[0]}"
    assert features.dtype == np.float64, "Features must be float64 precision"

    # Test batch processing on a small subset (5 images)
    df_subset = df_train_meta.head(5).copy()
    df_subset.to_csv(temp_meta_path, index=False)

    print("Processing subset batch...")
    df_geo_subset = extractor.process_dataset(
        df_subset, "temp_subset", load_cached_data=False
    )

    assert df_geo_subset.shape == (
        5,
        8,
    ), f"Expected (5, 8) shape (id + 7 features), got {df_geo_subset.shape}"
    assert "Area" in df_geo_subset.columns, "Geometric features missing from DataFrame"
    print("GeometryExtractor verification passed.\n")

    # 3. Verify Data Loading and Merging
    print("--- Verifying Data Loading ---")
    # We load the full training set here (it's small enough, ~700 rows)
    # This tests the integration of tabular features + geometric features
    X, y, ids = load_and_merge_data("train", load_cached_data=True)

    print(f"Loaded Data Shape: X={X.shape}, y={y.shape}, ids={ids.shape}")

    # Expected columns: 192 tabular + 7 geometric = 199 features
    # Note: If variance threshold was applied during loading it might differ,
    # but load_and_merge_data just merges raw features.
    assert X.shape[1] == 199, f"Expected 199 features, got {X.shape[1]}"
    assert X.dtypes.iloc[0] == np.float64, "Dataframe must be float64"
    assert len(y) == len(X), "Target and Features length mismatch"
    print("Data loading verification passed.\n")

    # 4. Verify Pipeline
    print("--- Verifying SanitizedTransformer ---")
    # Use a subset for speed
    X_sub = X.iloc[:50].copy()

    transformer = SanitizedTransformer()
    transformer.fit(X_sub)
    X_trans = transformer.transform(X_sub)

    print(f"Transformed Data Shape: {X_trans.shape}")
    assert isinstance(X_trans, np.ndarray), "Transformed output must be numpy array"
    assert X_trans.dtype == np.float64, "Transformed output must be float64"

    # Check scaling (roughly)
    mean_val = np.mean(X_trans)
    std_val = np.std(X_trans)
    print(f"Global Mean: {mean_val:.4f}, Global Std: {std_val:.4f}")
    # We don't assert strict 0/1 because of the subset and potential constant features,
    # but it should be reasonably bounded.

    print("SanitizedTransformer verification passed.\n")

    # 5. Verify Model (OASDiscriminant)
    print("--- Verifying OASDiscriminant ---")
    y_sub = y[:50]

    model = OASDiscriminant()
    model.fit(X_trans, y_sub)

    # Test Prediction
    probs = model.predict_proba(X_trans)
    preds = model.predict(X_trans)

    print(f"Probabilities Shape: {probs.shape}")

    # Assertions
    assert probs.shape == (50, len(np.unique(y_sub))), "Probability shape mismatch"

    # Check probability sums
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities must sum to 1"

    # Check Log Loss Calculation
    # Convert string labels to indices for the utility function
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(y_sub)
    y_indices = le.transform(y_sub)

    # Note: compute_log_loss expects full class set if y_true is indices,
    # or we need to ensure probs matches the indices.
    # The model.predict_proba returns cols sorted by class name (via LabelEncoder inside model).
    # Our external LE should match.
    loss = compute_log_loss(y_indices, probs)
    print(f"Computed Log Loss on subset: {loss:.4f}")
    assert loss > 0, "Log loss should be positive"

    print("OASDiscriminant verification passed.\n")

    # 6. Verify End-to-End Process
    print("--- Verifying Full Training Process (Debug Mode) ---")
    # This runs the pipeline provided in library/model.py
    # debug=True ensures it runs on a small slice of data for speed
    val_loss = run_training_process(load_cached_data=True, debug=True)

    print(f"Process finished with Validation Loss: {val_loss:.4f}")

    # Check if submission file was created
    assert os.path.exists(
        SUBMISSION_FILE
    ), f"Submission file not found at {SUBMISSION_FILE}"

    df_sub = pd.read_csv(SUBMISSION_FILE)
    print(f"Submission File Shape: {df_sub.shape}")
    assert "id" in df_sub.columns, "Submission missing 'id' column"

    # Clean up temporary file
    if os.path.exists(temp_meta_path):
        os.remove(temp_meta_path)

    print("\n=== All Validations Passed Successfully ===")


if __name__ == "__main__":
    main()
