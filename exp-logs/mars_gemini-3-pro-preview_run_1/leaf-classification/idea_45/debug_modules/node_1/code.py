import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

# Import provided library modules
import library.config as config
import library.utils as utils
import library.features as features
import library.data as data
import library.preprocessing as preprocessing
import library.model as model


def run_demo():
    print("=== Starting Model Pipeline Demo ===\n")

    # 1. Setup & Initialization
    utils.set_seed(42)
    print("Random seed set to 42.")

    # 2. Demonstrate Feature Extraction (Single Image)
    print("\n[Step 1] Demonstrating Feature Extraction...")

    # Locate a sample image using metadata
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA_PATH}")

    df_train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_row = df_train_meta.iloc[0]
    sample_img_rel_path = sample_row[config.FILE_PATH_COL]
    sample_img_full_path = os.path.join(config.INPUT_DIR, sample_img_rel_path)

    print(f"Extracting features from image: {sample_img_full_path}")
    img_feats = features.extract_features_from_image(sample_img_full_path)

    # Validation: Check keys and types
    expected_keys = set(config.IMAGE_FEATURE_COLS)
    extracted_keys = set(img_feats.keys())

    # Check for missing or extra keys
    assert (
        expected_keys == extracted_keys
    ), f"Feature keys mismatch. \nExpected: {expected_keys}\nGot: {extracted_keys}"

    # Check value types
    assert all(
        isinstance(v, float) for v in img_feats.values()
    ), "All extracted features must be floats."
    print("Feature extraction successful. Keys and types verified.")

    # 3. Load Data & Pipeline
    print("\n[Step 2] Loading and Processing Dataset...")
    # We load the full dataset (debug_sample_size=None) to ensure LabelEncoder sees all classes.
    # The pipeline in library.data handles:
    #   - Feature extraction (cached or computed)
    #   - Merging tabular and image features
    #   - Inductive Preprocessing (Yeo-Johnson + Scaling)
    #   - Label Encoding
    dataset = data.load_data(debug_sample_size=None, load_cached_data=True)

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_val = dataset["X_val"]
    y_val = dataset["y_val"]
    X_test = dataset["X_test"]
    test_ids = dataset["test_ids"]
    le = dataset["label_encoder"]

    print(f"Data Loaded:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    # Validation: Data Integrity
    assert (
        X_train.dtype == config.FLOAT_PRECISION
    ), "X_train must be float64 for high precision."
    assert not np.isnan(X_train).any(), "X_train contains NaNs."
    assert not np.isnan(X_val).any(), "X_val contains NaNs."

    # Verify feature count matches config (Tabular + Image)
    expected_feat_count = len(config.TABULAR_FEATURE_COLS) + len(
        config.IMAGE_FEATURE_COLS
    )
    assert (
        X_train.shape[1] == expected_feat_count
    ), f"Expected {expected_feat_count} features, got {X_train.shape[1]}."
    print("Data integrity checks passed.")

    # 4. Demonstrate Preprocessor Class Explicitly
    print("\n[Step 3] Demonstrating HighPrecisionPreprocessor API...")
    # While load_data uses this internally, we demonstrate independent usage here.
    dummy_X = np.random.rand(50, 5).astype(np.float64)
    # Add some variance to make Yeo-Johnson work meaningfully
    dummy_X[:, 0] = dummy_X[:, 0] * 10

    pp = preprocessing.HighPrecisionPreprocessor()
    dummy_transformed = pp.fit_transform(dummy_X)

    assert dummy_transformed.dtype == config.FLOAT_PRECISION
    assert dummy_transformed.shape == dummy_X.shape
    # Check if centered (mean approx 0)
    assert np.allclose(
        dummy_transformed.mean(axis=0), 0, atol=1e-6
    ), "Transformed data should be centered."
    print("Preprocessor API demonstration successful.")

    # 5. Model Training (OAS Discriminant)
    print("\n[Step 4] Training OAS Discriminant Model...")
    clf = model.OASDiscriminant()
    clf.fit(X_train, y_train)

    print(f"Model fitted on {len(clf.classes_)} classes.")

    # Validation: Model Attributes
    assert hasattr(clf, "precision_"), "Model should have estimated precision matrix."
    assert (
        clf.precision_.dtype == config.FLOAT_PRECISION
    ), "Precision matrix must be float64."

    # 6. Evaluation
    print("\n[Step 5] Evaluating on Validation Set...")
    val_probs = clf.predict_proba(X_val)
    val_preds = clf.predict(X_val)

    # Validation: Probabilities
    assert val_probs.shape == (X_val.shape[0], len(clf.classes_))
    # Ensure probabilities sum to 1 (with small floating point tolerance)
    row_sums = val_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities must sum to 1.0."

    # Metrics
    acc = accuracy_score(y_val, val_preds)
    loss = log_loss(y_val, val_probs, labels=clf.classes_)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Log Loss: {loss:.4f}")

    # 7. Generate Submission
    print("\n[Step 6] Generating Submission for Test Set...")
    test_probs = clf.predict_proba(X_test)

    # Get class names from LabelEncoder
    class_names = le.inverse_transform(clf.classes_)

    # Create DataFrame
    submission_df = pd.DataFrame(test_probs, columns=class_names)
    submission_df.insert(0, "id", test_ids)

    # Check output format
    assert "id" in submission_df.columns
    assert len(submission_df) == len(X_test)
    assert len(submission_df.columns) == len(class_names) + 1

    # Save
    output_path = os.path.join("./working", "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
