import os
import shutil
import numpy as np
import pandas as pd
import sys

# Import library modules
# Note: We import config first to override settings for a fast demo run
import library.config as config

# 1. Configuration Override for Speed and Isolation
print(">>> Configuring environment for demonstration...")
config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples for speed
config.CACHE_DIR = "./working/demo_cache"  # Use a separate cache for this demo
config.SUBMISSION_FILE = "./working/demo_submission.csv"

# Now import the rest of the library modules which rely on config
from library import utils, data, pipeline, model, features


def run_demonstration():
    # Set seed for reproducibility
    utils.set_seed(42)

    # Ensure working directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    print("\n" + "=" * 40)
    print("STEP 1: Data Loading & Feature Extraction")
    print("=" * 40)

    # Load Training Data
    # This triggers: Metadata loading -> Geometric Feature Extraction -> Caching -> Merging
    print("Loading training data...")
    X_train, y_train, ids_train = data.load_dataset(
        split="train", load_cached_data=False
    )

    # Validation
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert (
        len(X_train) == config.DEBUG_SAMPLE_SIZE
    ), f"Expected {config.DEBUG_SAMPLE_SIZE} training samples"
    assert X_train.dtypes.apply(
        lambda x: x == np.float64
    ).all(), "All feature columns must be float64"
    assert not X_train.isnull().values.any(), "X_train should not contain NaNs"

    print(f"Train Data Shape: {X_train.shape}")
    print(
        f"Geometric Features included: {any(col in config.GEOMETRIC_FEATURES for col in X_train.columns)}"
    )

    # Load Validation Data
    print("Loading validation data...")
    X_val, y_val, ids_val = data.load_dataset(split="val", load_cached_data=False)

    assert (
        len(X_val) == config.DEBUG_SAMPLE_SIZE
    ), f"Expected {config.DEBUG_SAMPLE_SIZE} validation samples"

    print("\n" + "=" * 40)
    print("STEP 2: Preprocessing Pipeline")
    print("=" * 40)

    # Initialize Pipeline
    # Strategy: VarianceThreshold -> Yeo-Johnson -> StandardScaler
    prep_pipeline = pipeline.get_preprocessing_pipeline(
        variance_threshold=0.0, use_yeo_johnson=True, standardize=True
    )

    print("Fitting pipeline on training data...")
    # Fit on Train
    prep_pipeline.fit(X_train, y_train)

    # Transform Train and Val
    print("Transforming datasets...")
    X_train_trans = prep_pipeline.transform(X_train)
    X_val_trans = prep_pipeline.transform(X_val)

    # Validation
    assert X_train_trans.shape[0] == len(X_train)
    assert (
        X_val_trans.shape[1] == X_train_trans.shape[1]
    ), "Feature count mismatch after transformation"
    # Check if standardization worked (mean approx 0, std approx 1)
    # Note: Yeo-Johnson might shift things, but StandardScaler is last, so mean should be ~0
    mean_vals = np.mean(X_train_trans, axis=0)
    std_vals = np.std(X_train_trans, axis=0)

    print(f"Transformed Data Shape: {X_train_trans.shape}")
    print(f"Mean of first 5 features: {mean_vals[:5]}")
    print(f"Std of first 5 features: {std_vals[:5]}")

    assert np.allclose(mean_vals, 0, atol=1e-6), "Features should be centered"
    assert np.allclose(
        std_vals, 1, atol=1e-6
    ), "Features should be scaled to unit variance"

    print("\n" + "=" * 40)
    print("STEP 3: Model Training (OAS Discriminant)")
    print("=" * 40)

    # Instantiate custom OAS Discriminant model
    clf = model.OASDiscriminant()

    print("Fitting OAS Discriminant...")
    clf.fit(X_train_trans, y_train)

    # Check learned attributes
    assert hasattr(clf, "classes_"), "Model should have classes_ attribute"
    assert hasattr(clf, "W_"), "Model should have weights W_"
    assert hasattr(clf, "b_"), "Model should have bias b_"

    print(f"Number of classes: {len(clf.classes_)}")

    print("Predicting probabilities on validation set...")
    y_pred_proba = clf.predict_proba(X_val_trans)

    # Validation of probabilities
    assert y_pred_proba.shape == (len(X_val), len(clf.classes_))
    # Sum of probabilities should be 1
    row_sums = np.sum(y_pred_proba, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), "Probabilities must sum to 1"

    # Predict labels
    y_pred_labels = clf.predict(X_val_trans)
    assert len(y_pred_labels) == len(X_val)

    print("Prediction shape verified.")

    print("\n" + "=" * 40)
    print("STEP 4: Evaluation")
    print("=" * 40)

    # Filter validation set to classes known to the model (Debug Mode Safety)
    # Cite debug_lesson_1: Filter Classes, Don't Just Slice Rows.
    known_mask = y_val.isin(clf.classes_)
    if not known_mask.all():
        print(
            f"DEBUG: Dropping {(~known_mask).sum()} validation samples with classes not seen in training set."
        )
        y_val_filtered = y_val[known_mask]
        y_pred_proba_filtered = y_pred_proba[known_mask.values]
    else:
        y_val_filtered = y_val
        y_pred_proba_filtered = y_pred_proba

    if len(y_val_filtered) > 0:
        # Compute Log Loss
        # Cite debug_lesson_15: Explicitly Define Labels in Metrics
        loss = utils.compute_log_loss(
            y_val_filtered, y_pred_proba_filtered, clf.classes_
        )
        print(f"Validation Log Loss: {loss:.4f}")
    else:
        print("WARNING: No validation samples remaining after filtering.")
        loss = 0.0

    # Sanity check: Loss should be positive
    assert loss >= 0, "Log loss cannot be negative"

    print("\n" + "=" * 40)
    print("STEP 5: Inference on Test Set & Submission")
    print("=" * 40)

    # Load Test Data
    print("Loading test data...")
    X_test, _, ids_test = data.load_dataset(split="test", load_cached_data=False)

    # Transform
    X_test_trans = prep_pipeline.transform(X_test)

    # Predict
    test_probs = clf.predict_proba(X_test_trans)

    # Save Submission
    print(f"Saving submission to {config.SUBMISSION_FILE}...")
    utils.save_submission(ids_test, test_probs, clf.classes_, config.SUBMISSION_FILE)

    # Verify File
    assert os.path.exists(config.SUBMISSION_FILE), "Submission file was not created"

    df_sub = pd.read_csv(config.SUBMISSION_FILE)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Verify format
    assert "id" in df_sub.columns, "Submission must have 'id' column"
    assert len(df_sub) == len(ids_test), "Submission row count mismatch"
    assert df_sub.shape[1] == len(clf.classes_) + 1, "Submission column count mismatch"

    print("\n" + "=" * 40)
    print("STEP 6: Direct Feature Extraction Check")
    print("=" * 40)

    # Pick an image from the loaded training set metadata to test raw feature extraction
    # We need to map the ID back to a file path.
    # Since we loaded X_train via data.load_dataset, we don't have the raw metadata df handy here.
    # Let's just use the first file path from the metadata file directly.
    df_meta_train = pd.read_csv(config.TRAIN_CSV)
    first_image_rel_path = df_meta_train.iloc[0]["file_path"]
    full_image_path = os.path.join(config.INPUT_DIR, first_image_rel_path)

    print(f"Testing geometric extraction on: {full_image_path}")
    if os.path.exists(full_image_path):
        geo_feats = features.extract_geometric_features(full_image_path)
        print("Extracted Features:", list(geo_feats.keys()))

        # Verify specific keys exist
        for key in config.GEOMETRIC_FEATURES:
            assert key in geo_feats, f"Missing geometric feature: {key}"
            assert isinstance(geo_feats[key], float), f"Feature {key} should be float"

        print("Geometric feature extraction verified.")
    else:
        print(
            "Image file not found (expected in demo environment if data is present). Skipping."
        )

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    run_demonstration()
