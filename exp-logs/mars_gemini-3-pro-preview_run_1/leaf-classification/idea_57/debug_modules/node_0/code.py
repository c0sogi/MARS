import os
import numpy as np
import pandas as pd
from library.config import Config, set_seed
from library.feature_extraction import extract_geometric_features
from library.data_loader import LeafDataManager
from library.preprocessor import SanitizedTransformPipeline
from library.oas_discriminant import OASLinearDiscriminant


def main():
    print("Initializing Demonstration...")
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # 1. Demonstrate Feature Extraction on a Single Image
    # ---------------------------------------------------------
    print("\n[1] Testing Geometric Feature Extraction...")
    # Load train metadata to get a valid file path for demonstration
    train_meta = pd.read_csv(Config.TRAIN_META)
    sample_row = train_meta.iloc[0]
    sample_img_path = os.path.join(Config.INPUT_DIR, sample_row["file_path"])

    print(f"Extracting features from: {sample_img_path}")
    geo_feats = extract_geometric_features(sample_img_path)

    print(f"Extracted features: {geo_feats}")

    # Verification
    assert isinstance(geo_feats, list), "Features should be a list."
    assert len(geo_feats) == 7, "Should extract exactly 7 geometric features."
    assert all(isinstance(x, float) for x in geo_feats), "All features must be floats."
    print("Feature extraction successful.")

    # ---------------------------------------------------------
    # 2. Demonstrate Data Loading (Manager & Caching)
    # ---------------------------------------------------------
    print("\n[2] Testing LeafDataManager...")
    data_manager = LeafDataManager()

    # Load a subset of training data (max_samples=100 for speed)
    # We disable cache loading to demonstrate the processing logic
    print("Loading Training Data (Subset)...")
    X_train, y_train, ids_train, feat_names = data_manager.load_data(
        subset="train", load_cached_data=False, max_samples=100
    )

    print(f"Loaded Train Shape: {X_train.shape}")
    print(f"Labels Shape: {y_train.shape}")

    # Verification
    assert X_train.shape[0] == 100, "Should have loaded 100 samples."
    assert X_train.shape[1] == len(
        feat_names
    ), "Feature matrix width must match feature names."
    assert y_train.shape[0] == 100, "Label count must match sample count."
    assert X_train.dtype == np.float64, "Data must be float64."

    # Check if geometric features are included in the feature names
    for geo_feat in Config.GEOMETRIC_FEATURES:
        assert geo_feat in feat_names, f"Feature {geo_feat} missing from loaded data."
    print("Data loading successful.")

    # ---------------------------------------------------------
    # 3. Demonstrate Preprocessing Pipeline
    # ---------------------------------------------------------
    print("\n[3] Testing SanitizedTransformPipeline...")
    pipeline = SanitizedTransformPipeline()

    # Fit on the training subset
    print("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    # Transform training data
    print("Transforming training data...")
    X_train_trans = pipeline.transform(X_train)

    print(f"Transformed Data Shape: {X_train_trans.shape}")

    # Verification
    # VarianceThreshold might drop constant columns, so width <= original width
    assert X_train_trans.shape[1] <= X_train.shape[1]
    assert X_train_trans.dtype == np.float64

    # Check statistics (StandardScaler should result in mean ~ 0 and std ~ 1)
    means = np.mean(X_train_trans, axis=0)
    stds = np.std(X_train_trans, axis=0)

    assert np.all(np.abs(means) < 1e-6), "Transformed data mean is not zero."
    assert np.all(np.abs(stds - 1.0) < 1e-6), "Transformed data std is not one."
    print("Preprocessing successful.")

    # ---------------------------------------------------------
    # 4. Demonstrate OAS Linear Discriminant Model
    # ---------------------------------------------------------
    print("\n[4] Testing OASLinearDiscriminant...")
    model = OASLinearDiscriminant()

    # Fit model
    print("Fitting model...")
    model.fit(X_train_trans, y_train)

    # Check learned attributes
    assert hasattr(model, "classes_")
    assert hasattr(model, "weights_")
    print(f"Model fitted on {len(model.classes_)} classes.")

    # Predict probabilities on training subset
    print("Predicting probabilities...")
    probs = model.predict_proba(X_train_trans)

    # Verification
    assert probs.shape == (100, len(model.classes_))
    # Check probability constraints
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities out of range."
    assert np.allclose(np.sum(probs, axis=1), 1.0), "Probabilities do not sum to 1."

    # Predict labels
    preds = model.predict(X_train_trans)
    assert preds.shape == (100,)
    print("Model inference successful.")

    # ---------------------------------------------------------
    # 5. Integration: Process Test Data and Generate Predictions
    # ---------------------------------------------------------
    print("\n[5] Integration Test: Test Set Inference...")

    # Load test data
    print("Loading Test Data (Subset)...")
    X_test, _, ids_test, _ = data_manager.load_data(
        subset="test", load_cached_data=False, max_samples=20
    )

    # Transform test data using the pipeline fitted on train
    X_test_trans = pipeline.transform(X_test)

    # Predict
    test_probs = model.predict_proba(X_test_trans)

    # Verification
    assert X_test_trans.shape[0] == 20
    assert (
        X_test_trans.shape[1] == X_train_trans.shape[1]
    ), "Feature count mismatch between train and test."
    assert test_probs.shape == (20, len(model.classes_))

    print("Integration test successful.")
    print("\nAll library components verified successfully.")


if __name__ == "__main__":
    main()
