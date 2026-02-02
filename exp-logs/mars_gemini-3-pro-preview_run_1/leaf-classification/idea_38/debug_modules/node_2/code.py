import os
import sys
import numpy as np
import pandas as pd
import warnings

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import (
    TRAIN_CSV,
    SUBMISSION_PATH,
    SEED,
    INPUT_DIR,
    GEOMETRIC_FEATURES,
)
from library.feature_extraction import extract_geometric_features, process_dataset
from library.data_loader import get_data_loaders
from library.preprocessing import get_preprocessed_data, Float64Pipeline
from library.model import OASLinearDiscriminant, train_and_predict


def demo_feature_extraction():
    print("\n=== Demo: Feature Extraction ===")

    # 1. Test single image extraction
    # Load metadata to find a valid image path
    df_train = pd.read_csv(TRAIN_CSV)
    sample_row = df_train.iloc[0]
    image_rel_path = sample_row["file_path"]
    full_image_path = os.path.join(INPUT_DIR, image_rel_path)

    print(f"Extracting features from: {full_image_path}")
    features = extract_geometric_features(full_image_path)

    # Validation
    assert isinstance(features, dict), "Output must be a dictionary"
    for key in GEOMETRIC_FEATURES:
        assert key in features, f"Missing feature: {key}"
        assert isinstance(
            features[key], (float, np.floating)
        ), f"Feature {key} is not a float"

    print("Single image feature extraction successful.")
    print(
        f"Sample features: {list(features.keys())[:3]} -> {list(features.values())[:3]}"
    )

    # 2. Test batch processing via process_dataset
    limit = 10
    print(f"Processing dataset with limit={limit}...")
    df_features = process_dataset(
        TRAIN_CSV, "train", load_cached_data=False, limit=limit
    )

    # Validation
    assert isinstance(df_features, pd.DataFrame), "Output must be a DataFrame"
    assert len(df_features) == limit, f"Expected {limit} rows, got {len(df_features)}"
    assert df_features.index.name == "id", "Index must be 'id'"

    print("Batch processing successful.")


def demo_data_loading():
    print("\n=== Demo: Data Loading ===")
    limit = 20

    # Load data using the main loader
    (train_data, val_data, test_data) = get_data_loaders(
        load_cached_data=False, limit=limit
    )

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, y_test, ids_test = test_data

    # Validation
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}, ids={ids_train.shape}")

    # Check X
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert len(X_train) == limit, "Incorrect number of training samples"

    # Check y
    assert isinstance(y_train, np.ndarray), "y_train should be a numpy array"
    assert len(y_train) == limit, "y_train length mismatch"

    # Check ids
    assert isinstance(ids_train, np.ndarray), "ids_train should be a numpy array"

    # Check Test set (y should be None)
    assert y_test is None, "Test set labels should be None"
    assert len(X_test) == limit, "Incorrect number of test samples"

    print("Data loading successful.")


def demo_preprocessing():
    print("\n=== Demo: Preprocessing ===")
    limit = 20

    # Get raw data first to demonstrate the pipeline manually
    (train_data, _, _) = get_data_loaders(load_cached_data=False, limit=limit)
    X_train_raw, _, _ = train_data

    # 1. Manual Pipeline Usage
    pipeline = Float64Pipeline()
    print("Fitting Float64Pipeline...")
    pipeline.fit(X_train_raw)
    X_transformed = pipeline.transform(X_train_raw)

    # Validation
    assert isinstance(
        X_transformed, np.ndarray
    ), "Transformed data should be numpy array"
    assert X_transformed.dtype == np.float64, "Transformed data should be float64"
    assert not np.isnan(X_transformed).any(), "Transformed data contains NaNs"
    assert (
        X_transformed.shape == X_train_raw.shape
    ), "Shape mismatch after transformation"

    # 2. Integrated Preprocessing Loader
    print("Testing get_preprocessed_data...")
    (train_p, val_p, test_p) = get_preprocessed_data(
        load_cached_data=False, limit=limit
    )

    X_train_p, y_train_p, _ = train_p

    # Validation
    assert X_train_p.shape[0] == limit
    assert isinstance(X_train_p, np.ndarray)

    print("Preprocessing successful.")


def demo_model_training():
    print("\n=== Demo: Model Training (OASLinearDiscriminant) ===")
    limit = 50

    # Get data
    (train_data, val_data, _) = get_preprocessed_data(
        load_cached_data=False, limit=limit
    )
    X_train, y_train, _ = train_data
    X_val, _, _ = val_data

    # Instantiate Model
    model = OASLinearDiscriminant()

    # Fit
    print("Fitting model...")
    model.fit(X_train, y_train)

    # Validation of internal state
    assert model.classes_ is not None, "Model classes not set"
    assert model.W_ is not None, "Weight matrix not calculated"
    assert model.b_ is not None, "Bias vector not calculated"

    # Predict
    print("Predicting on validation set...")
    probs = model.predict_proba(X_val)
    preds = model.predict(X_val)

    # Validation of outputs
    assert probs.shape == (
        len(X_val),
        len(model.classes_),
    ), "Probability shape mismatch"
    assert preds.shape == (len(X_val),), "Prediction shape mismatch"

    # Check probability properties
    # Note: Due to floating point arithmetic, sum might be slightly off 1.0, but should be very close
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Check clipping (no exact 0 or 1 unless clipped epsilon is 0)
    assert np.all(probs > 0.0), "Probabilities contain zeros"
    assert np.all(probs < 1.0), "Probabilities contain ones"

    print(f"Validation Log Loss would be calculated on {len(X_val)} samples.")
    print("Model training and prediction successful.")


def demo_full_pipeline():
    print("\n=== Demo: Full Pipeline Execution ===")
    limit = 30

    # Run the main driver function
    # This handles loading, preprocessing, training, predicting, and saving submission
    loss = train_and_predict(load_cached_data=False, limit=limit)

    print(f"Pipeline finished with Validation Log Loss: {loss}")

    # Verify Submission File
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Validation
    assert len(df_sub) == limit, f"Submission should have {limit} rows"
    assert "id" in df_sub.columns, "Submission missing 'id' column"

    # Check if all probability columns are present (number of columns should be n_classes + 1 for id)
    # We can estimate n_classes from the training data used inside train_and_predict,
    # but here we just check basic structure.
    assert df_sub.shape[1] > 2, "Submission seems to lack class columns"

    # Check values
    feature_cols = [c for c in df_sub.columns if c != "id"]
    probs = df_sub[feature_cols].values
    assert np.all(probs >= 0) and np.all(
        probs <= 1
    ), "Submission probabilities out of range"

    print("Full pipeline execution successful.")


if __name__ == "__main__":
    # Set global seed
    np.random.seed(SEED)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    try:
        demo_feature_extraction()
        demo_data_loading()
        demo_preprocessing()
        demo_model_training()
        demo_full_pipeline()
        print("\nAll demonstrations completed successfully!")
    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}")
        exit(1)
    except Exception as e:
        print(f"\n[FAILED] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
