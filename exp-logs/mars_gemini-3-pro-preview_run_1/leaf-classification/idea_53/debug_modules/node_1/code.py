import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
from library import config
from library import feature_engineering
from library import data_loader
from library import model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demo_feature_engineering():
    print("\n=== Demo: Feature Engineering ===")

    # 1. Load training metadata to find a sample image
    if not os.path.exists(config.TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_DATA_PATH}")

    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    sample_row = df_train.iloc[0]

    # Construct full path (metadata contains relative path)
    rel_path = sample_row[config.FILE_PATH_COL]
    full_path = os.path.join(config.INPUT_DIR, rel_path)

    print(f"Extracting features from: {rel_path}")

    # 2. Extract Geometric Features
    features = feature_engineering.extract_geometric_features(full_path)

    # 3. Validation
    print("Extracted Features:", features)

    # Check if all expected keys are present
    assert all(
        k in features for k in config.GEOMETRIC_FEATURES
    ), "Missing geometric features in extraction output."

    # Check value types
    assert all(
        isinstance(v, (float, np.floating)) for v in features.values()
    ), "Feature values must be floats."

    print("Feature engineering verification passed.")


def demo_data_loader_and_preprocessing():
    print("\n=== Demo: Data Loader & Preprocessing ===")

    # 1. Load Dataset
    # We use load_cached_data=False to ensure we demonstrate the processing logic
    # though in practice caching is preferred for speed.
    print("Loading dataset (raw)...")
    (
        (X_train_raw, y_train, ids_train),
        (X_val_raw, y_val, ids_val),
        (X_test_raw, ids_test),
    ) = data_loader.load_dataset(load_cached_data=True)

    # Validation of raw load
    print(
        f"Train shape: {X_train_raw.shape}, Val shape: {X_val_raw.shape}, Test shape: {X_test_raw.shape}"
    )
    assert len(X_train_raw) == len(y_train) == len(ids_train)
    assert len(X_val_raw) == len(y_val) == len(ids_val)
    assert len(X_test_raw) == len(ids_test)

    # 2. Preprocess Data
    print("Preprocessing data...")
    X_train, X_val, X_test = data_loader.preprocess_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=False
    )

    # Validation of preprocessing
    # Check statistics of training data (should be roughly mean=0, std=1)
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)

    print(f"Processed Train Mean (avg across features): {np.mean(train_mean):.4f}")
    print(f"Processed Train Std (avg across features): {np.mean(train_std):.4f}")

    # Allow small tolerance due to floating point arithmetic
    assert np.allclose(
        train_mean, 0, atol=1e-5
    ), "Training data mean is not centered at 0."
    assert np.allclose(train_std, 1, atol=1e-5), "Training data std is not scaled to 1."

    print("Data loading and preprocessing verification passed.")

    return X_train, y_train, X_val, y_val, X_test, ids_test


def demo_model_usage(X_train, y_train, X_val):
    print("\n=== Demo: Model Usage (ParsimoniousOASDiscriminant) ===")

    # 1. Instantiate
    clf = model.ParsimoniousOASDiscriminant()

    # 2. Fit
    print("Fitting model...")
    clf.fit(X_train, y_train)

    assert hasattr(clf, "classes_"), "Model did not set classes_ attribute."
    assert hasattr(clf, "precision_"), "Model did not compute precision matrix."

    # 3. Predict
    print("Predicting probabilities on validation set...")
    probs = clf.predict_proba(X_val)

    # Validation
    print(f"Probabilities shape: {probs.shape}")
    assert probs.shape == (
        X_val.shape[0],
        len(clf.classes_),
    ), "Probability output shape mismatch."

    # Check if probabilities sum to 1
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1."

    print("Model verification passed.")


def demo_full_pipeline():
    print("\n=== Demo: Full Training Pipeline ===")

    # 1. Run Pipeline
    # This wraps loading, preprocessing, training, validating, and submission generation
    trained_model, val_loss = model.run_training_pipeline(load_cached_data=True)

    print(f"Pipeline finished. Validation Log Loss: {val_loss:.4f}")

    # 2. Verify Submission File
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check columns
    expected_cols = [config.ID_COL] + list(trained_model.classes_)
    assert (
        list(df_sub.columns) == expected_cols
    ), "Submission columns do not match expected classes."

    # Check ID column
    assert df_sub[config.ID_COL].dtype in [
        np.int64,
        int,
    ], "ID column should be integer."

    print("Full pipeline verification passed.")


if __name__ == "__main__":
    set_seed(42)

    try:
        # 1. Feature Engineering
        demo_feature_engineering()

        # 2. Data Loading & Preprocessing
        # We capture the processed data to use in the model demo
        X_train, y_train, X_val, y_val, X_test, ids_test = (
            demo_data_loader_and_preprocessing()
        )

        # 3. Model Usage
        demo_model_usage(X_train, y_train, X_val)

        # 4. Full Pipeline
        demo_full_pipeline()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        # Print traceback for debugging if needed
        import traceback

        traceback.print_exc()
        sys.exit(1)
