import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Import provided library modules
from library import config
from library import data_loader
from library import preprocessing
from library import factorized_lda

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demo_data_loader():
    print("\n--- 1. Demonstrating Data Loader ---")
    debug_size = 50

    # Test loading datasets with debug slicing
    print(f"Loading datasets with debug_size={debug_size}...")
    df_train, df_val, df_test = data_loader.load_datasets(
        load_cached_data=False,  # Force reload from CSVs to test logic
        debug_size=debug_size,
    )

    # Validations
    print("Validating data loader output...")
    assert isinstance(df_train, pd.DataFrame), "Train set is not a DataFrame"
    assert (
        len(df_train) == debug_size
    ), f"Train size mismatch: expected {debug_size}, got {len(df_train)}"
    assert (
        len(df_val) == debug_size
    ), f"Val size mismatch: expected {debug_size}, got {len(df_val)}"
    assert config.TARGET_COLUMN in df_train.columns, "Target column missing in train"

    # Test feature splitting
    print("Testing feature splitting by group...")
    feature_groups = data_loader.split_features_by_group(df_train)

    expected_groups = config.FEATURE_PREFIXES  # ['margin', 'shape', 'texture']
    for group in expected_groups:
        assert group in feature_groups, f"Missing feature group: {group}"
        # Each feature group has 64 attributes
        assert (
            feature_groups[group].shape[1] == 64
        ), f"Incorrect feature count for {group}"
        assert (
            feature_groups[group].shape[0] == debug_size
        ), f"Incorrect row count for {group}"

    print("Data Loader demonstration successful.")
    return df_train


def demo_preprocessing(df_train):
    print("\n--- 2. Demonstrating Preprocessing ---")

    # Instantiate Preprocessor
    preprocessor = preprocessing.GroupWisePreprocessor()

    # Fit on training data
    print("Fitting preprocessor...")
    preprocessor.fit(df_train)

    # Check internal state
    for group in config.FEATURE_PREFIXES:
        assert group in preprocessor.transformers, f"Transformer missing for {group}"
        assert group in preprocessor.scalers, f"Scaler missing for {group}"

    # Transform data
    print("Transforming training data...")
    X_train_dict = preprocessor.transform(df_train)

    # Validate transformed data
    for group in config.FEATURE_PREFIXES:
        data = X_train_dict[group]
        assert isinstance(
            data, np.ndarray
        ), f"Transformed data for {group} is not numpy array"
        assert data.dtype == config.FLOAT_PRECISION, f"Precision mismatch for {group}"

        # Check standardization (Mean ~ 0, Std ~ 1)
        # Note: With N=50, stats might fluctuate, but should be reasonably close
        mean = np.mean(data)
        std = np.std(data)
        print(f"  Group {group}: Mean={mean:.4f}, Std={std:.4f}")
        assert (
            abs(mean) < 1e-1
        ), f"Standardization failed for {group} (Mean not close to 0)"
        assert (
            abs(std - 1.0) < 0.2
        ), f"Standardization failed for {group} (Std not close to 1)"

    # Test the orchestration function
    print("Testing get_preprocessed_data orchestration...")
    # We use load_cached_data=False to force the pipeline to run
    X_train, y_train, X_val, y_val, X_test, ids_test = (
        preprocessing.get_preprocessed_data(load_cached_data=False, debug_size=50)
    )

    assert len(y_train) == 50
    assert isinstance(X_train, dict)

    print("Preprocessing demonstration successful.")
    return X_train, y_train, X_val, y_val


def demo_model_training(X_train, y_train, X_val, y_val):
    print("\n--- 3. Demonstrating Factorized OAS LDA Model ---")

    model = factorized_lda.FactorizedOASLDA()

    # Fit model
    print("Fitting model...")
    model.fit(X_train, y_train)

    # Check if model learned classes
    n_classes = len(np.unique(y_train))
    assert len(model.classes_) == n_classes, "Model class count mismatch"

    # Predict on validation
    print("Predicting probabilities on validation set...")
    probs = model.predict_proba(X_val)

    # Validate predictions
    assert probs.shape == (
        len(y_val),
        n_classes,
    ), f"Probability shape mismatch: {probs.shape}"

    # Check if probabilities sum to 1
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Check range
    assert probs.min() >= 0 and probs.max() <= 1.0, "Probabilities out of range [0, 1]"

    print(f"Prediction shape: {probs.shape}")
    print("Model training demonstration successful.")


def demo_full_pipeline():
    print("\n--- 4. Demonstrating Full Pipeline Execution ---")

    # Clean up previous submission if exists
    if os.path.exists(config.SUBMISSION_PATH):
        os.remove(config.SUBMISSION_PATH)

    # Run the main execution function provided in the library
    # This handles loading, training, validation, and submission generation
    factorized_lda.run_training_and_submission(debug_size=20)

    # Verify submission file creation
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {config.SUBMISSION_PATH}"
        )

    # Verify submission content
    print("Verifying submission file content...")
    sub_df = pd.read_csv(config.SUBMISSION_PATH)

    # Check ID column
    assert config.ID_COLUMN in sub_df.columns, "ID column missing in submission"

    # Check shape (20 rows for debug_size=20)
    assert (
        len(sub_df) == 20
    ), f"Submission row count mismatch. Expected 20, got {len(sub_df)}"

    # Check probability columns (should match number of classes in training)
    # We exclude the ID column
    prob_cols = [c for c in sub_df.columns if c != config.ID_COLUMN]
    assert len(prob_cols) > 0, "No probability columns found"

    # Verify values are numeric
    assert pd.api.types.is_numeric_dtype(
        sub_df[prob_cols[0]]
    ), "Probability columns are not numeric"

    print("Full pipeline demonstration successful.")


if __name__ == "__main__":
    set_seed(42)

    print("Starting Library Usage Demonstration...")
    print("=" * 40)

    try:
        # 1. Data Loader
        df_train_sample = demo_data_loader()

        # 2. Preprocessing
        X_train, y_train, X_val, y_val = demo_preprocessing(df_train_sample)

        # 3. Model Training
        demo_model_training(X_train, y_train, X_val, y_val)

        # 4. Full Pipeline
        demo_full_pipeline()

        print("\n" + "=" * 40)
        print("ALL DEMONSTRATIONS PASSED SUCCESSFULLY")

    except AssertionError as e:
        print(f"\nFAILED: Assertion Error - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILED: Unexpected Error - {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
