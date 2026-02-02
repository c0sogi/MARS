import os
import sys
import numpy as np
import pandas as pd
import shutil
from sklearn.utils.validation import check_is_fitted

# Import the provided library modules
import library.config as config
import library.image_features as image_features
import library.data_manager as data_manager
import library.models as models
import library.engine as engine

# Set random seeds for reproducibility
np.random.seed(42)


def demo_image_features():
    print("\n=== Demo: Image Feature Extraction ===")

    # 1. Test Single Image Extraction
    # Pick an image from the metadata to ensure it exists
    train_df = pd.read_csv(config.TRAIN_CSV)
    sample_image_path = train_df.iloc[0]["image_path"]
    print(f"Extracting features from: {sample_image_path}")

    features = image_features.extract_single_image_features(sample_image_path)

    print("Extracted Features:", features)

    # Verification
    assert isinstance(features, dict), "Output should be a dictionary"
    for key in config.META_FEATURES:
        assert key in features, f"Missing key {key} in extracted features"
        assert isinstance(features[key], float), f"Feature {key} should be a float"

    print("Single image extraction verified.")

    # 2. Test Augmented Dataset Creation (Caching)
    # We will force re-computation by setting load_cached_data=False
    # Note: This processes the whole train set defined in config.TRAIN_CSV
    # Since the dataset is small (~700 images), this is acceptable for a demo.
    print("Testing augmented dataset generation (this may take a few seconds)...")

    # We'll use a temporary cache path to avoid messing with the main workflow's cache if needed,
    # but for this demo, we can just use the one defined in config or a temp one.
    # Let's use a temp one to prove it works.
    temp_cache_path = os.path.join(config.WORKING_DIR, "demo_train_augmented.parquet")
    if os.path.exists(temp_cache_path):
        os.remove(temp_cache_path)

    df_aug = image_features.get_augmented_dataset(
        config.TRAIN_CSV, temp_cache_path, load_cached_data=False
    )

    # Verification
    assert os.path.exists(temp_cache_path), "Cache file was not created"
    expected_cols = len(train_df.columns) + len(config.META_FEATURES)
    assert (
        df_aug.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, got {df_aug.shape[1]}"
    assert df_aug.shape[0] == len(train_df), "Row count mismatch"

    print("Augmented dataset generation verified.")


def demo_data_manager():
    print("\n=== Demo: Data Manager ===")

    # 1. Load Train Data
    print("Loading Training Data...")
    X_train, y_train, ids_train = data_manager.load_dataset(
        "train", load_cached_data=True
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Verification
    assert not X_train.empty, "X_train is empty"
    assert len(X_train) == len(y_train), "Mismatch between X and y length"
    assert "aspect_ratio" in X_train.columns, "Meta-features missing from X_train"

    # 2. Load Combined Data
    print("Loading Combined Train+Val Data...")
    X_comb, y_comb, ids_comb = data_manager.load_combined_train_val(
        load_cached_data=True
    )

    print(f"Combined shape: {X_comb.shape}")

    # Verification
    # Train (approx 80%) + Val (approx 20%)
    assert len(X_comb) > len(X_train), "Combined set should be larger than train set"

    print("Data loading verified.")
    return X_train, y_train


def demo_models(X_sample, y_sample):
    print("\n=== Demo: Model Instantiation and Training ===")

    # OPTIMIZATION: Patch parameters for speed
    print("Patching model parameters for fast execution...")

    # Reduce Grid Search for Logistic Regression
    # Original: np.logspace(-3, 5, 50), cv=3
    models.LOGISTIC_REGRESSION_PARAMS["Cs"] = [1.0]  # Single value
    models.LOGISTIC_REGRESSION_PARAMS["cv"] = 2  # Fewer folds
    models.LOGISTIC_REGRESSION_PARAMS["max_iter"] = 100  # Lower iter for demo

    # Reduce Kernel Approximation complexity
    # Original: n_components=400
    models.KERNEL_NYSTROEM_PARAMS["n_components"] = 50

    # 1. Linear Branch
    print("Testing Linear Branch...")
    clf_linear = models.get_linear_branch()
    # Use full sample (712 rows) to ensure all classes are present for CV
    # Subsampling to 50 rows for 99 classes caused CV failure (Cite debug_lesson_2)
    clf_linear.fit(X_sample, y_sample)
    check_is_fitted(clf_linear)
    probs = clf_linear.predict_proba(X_sample)

    assert probs.shape == (
        len(X_sample),
        len(clf_linear.classes_),
    ), "Output probability shape mismatch"
    print("Linear Branch verified.")

    # 2. Generative Branch
    print("Testing Generative Branch...")
    clf_gen = models.get_generative_branch()
    clf_gen.fit(X_sample, y_sample)
    check_is_fitted(clf_gen)
    probs_gen = clf_gen.predict_proba(X_sample)

    assert probs_gen.shape == (
        len(X_sample),
        len(clf_gen.classes_),
    ), "Output probability shape mismatch"
    print("Generative Branch verified.")

    # 3. Kernel Branch
    print("Testing Kernel Branch...")
    clf_kernel = models.get_kernel_branch()
    clf_kernel.fit(X_sample, y_sample)
    # Pipeline doesn't have check_is_fitted on itself easily, check the final estimator
    check_is_fitted(clf_kernel.named_steps["classifier"])
    probs_kernel = clf_kernel.predict_proba(X_sample)

    assert probs_kernel.shape == (
        len(X_sample),
        len(clf_kernel.classes_),
    ), "Output probability shape mismatch"
    print("Kernel Branch verified.")


def demo_engine():
    print("\n=== Demo: Full Engine Execution ===")

    # The parameters are already patched in `demo_models` because we modified the dictionary objects
    # in the `models` module which `engine` also imports.

    print("Running train_and_predict_ensemble...")
    # We use load_cached_data=True to leverage the files created in demo_image_features/demo_data_manager
    engine.train_and_predict_ensemble(load_cached_data=True)

    # Verification
    submission_path = config.SUBMISSION_FILE
    print(f"Checking submission file at: {submission_path}")

    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")

    # Check format
    assert "id" in df_sub.columns, "id column missing in submission"
    # Check if probabilities are valid
    feature_cols = [c for c in df_sub.columns if c != "id"]
    probs = df_sub[feature_cols].values

    assert np.all(probs >= 0) and np.all(
        probs <= 1
    ), "Probabilities out of range [0, 1]"
    # Note: The problem statement says probabilities are rescaled by row sum,
    # but our code output should ideally be somewhat normalized or at least valid.
    # The code clips them to [eps, 1-eps].

    print("Engine execution verified.")


if __name__ == "__main__":
    print("Starting Library Demo...")

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    try:
        # Step 1: Image Features
        demo_image_features()

        # Step 2: Data Manager
        X_train, y_train = demo_data_manager()

        # Step 3: Models (with parameter patching)
        demo_models(X_train, y_train)

        # Step 4: Engine
        demo_engine()

        print("\nAll demonstrations completed successfully!")

    except AssertionError as e:
        print(f"\nDEMO FAILED: Assertion Error - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nDEMO FAILED: An unexpected error occurred - {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
