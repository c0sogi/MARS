import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Import library modules
from library import config
from library import image_features
from library import data_manager
from library import model_factory


def main():
    print("Starting demonstration of Leaf Classification pipeline...")

    # =========================================================================
    # 1. OPTIMIZATION FOR DEMO SPEED
    # =========================================================================
    print("\n[1] Optimizing configuration for rapid demonstration...")
    # Monkey-patch config to reduce computational load for this demo run
    # Reduce Logistic Regression Grid Search complexity
    config.LR_CS_GRID = np.logspace(-1, 1, 2)  # Only test 2 regularization strengths
    config.LR_CV_FOLDS = 2  # Reduce CV folds from 3 to 2
    config.LR_MAX_ITER = 100  # Limit iterations for speed

    # Reduce Nystroem Kernel approximation complexity
    config.NYSTROEM_COMPONENTS = 50  # Reduce components from 400 to 50

    print(f"  - LR Grid Size: {len(config.LR_CS_GRID)}")
    print(f"  - CV Folds: {config.LR_CV_FOLDS}")
    print(f"  - Nystroem Components: {config.NYSTROEM_COMPONENTS}")

    # =========================================================================
    # 2. DEMONSTRATE IMAGE FEATURE EXTRACTION
    # =========================================================================
    print("\n[2] Demonstrating Image Feature Extraction...")

    # Load a sample from the metadata to get a valid image path
    df_train_meta = pd.read_csv(config.TRAIN_META_PATH)
    sample_row = df_train_meta.iloc[0]
    image_rel_path = sample_row["image_path"]
    full_image_path = os.path.join(config.INPUT_DIR, image_rel_path)

    print(f"  - Processing single image: {full_image_path}")

    # Test single image extraction
    props = image_features.extract_morphological_props(full_image_path)
    print(f"  - Extracted properties: {props}")

    # Validation
    expected_keys = {"aspect_ratio", "solidity", "extent", "eccentricity"}
    assert set(props.keys()) == expected_keys, "Missing keys in extracted properties"
    assert all(
        isinstance(v, float) for v in props.values()
    ), "Properties must be floats"

    # Test dataframe augmentation on a small subset
    print("  - Augmenting a small dataframe subset (5 rows)...")
    subset_df = df_train_meta.head(5).copy()
    # Force recompute to test logic (ignoring cache for this specific call)
    augmented_df = image_features.augment_dataframe(
        subset_df, load_cached_data=False, cache_name="demo_subset_augmented"
    )

    # Validation
    assert augmented_df.shape[0] == 5, "Subset row count mismatch"
    assert all(
        k in augmented_df.columns for k in expected_keys
    ), "Augmented columns missing"
    print("  - Augmentation successful.")

    # =========================================================================
    # 3. DEMONSTRATE DATA MANAGEMENT
    # =========================================================================
    print("\n[3] Demonstrating Data Loading and Preparation...")

    # Load full dataset (this handles augmentation, splitting, encoding, and caching)
    # We use load_cached_data=True to use existing artifacts if available, speeding up the run
    X_train, y_train, X_test, test_ids, classes = data_manager.load_and_prepare_data(
        load_cached_data=True
    )

    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    print(f"  - X_test shape:  {X_test.shape}")
    print(f"  - Classes count: {len(classes)}")

    # Validation
    assert len(X_train) == len(y_train), "Mismatch between X_train and y_train length"
    assert len(X_test) == len(test_ids), "Mismatch between X_test and test_ids length"
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), "Feature dimension mismatch between train and test"
    # Check if morphological features are included (192 original + 4 morphological = 196 features)
    # Note: The original dataset has 192 features (3 types * 64).
    # The metadata CSVs have these plus id, species, image_path.
    # augment_dataframe adds 4 columns.
    # data_manager drops non-feature cols.
    # So expected feature count is 192 + 4 = 196.
    assert X_train.shape[1] == 196, f"Expected 196 features, got {X_train.shape[1]}"

    print("  - Data integrity checks passed.")

    # =========================================================================
    # 4. DEMONSTRATE MODEL TRAINING (SOFT VOTING ENSEMBLE)
    # =========================================================================
    print("\n[4] Demonstrating Model Training (SoftVotingEnsemble)...")

    # Instantiate the ensemble
    model = model_factory.SoftVotingEnsemble()

    # For the sake of the demo, we will train on the full training data.
    # The dataset is small enough (~900 samples) that with optimized hyperparameters
    # (reduced grid, folds, iterations), it will still run quickly.
    # Subsampling to 200 samples with 99 classes causes cross-validation failures
    # due to missing classes in folds (Cite debug_lesson_2).
    print(f"  - Fitting model on full training set ({len(X_train)} samples)...")
    model.fit(X_train, y_train)
    print("  - Model fitting complete.")

    # =========================================================================
    # 5. DEMONSTRATE PREDICTION AND SUBMISSION GENERATION
    # =========================================================================
    print("\n[5] Demonstrating Prediction...")

    # Predict on a subset of test data
    X_test_sub = X_test[:10]
    test_ids_sub = test_ids[:10]

    # Predict probabilities
    probas = model.predict_proba(X_test_sub)
    print(f"  - Probability matrix shape: {probas.shape}")

    # Predict labels
    preds = model.predict(X_test_sub)
    print(f"  - Predicted labels: {preds}")

    # Validation
    assert probas.shape == (10, len(classes)), "Probability shape mismatch"
    # Check that probabilities sum to roughly 1 (allow for float precision)
    row_sums = np.sum(probas, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Generate Submission DataFrame format
    print("  - formatting submission...")
    submission_df = pd.DataFrame(probas, columns=classes)
    submission_df.insert(0, "id", test_ids_sub)

    print("\nSample Submission Output (First 2 rows):")
    print(submission_df.head(2).to_string())

    # Verify submission format
    assert "id" in submission_df.columns, "Submission missing 'id' column"
    assert (
        len(submission_df.columns) == len(classes) + 1
    ), "Incorrect column count in submission"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
