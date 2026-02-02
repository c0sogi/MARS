import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library import config
from library import data_loader
from library import fe_bgp_model


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("Initializing Demo...")
    set_seed(config.RANDOM_SEED)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # ------------------------------------------------------------------------
    # 1. Demonstrate Data Loading
    # ------------------------------------------------------------------------
    print("\n--- [Demo] Testing Data Loader ---")

    # We load the full dataset. The dataset is small (Train ~712 rows),
    # so this is computationally feasible and ensures all classes are present
    # for the LDA component of the model.
    X_train, y_train, X_val, y_val, X_test, test_ids = data_loader.load_data(
        load_cached_data=False,  # Force load from CSV to demonstrate raw loading
        sample_size=None,  # Use full data to satisfy LDA n_components constraints
    )

    # Assertions to verify data integrity
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    assert X_train.shape[1] == 192, f"Expected 192 features, got {X_train.shape[1]}"
    assert len(X_train) == len(y_train), "Mismatch between X_train and y_train length"
    assert len(X_val) == len(y_val), "Mismatch between X_val and y_val length"
    assert len(X_test) == len(test_ids), "Mismatch between X_test and test_ids length"

    print("Data Loader assertions passed.")

    # ------------------------------------------------------------------------
    # 2. Demonstrate Model Usage (FisherEmbeddedBGP)
    # ------------------------------------------------------------------------
    print("\n--- [Demo] Testing FisherEmbeddedBGP Model ---")

    # Instantiate the model
    model = fe_bgp_model.FisherEmbeddedBGP(random_state=config.RANDOM_SEED)

    # Fit on training data
    # Note: This involves PowerTransformer -> StandardScaler -> LDA -> GaussianProcessClassifier
    print("Fitting model (this may take a moment due to GPC)...")
    model.fit(X_train, y_train)

    # Predict on validation data
    print("Predicting on validation set...")
    val_probs = model.predict_proba(X_val)

    # Verify output shape and values
    n_classes = len(np.unique(y_train))
    assert val_probs.shape == (
        len(X_val),
        n_classes,
    ), f"Expected prob shape {(len(X_val), n_classes)}, got {val_probs.shape}"

    # Check probability properties
    # Note: Floating point precision might cause slight deviations, so we use a small epsilon
    assert np.all(val_probs >= 0), "Probabilities should be non-negative"
    assert np.all(val_probs <= 1), "Probabilities should be <= 1"

    # Check if classes are stored correctly
    assert len(model.classes_) == n_classes, "Model classes attribute mismatch"

    print("Model assertions passed.")

    # ------------------------------------------------------------------------
    # 3. Demonstrate Full Pipeline Execution
    # ------------------------------------------------------------------------
    print("\n--- [Demo] Executing Full Pipeline ---")

    # This function handles loading, training on Train+Val, and generating submission
    # We use load_cached_data=True here since we just processed/cached data in step 1
    # (data_loader saves to cache automatically).
    fe_bgp_model.run_fe_bgp_pipeline(load_cached_data=True, sample_size=None)

    # ------------------------------------------------------------------------
    # 4. Validate Submission File
    # ------------------------------------------------------------------------
    print("\n--- [Demo] Validating Submission File ---")

    if not os.path.exists(config.SUBMISSION_FILE_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_FILE_PATH}"
        )

    df_sub = pd.read_csv(config.SUBMISSION_FILE_PATH)
    print(f"Submission shape: {df_sub.shape}")

    # Check required columns
    assert config.ID_COL in df_sub.columns, f"Missing ID column '{config.ID_COL}'"

    # Check row count matches test set
    # Note: df_test loaded in step 1
    assert len(df_sub) == len(
        X_test
    ), f"Submission row count {len(df_sub)} != Test set size {len(X_test)}"

    # Check probability columns
    # Excluding ID column
    prob_cols = [c for c in df_sub.columns if c != config.ID_COL]
    assert len(prob_cols) == 99, f"Expected 99 species columns, found {len(prob_cols)}"

    # Check value ranges in submission
    probs = df_sub[prob_cols].values
    assert np.all(probs >= 0) and np.all(
        probs <= 1
    ), "Submission probabilities out of range [0, 1]"

    print("Submission file validation passed.")
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
