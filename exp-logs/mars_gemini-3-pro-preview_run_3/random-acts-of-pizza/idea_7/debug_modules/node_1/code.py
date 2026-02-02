import os
import sys
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_datasets
from library.feature_extraction import extract_features
from library.model_pipeline import TriViewStackingClassifier

# Setup Logger
logger = setup_logger("main_demo")


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    logger.info("--- Step 1: Configuring environment for fast demonstration ---")

    # Override Config parameters for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50  # Small subset for demonstration
    Config.N_FOLDS = 2  # Minimal folds for CV

    # Reduce Random Forest complexity for speed
    Config.RF_PARAMS["n_estimators"] = 10
    Config.RF_PARAMS["min_samples_split"] = 2

    # Reduce Vectorizer features for speed
    Config.LEXICAL_TFIDF_PARAMS["max_features"] = 100
    Config.BEHAVIORAL_TFIDF_PARAMS["max_features"] = 50

    # Ensure reproducibility
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    logger.info("--- Step 2: Loading Data (Debug Mode) ---")

    # Load raw data subsets
    (train_data, y_train_raw), (val_data, y_val_raw), (test_data, test_ids_raw) = (
        load_datasets(debug=True)
    )

    # Verification
    assert (
        len(train_data) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} train samples, got {len(train_data)}"
    assert (
        len(val_data) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} val samples, got {len(val_data)}"
    assert (
        len(test_data) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} test samples, got {len(test_data)}"

    # Check for class balance in debug set to avoid AUC errors
    unique_classes = y_train_raw.unique()
    if len(unique_classes) < 2:
        logger.warning(
            "Debug subset contains only one class. Manually injecting a sample of the other class for valid AUC calculation."
        )
        # Flip the last label just for the demo to run without crashing on ROC AUC
        y_train_raw.iloc[-1] = 1 - y_train_raw.iloc[0]

    logger.info("Data loaded and verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Feature Extraction
    # -------------------------------------------------------------------------
    logger.info("--- Step 3: Extracting Features ---")

    # Run extraction (load_cache=False to demonstrate the actual processing)
    (X_train_dict, y_train), (X_val_dict, y_val), (X_test_dict, test_ids) = (
        extract_features(debug=True, load_cache=False)
    )

    # Verification of Feature Dictionary Structure
    expected_keys = {"lexical", "semantic", "behavioral", "meta"}
    assert expected_keys.issubset(
        X_train_dict.keys()
    ), "Missing keys in feature dictionary"

    # Verification of Shapes
    n_train = len(y_train)
    assert X_train_dict["lexical"].shape[0] == n_train, "Lexical features row mismatch"
    assert (
        X_train_dict["semantic"].shape[0] == n_train
    ), "Semantic features row mismatch"
    assert X_train_dict["meta"].shape[0] == n_train, "Meta features row mismatch"

    logger.info(f"Features extracted. Meta feature shape: {X_train_dict['meta'].shape}")

    # -------------------------------------------------------------------------
    # 4. Model Training (Stacking)
    # -------------------------------------------------------------------------
    logger.info("--- Step 4: Training Tri-View Stacking Classifier ---")

    model = TriViewStackingClassifier()

    # Fit the model using Cross-Validation for the meta-learner
    model.fit_cv(X_train_dict, y_train)

    # Verification
    assert model.is_fitted, "Model should be marked as fitted after fit_cv"
    assert hasattr(model.meta_learner, "coef_"), "Meta learner should have coefficients"

    logger.info("Model training complete.")

    # -------------------------------------------------------------------------
    # 5. Prediction & Evaluation
    # -------------------------------------------------------------------------
    logger.info("--- Step 5: Generating Predictions ---")

    # Predict on Validation set (just to check logic)
    val_probs = model.predict_proba(X_val_dict)

    assert val_probs.shape == (len(y_val),), "Validation predictions shape mismatch"
    assert np.all(
        (val_probs >= 0) & (val_probs <= 1)
    ), "Probabilities must be between 0 and 1"

    # Predict on Test set
    test_probs = model.predict_proba(X_test_dict)

    assert test_probs.shape == (len(test_ids),), "Test predictions shape mismatch"

    logger.info(f"Generated {len(test_probs)} predictions for test set.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    logger.info("--- Step 6: Saving Submission ---")

    submission_df = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: test_probs}
    )

    # Save to the configured submission path
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verification
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Check file content format
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(saved_df.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Submission columns mismatch"
    assert len(saved_df) == Config.DEBUG_SAMPLES, "Submission row count mismatch"

    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
