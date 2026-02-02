import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import VotingClassifier

# Import provided library modules
from library.config import Config
from library.data_processing import load_and_preprocess_data
from library.model_factory import create_hybrid_ensemble, train_and_evaluate
from library.utils import save_submission, calculate_metric

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def optimize_config_for_demo():
    """
    Modifies the Config class attributes to ensure the demonstration runs quickly.
    Reduces the number of iterations, CV folds, and grid search parameters.
    """
    print("Optimizing configuration for speed...")

    # 1. Logistic Regression Optimization
    # Reduce grid size for C parameter and reduce max iterations
    Config.LR_PARAMS["Cs"] = 2  # Only check 2 values instead of 20
    Config.LR_PARAMS["cv"] = 2  # Reduce CV folds from 5 to 2
    Config.LR_PARAMS["max_iter"] = 100  # Reduce max iterations

    # 2. SVM Optimization
    # Reduce the hyperparameter grid significantly
    Config.SVM_PARAM_GRID = {
        "C": [1.0],  # Single value
        "gamma": ["scale"],  # Single value
    }
    # Reduce calibration CV folds
    Config.SVM_CALIBRATION_PARAMS["cv"] = 2
    Config.SVM_GRID_CV = 2

    # 3. Ensure output directory matches current working expectation
    # (Though Config already sets this, we ensure it's valid)
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-run setup to create new directories
    Config.setup_directories()


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    optimize_config_for_demo()

    # 2. Data Loading and Preprocessing
    print("\n=== Step 1: Loading and Preprocessing Data ===")
    # We set load_cached_data=False to demonstrate the full processing pipeline
    # and because we changed the cache directory in optimize_config_for_demo
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = (
        load_and_preprocess_data(load_cached_data=False)
    )

    # Validation: Check data shapes
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")
    print(f"Number of Classes: {len(classes)}")

    assert X_train.shape[0] > 0, "Training data is empty"
    assert (
        X_train.shape[1] == 192
    ), f"Expected 192 features (64*3), got {X_train.shape[1]}"
    assert len(y_train) == X_train.shape[0], "Mismatch between X_train and y_train"
    assert len(classes) == 99, f"Expected 99 species classes, got {len(classes)}"

    # 3. Model Construction
    print("\n=== Step 2: Constructing Hybrid Ensemble ===")
    model = create_hybrid_ensemble()

    # Validation: Check model type
    assert isinstance(model, VotingClassifier), "Model is not a VotingClassifier"
    print("Ensemble model created successfully.")

    # 4. Training and Evaluation
    print("\n=== Step 3: Training and Evaluation ===")
    # Train the model and get the trained instance back
    trained_model = train_and_evaluate(model, X_train, y_train, X_val, y_val, classes)

    # Validation: Check if model is fitted
    # VotingClassifier usually has 'estimators_' attribute after fitting
    assert hasattr(trained_model, "estimators_"), "Model does not appear to be fitted"

    # 5. Prediction on Test Set
    print("\n=== Step 4: Generating Predictions ===")
    y_test_proba = trained_model.predict_proba(X_test)

    # Validation: Check prediction shape
    assert y_test_proba.shape == (
        len(X_test),
        len(classes),
    ), f"Prediction shape mismatch. Expected ({len(X_test)}, {len(classes)}), got {y_test_proba.shape}"

    # Check probability constraints (rows sum to ~1, values in [0, 1])
    row_sums = np.sum(y_test_proba, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"
    assert (y_test_proba >= 0).all() and (
        y_test_proba <= 1
    ).all(), "Probabilities out of range [0, 1]"

    # 6. Submission Generation
    print("\n=== Step 5: Saving Submission ===")
    save_submission(test_ids, classes, y_test_proba, output_path=Config.SUBMISSION_PATH)

    # Validation: Check if file exists and format is correct
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print(f"Submission dimensions: {df_sub.shape}")

    # Check columns: id + 99 classes = 100 columns
    expected_cols = 1 + len(classes)
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, got {df_sub.shape[1]}"
    assert "id" in df_sub.columns, "Column 'id' missing from submission"
    assert df_sub.shape[0] == len(X_test), "Row count mismatch in submission"

    print("\n=== Workflow Completed Successfully ===")


if __name__ == "__main__":
    main()
