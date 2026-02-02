import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_factory import DataFactory
from library.classifier import TokenClassifier
from library.inference import generate_predictions
from library.dictionary import NormalizationDictionary


def run_demo():
    print("=== Text Normalization Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Overriding Configuration for Fast Demonstration...")

    # Limit data size significantly for the demo
    Config.DEBUG_ROW_LIMIT = 5000

    # Adjust XGBoost parameters for speed
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["early_stopping_rounds"] = 2
    Config.XGB_PARAMS["max_depth"] = 4  # Shallower trees for speed

    # Ensure working directories are clean or ready (optional, but good practice)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"DEBUG_ROW_LIMIT set to: {Config.DEBUG_ROW_LIMIT}")
    print(f"XGB_PARAMS n_estimators set to: {Config.XGB_PARAMS['n_estimators']}")

    # ---------------------------------------------------------
    # 2. Data Preparation (DataFactory)
    # ---------------------------------------------------------
    print("\n[2] Testing DataFactory and Feature Pipeline...")

    # Instantiate DataFactory
    factory = DataFactory()

    # Prepare training data (forces re-computation with load_cached_data=False)
    # This tests: FeaturePipeline.get_train_data, FeaturePipeline.get_val_data, NormalizationDictionary.build
    (X_train, y_train), (X_val, y_val) = factory.prepare_training_data(
        load_cached_data=False
    )

    print(f"Training Data Shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation Data Shape: X={X_val.shape}, y={y_val.shape}")

    # Assertions to verify data integrity
    assert len(X_train) == len(y_train), "Mismatch in training features and labels"
    assert len(X_val) == len(y_val), "Mismatch in validation features and labels"
    assert not X_train.empty, "Training dataframe is empty"
    assert not X_val.empty, "Validation dataframe is empty"

    # Verify features were generated (check for specific columns)
    expected_cols = ["len", "is_digit", "token_hash"]  # Orthographic features
    for col in expected_cols:
        assert col in X_train.columns, f"Missing expected feature column: {col}"

    # Check if vocabulary was saved
    vocab_path = os.path.join(Config.WORKING_DIR, "vectorizer_vocab.json")
    assert os.path.exists(vocab_path), f"Vocabulary file not found at {vocab_path}"

    # ---------------------------------------------------------
    # 3. Normalization Dictionary Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Normalization Dictionary...")

    # Get the dictionary instance
    norm_dict = factory.get_normalization_dictionary()

    # Verify the dictionary file exists
    assert os.path.exists(
        Config.NORM_DICT_PATH
    ), "Normalization dictionary JSON not found"

    # Verify internal mapping is populated
    assert len(norm_dict.mapping) > 0, "Normalization dictionary mapping is empty"

    # Test a simple lookup (Self-consistency check)
    # We pick a class and token that likely exists in the first 5000 rows, e.g., PLAIN "the"
    # Note: Since we downsample PLAIN, it might be sparse, but 'PLAIN' class usually exists.
    sample_class = list(norm_dict.mapping.keys())[0]
    sample_token = list(norm_dict.mapping[sample_class].keys())[0]
    normalized = norm_dict.get_normalization(sample_token, sample_class)

    print(
        f"Dictionary Test: Class='{sample_class}', Token='{sample_token}' -> '{normalized}'"
    )
    assert isinstance(normalized, str), "Normalized output must be a string"

    # ---------------------------------------------------------
    # 4. Model Training (TokenClassifier)
    # ---------------------------------------------------------
    print("\n[4] Testing TokenClassifier (XGBoost)...")

    classifier = TokenClassifier()

    # Train the model
    classifier.train(X_train, y_train, X_val, y_val)

    # Verify model object exists
    assert classifier.model is not None, "Model was not trained successfully"

    # Test Save/Load
    print("Testing Model Persistence...")
    classifier.save()
    assert os.path.exists(Config.MODEL_FILE), "Model file was not saved"

    # Create a new instance and load
    classifier_loaded = TokenClassifier()
    classifier_loaded.load()
    assert classifier_loaded.model is not None, "Failed to load model"

    # ---------------------------------------------------------
    # 5. Inference Pipeline
    # ---------------------------------------------------------
    print("\n[5] Testing Inference Generation...")

    # Run the full inference pipeline provided in library.inference
    # This uses the test set (limited by DEBUG_ROW_LIMIT)
    generate_predictions(load_cached_data=False)

    # Verify submission file
    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Load submission and check format
    df_sub = pd.read_csv(submission_path)
    print(f"Submission Shape: {df_sub.shape}")
    print("Submission Head:")
    print(df_sub.head(3))

    # Check columns
    assert list(df_sub.columns) == [
        "id",
        "after",
    ], f"Invalid submission columns: {df_sub.columns}"

    # Check row count matches the debug limit (or test set size if smaller)
    # Note: FeaturePipeline limits the input DF, so output should match that size
    # However, DataFactory.prepare_test_data returns X_test and meta_df.
    # We can check if the submission length matches the meta_df length implicitly via logic.
    # Since we set DEBUG_ROW_LIMIT=5000, we expect roughly 5000 rows (or less if test set is smaller).
    assert len(df_sub) > 0, "Submission file is empty"
    if len(df_sub) > Config.DEBUG_ROW_LIMIT:
        print(
            "Warning: Submission size exceeds debug limit (check if limit applied to test loading correctly)."
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Set global seed for reproducibility
    set_seed(42)

    # Run the demonstration
    try:
        run_demo()
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
