import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# Import from the provided library
from library.config import Config
from library.utils import set_seed, ensure_directory
from library.data_loader import load_data
from library.feature_engineering import get_processed_data
from library.model_trainer import ModelTrainer


def main():
    print("Initializing Demonstration Script...")

    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config class attributes directly to optimize for a quick demo run.
    print("Configuring environment for rapid execution...")

    # Enable Debug mode to limit dataset size to 100 samples
    Config.DEBUG = True
    Config.MAX_SAMPLES = 50  # Small sample size for demonstration

    # Reduce the Hyperparameter Grid to a single point to skip time-consuming search
    Config.PARAM_GRID = {"C": [1.0], "l1_ratio": [0.5], "class_weight": [None]}

    # Reduce Bagging Ensemble complexity
    Config.BAGGING_PARAMS["n_estimators"] = 2  # Only 2 estimators for speed

    # Ensure reproducible results
    set_seed(Config.SEED)

    # 2. Data Loading
    print("\n--- Step 1: Data Loading ---")
    # Force reload from raw files/metadata to demonstrate logic, ignoring cache if present
    # Note: In a real run, we might want to use cache, but here we want to test the loading logic.
    # However, to be safe with time, we let the loader decide, but since we changed MAX_SAMPLES,
    # we should be careful. The loader samples *after* loading/caching.

    df_train, df_test = load_data(load_from_cache=False)

    print(f"Train Data Shape: {df_train.shape}")
    print(f"Test Data Shape: {df_test.shape}")

    # Validation: Check if data is loaded and sampled correctly
    assert not df_train.empty, "Training DataFrame is empty."
    assert not df_test.empty, "Test DataFrame is empty."
    assert (
        len(df_train) <= Config.MAX_SAMPLES
    ), f"Train data not sampled correctly. Size: {len(df_train)}"
    assert (
        "requester_received_pizza" in df_train.columns
    ), "Target column missing in Train Data."
    assert "request_id" in df_test.columns, "ID column missing in Test Data."

    print("Data loading verification passed.")

    # 3. Feature Engineering
    print("\n--- Step 2: Feature Engineering ---")
    # This step handles Text Embedding (SentenceTransformers) and Tabular Preprocessing

    # We force `load_from_cache=False` to ensure the code runs the embedding logic
    # rather than just loading old files which might have different sample counts.
    X_train, y_train, X_test = get_processed_data(
        df_train, df_test, load_from_cache=False
    )

    print(f"Processed Train Features Shape: {X_train.shape}")
    print(f"Processed Test Features Shape: {X_test.shape}")
    print(f"Target Shape: {y_train.shape}")

    # Validation: Check shapes and types
    assert X_train.shape[0] == len(
        df_train
    ), "Mismatch between Train Features and DataFrame rows."
    assert X_test.shape[0] == len(
        df_test
    ), "Mismatch between Test Features and DataFrame rows."
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), "Feature dimension mismatch between Train and Test."
    assert isinstance(X_train, np.ndarray), "X_train is not a numpy array."
    assert not np.isnan(X_train).any(), "NaN values found in X_train."

    print("Feature engineering verification passed.")

    # 4. Model Training
    print("\n--- Step 3: Model Training ---")
    trainer = ModelTrainer()

    # This will run the (simplified) GridSearch and then train the BaggingClassifier
    model, best_params = trainer.optimize_and_train(X_train, y_train)

    print(f"Training complete. Best Params: {best_params}")

    # Validation: Check if model is fitted and is the expected type
    assert isinstance(
        model, BaseEstimator
    ), "Returned model is not a scikit-learn estimator."
    # Check if we can predict (implies fitted)
    try:
        _ = model.predict(X_train[:5])
    except Exception as e:
        raise AssertionError(
            f"Model prediction failed (model might not be fitted): {e}"
        )

    print("Model training verification passed.")

    # 5. Prediction and Submission Generation
    print("\n--- Step 4: Prediction & Submission ---")

    # Predict probabilities for the positive class (1)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    print(f"Predictions generated. Shape: {y_pred_prob.shape}")
    print(f"Sample Predictions: {y_pred_prob[:5]}")

    # Validation: Check probability range
    assert (
        y_pred_prob.min() >= 0.0 and y_pred_prob.max() <= 1.0
    ), "Predictions out of probability range [0, 1]."

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": y_pred_prob}
    )

    # Save submission (as per Config path)
    ensure_directory(Config.SUBMISSION_PATH)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Verify file existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
