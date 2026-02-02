import os
import sys
import gc
import numpy as np
import pandas as pd

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library import config
from library import preprocessor
from library import feature_engine
from library import model_handler


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Initialization
    set_seed(config.RANDOM_SEED)
    print("Starting pipeline execution...")

    # 2. Data Loading and Preprocessing
    # Loads raw data, applies clamping/rounding, and caches results
    print("Loading and preprocessing data splits...")
    full_train_df, train_subsample_df, val_df, test_df = (
        preprocessor.get_preprocessed_splits(load_cached_data=True)
    )

    # 2.5. Data Filtering
    # Cite solution_lesson_node_00017: Sanitize target variable before scaling up.
    print("Filtering outliers from training data...")
    full_train_df = preprocessor.filter_target_outliers(full_train_df)
    train_subsample_df = preprocessor.filter_target_outliers(train_subsample_df)

    # 3. Feature Engineering
    # 3a. Geometric Features
    # Cite solution_lesson_node_00046: Prioritize Inductive Bias (Explicit Features).
    print("Adding geometric features...")
    train_subsample_df = feature_engine.add_geometric_features(train_subsample_df)
    val_df = feature_engine.add_geometric_features(val_df)
    test_df = feature_engine.add_geometric_features(test_df)

    # 3b. Global-Prior Augmented Target Encoding
    print("Initializing Global Route Encoder...")
    encoder = feature_engine.GlobalRouteEncoder()

    # Fit on the full dataset (Knowledge Aggregation)
    # We force re-computation (load_cached_data=True but new dir)
    # to ensure stats are computed on the filtered dataset.
    encoder.fit(full_train_df, load_cached_data=True)

    # Release memory for the full dataset as it's no longer needed
    del full_train_df
    gc.collect()

    # Transform the Training Subsample using Vectorized Subtraction
    # This prevents leakage while using global stats
    print("Applying background-augmented encoding to training subsample...")
    train_ready = encoder.transform_train_vectorized(train_subsample_df)

    # Transform Validation and Test sets (Direct Inference)
    print("Applying global encoding to validation and test sets...")
    val_ready = encoder.transform_inference(val_df)
    test_ready = encoder.transform_inference(test_df)

    # Cleanup raw dataframes
    del train_subsample_df
    gc.collect()

    # 4. Model Training
    print("Initializing Model Handler...")
    handler = model_handler.ModelHandler()
    target_col = config.TARGET_COL

    # Train the model
    # The handler automatically selects numeric features and excludes 'key', etc.
    model = handler.train_model(
        train_ready, train_ready[target_col], val_ready, val_ready[target_col]
    )

    # 5. Validation Assessment
    print("Performing validation assessment...")
    # Prepare validation features explicitly to get predictions for analysis
    X_val = handler._prepare_features(val_ready)
    y_val = val_ready[target_col]

    # Generate predictions (batch inference is handled by XGBoost)
    val_preds = model.predict(X_val)

    # Calculate RMSE
    mse = np.mean((y_val - val_preds) ** 2)
    rmse = np.sqrt(mse)

    # Print required metric format
    print(f"Final Validation Metric: {rmse}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - val_preds)

    # Create a temporary dataframe for correlation analysis
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlation between features and error magnitude
    correlations = (
        analysis_df.corr()["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False)
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.head(5))

    # 7. Submission Generation
    # Threshold defined in the task description logic
    THRESHOLD = 3.5069767944123895

    if rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({rmse}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions for test set
        # handler.generate_predictions applies the min_fare floor automatically
        test_preds = handler.generate_predictions(test_ready)

        # Create submission file
        # We pass test_ready because it still contains the 'key' column required for submission
        handler.create_submission(test_ready, test_preds)

    else:
        print(
            f"\nValidation RMSE ({rmse}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
