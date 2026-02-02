import os
import sys
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library import config
from library import utils
from library.feature_engine import TabularMessagePasser
from library.model_trainer import StratifiedModelManager


def main():
    # 1. Configuration and Setup
    # Suppress warnings and progress bars for clean output
    warnings.filterwarnings("ignore")
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_STATE)

    # 2. Feature Engineering
    # Initialize engine with verbose=False to suppress tqdm progress bars
    passer = TabularMessagePasser(verbose=False)

    # Load Validation Data (Full set for accurate metric calculation)
    # We load this first to ensure we have the ground truth for evaluation
    val_df = passer.get_val_data(load_cached_data=True)

    # Load Training Data (Full Dataset)
    # Cite solution_lesson_node_00021: Prioritize Data Volume over Feature Complexity.
    # We use the full dataset to maximize model performance, relying on the
    # vectorized feature engine to handle the scale efficiently.
    train_df = passer.get_train_data(load_cached_data=True, nrows=None)

    # 3. Model Training
    # Initialize manager with verbose=False to reduce log noise
    manager = StratifiedModelManager(verbose=False)

    # Train models for all coupling types
    # This returns a dictionary of Log MAE scores per type
    # The manager handles feature selection and GPU acceleration internally
    type_scores = manager.train_all_types(train_df, val_df)

    # 4. Evaluation
    # Calculate the global metric (mean of Log MAE across types)
    final_metric = np.mean(list(type_scores.values()))

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Generate predictions for the validation set to analyze errors
    val_preds = manager.predict_all_types(val_df)

    # Merge predictions with actual values
    # Note: val_preds contains ['id', 'scalar_coupling_constant']
    val_analysis = val_df.merge(val_preds, on="id", suffixes=("", "_pred"))

    # Calculate Absolute Error
    val_analysis["abs_error"] = np.abs(
        val_analysis["scalar_coupling_constant"]
        - val_analysis["scalar_coupling_constant_pred"]
    )

    # Calculate correlation between numeric features and absolute error
    # Identify numeric columns
    numeric_cols = val_analysis.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude metadata and target columns from correlation analysis
    exclude_cols = [
        "id",
        "scalar_coupling_constant",
        "scalar_coupling_constant_pred",
        "abs_error",
        "atom_index_0",
        "atom_index_1",
    ]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Compute correlations
    correlations = (
        val_analysis[feature_cols]
        .corrwith(val_analysis["abs_error"])
        .sort_values(ascending=False)
    )

    print("\nFailure Analysis - Top 5 Features Correlated with Error:")
    print(correlations.head(5))

    # 6. Submission Logic
    # Threshold defined in task
    TARGET_THRESHOLD = -0.7386035268505905

    if final_metric < TARGET_THRESHOLD:
        # Load Test Data
        test_df = passer.get_test_data(load_cached_data=True)

        # Generate Predictions
        submission_df = manager.predict_all_types(test_df)

        # Save Submission
        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)


if __name__ == "__main__":
    main()
