import pandas as pd
import numpy as np
import os
import gc
import warnings
import sys

# Import library modules
import library.config as config
import library.model as model_lib
from library.features import generate_features
from library.utils import set_seed, calculate_competition_metric

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    set_seed(config.RANDOM_STATE)

    # Configure for Full Dataset Execution
    # We modify the configuration dictionary in-place. Since library.model imports
    # this dictionary object, it will see the changes.
    print("Configuring parameters for full dataset execution...")
    config.XGB_PARAMS["n_estimators"] = (
        6000  # Increased to ensure convergence (Cite solution_lesson_node_00013)
    )
    config.XGB_PARAMS["learning_rate"] = 0.05  # Kept at 0.05 for efficiency
    config.XGB_PARAMS["early_stopping_rounds"] = 50
    config.XGB_PARAMS["device"] = "cuda"  # Ensure GPU usage
    config.XGB_PARAMS["tree_method"] = "hist"

    # 2. Data Loading and Feature Engineering
    # -------------------------------------------------------------------------
    # Loads cached features if available, otherwise generates them
    print("Loading datasets...")
    train_df, val_df, test_df = generate_features(load_cached_data=True)

    # 3. Data Subsampling
    # -------------------------------------------------------------------------
    # Using full dataset for maximum performance (Cite solution_lesson_node_00021)
    print("Using full training dataset...")
    print(f"Training set shape: {train_df.shape}")

    # 4. Model Training
    # -------------------------------------------------------------------------
    print("Initializing Stratified Ensemble...")
    model = model_lib.StratifiedEnsemble()

    print("Starting training loop...")
    # fit() returns the validation dataframe with 'prediction' column appended
    val_preds = model.fit(train_df, val_df)

    # Clean up training data to free memory
    del train_df
    gc.collect()

    # 5. Validation and Metric Calculation
    # -------------------------------------------------------------------------
    print("Calculating final validation metric...")
    metric = calculate_competition_metric(
        val_preds,
        prediction_col="prediction",
        target_col="scalar_coupling_constant",
        type_col="type",
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    val_preds["abs_error"] = (
        val_preds["scalar_coupling_constant"] - val_preds["prediction"]
    ).abs()

    # identify numeric feature columns
    numeric_cols = val_preds.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude non-feature columns
    exclude_cols = [
        "id",
        "scalar_coupling_constant",
        "prediction",
        "abs_error",
        "atom_index_0",
        "atom_index_1",
    ]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Calculate correlations with error magnitude
    if feature_cols:
        correlations = (
            val_preds[feature_cols + ["abs_error"]]
            .corr()["abs_error"]
            .sort_values(ascending=False)
        )
        print(
            "Top 10 Features positively correlated with Error Magnitude (Systematic Failure Modes):"
        )
        print(correlations.head(11).iloc[1:])  # Skip abs_error itself
    else:
        print("No numeric features available for correlation analysis.")

    # 7. Submission Generation
    # -------------------------------------------------------------------------
    # Threshold defined in task: -1.2350378036499023
    # Metric is Log MAE (lower is better).
    THRESHOLD = -1.2350378036499023

    if metric < THRESHOLD:
        print(
            f"\nMetric ({metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set
        submission = model.predict(test_df)

        # Save submission
        save_path = config.DATA_PATHS["submission_output"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({metric}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
