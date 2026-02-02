import os
import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from library.config import VAL_PATH, SUBMISSION_PATH, SEED
from library.ensemble_trainer import EnsembleTrainer
from library.feature_engineering import FeatureEngineer
from library.data_loader import clean_training_data


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Ensure reproducibility
    np.random.seed(SEED)

    print("=== Starting Full Training Run ===")

    # 1. Train Stacked Ensemble on Full Data
    # Utilizing the full dataset to maximize model performance (Cite solution_lesson_node_00012).
    # The train_stack method handles the splitting of Base/Meta sets and training.
    trainer = EnsembleTrainer()
    print("Training models on full dataset...")
    models = trainer.train_stack(debug=False)

    # 2. rigorous Evaluation on Full Validation Set
    # The metric returned by train_stack is based on the debug subset.
    # We must evaluate on the full hold-out validation set for the official metric.
    print("\n=== Performing Full Validation ===")
    print(f"Loading full validation set from {VAL_PATH}...")
    val_df = pd.read_parquet(VAL_PATH)

    # Apply the same cleaning logic as training (filtering invalid coordinates/fares)
    val_df = clean_training_data(val_df)

    # Feature Engineering for the full validation set
    # We use a distinct name 'val_full_eval' to avoid cache collisions with the debug run
    fe = FeatureEngineer()
    val_df = fe.process(val_df, name="val_full_eval")

    # Prepare Feature Matrix
    target_col = "fare_amount"
    ignore_cols = ["key", "fare_amount", "pickup_datetime"]
    features = [c for c in val_df.columns if c not in ignore_cols]

    X_val = val_df[features]
    y_val = val_df[target_col]

    # Generate Predictions
    print("Generating predictions on full validation set...")
    xgb_model = models["xgb"]
    lgbm_model = models["lgbm"]
    meta_model = models["meta"]

    # Base Learner Predictions
    xgb_pred = xgb_model.predict(X_val)
    lgbm_pred = lgbm_model.predict(X_val)

    # Stack Predictions for Meta Learner
    X_stack = np.column_stack((xgb_pred, lgbm_pred))

    # Final Ensemble Prediction
    final_pred = meta_model.predict(X_stack)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, final_pred))

    # REQUIRED OUTPUT: Print Final Metric
    print(f"Final Validation Metric: {rmse}")

    # 3. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Absolute Error
    val_df["predicted_fare"] = final_pred
    val_df["abs_error"] = np.abs(val_df["fare_amount"] - val_df["predicted_fare"])

    # Calculate correlation between features and error
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns
    correlations = (
        val_df[numeric_cols].corrwith(val_df["abs_error"]).sort_values(ascending=False)
    )

    print("Top features correlated with prediction error:")
    print(correlations.head(10))

    # Clean up memory
    del val_df, X_val, X_stack, xgb_pred, lgbm_pred, final_pred
    gc.collect()

    # 4. Submission Management
    # Requirement: Generate submission IF AND ONLY IF metric < threshold.
    # train_stack has already generated a submission file based on the test set.
    # We verify the metric and delete the file if the model is not good enough.
    THRESHOLD = 3.3935366001817666

    if rmse < THRESHOLD:
        print(f"\nValidation RMSE ({rmse}) is below threshold ({THRESHOLD}).")
        print(f"Submission file preserved at {SUBMISSION_PATH}")
    else:
        print(f"\nValidation RMSE ({rmse}) is above threshold ({THRESHOLD}).")
        if os.path.exists(SUBMISSION_PATH):
            print("Removing submission file...")
            os.remove(SUBMISSION_PATH)
        else:
            print("No submission file found to remove.")


if __name__ == "__main__":
    main()
