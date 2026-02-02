import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Import from provided libraries
from library.data import load_dataset
from library.model import DualTargetRegressor, save_submission
from library.config import TARGET_COLS


def main():
    # 1. Load Data
    # load_cached_data=True utilizes the preprocessed parquet files in ./working
    print("Loading datasets...")
    X_train, y_train = load_dataset("train", load_cached_data=True)
    X_val, y_val = load_dataset("val", load_cached_data=True)
    X_test, _ = load_dataset("test", load_cached_data=True)

    # 2. Preprocessing
    # Remove 'id' column from features if present, as it's not a predictive feature
    # but keep track of test IDs for the submission file.
    if "id" in X_test.columns:
        test_ids = X_test["id"]
        X_test = X_test.drop(columns=["id"])
    else:
        # Fallback: recreate IDs if missing (though library guarantees existence)
        test_ids = pd.Series(range(1, len(X_test) + 1), name="id")

    if "id" in X_train.columns:
        X_train = X_train.drop(columns=["id"])
    if "id" in X_val.columns:
        X_val = X_val.drop(columns=["id"])

    # 3. Model Training
    print("Initializing and training model...")
    regressor = DualTargetRegressor()

    # Train the model
    # The train method returns a dict of metrics, but we will compute the final metric
    # explicitly to ensure it matches the exact definition.
    regressor.train(X_train, y_train, X_val, y_val)

    # 4. Validation Assessment
    print("Performing validation...")
    val_preds_log = regressor.predict(X_val)

    rmsle_scores = []
    for target in TARGET_COLS:
        # y_val is already log1p transformed (z = log(1+y))
        # val_preds_log is the raw output of the model (predicting z)
        # Therefore, RMSE in this space is equivalent to RMSLE in original space.
        mse = mean_squared_error(y_val[target], val_preds_log[target])
        rmsle = np.sqrt(mse)
        rmsle_scores.append(rmsle)
        # print(f"RMSLE for {target}: {rmsle}")

    # Metric: Column-wise root mean squared logarithmic error
    # We take the mean of the RMSLEs for the two targets.
    final_metric = np.mean(rmsle_scores)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    # Calculate absolute error in log space
    errors = np.abs(y_val - val_preds_log)

    # Select numeric features for correlation analysis
    X_val_numeric = X_val.select_dtypes(include=[np.number])

    for target in TARGET_COLS:
        print(f"\n--- Top feature correlations with error for {target} ---")
        target_error = errors[target]
        # Compute correlation between feature values and error magnitude
        # This helps identify if errors are larger for specific physical properties
        correlations = (
            X_val_numeric.corrwith(target_error).abs().sort_values(ascending=False)
        )
        print(correlations.head(5))

    # 6. Submission Generation
    # Threshold defined in task description
    threshold = 0.056919346405286564

    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )
        test_preds_log = regressor.predict(X_test)
        save_submission(test_ids, test_preds_log)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
