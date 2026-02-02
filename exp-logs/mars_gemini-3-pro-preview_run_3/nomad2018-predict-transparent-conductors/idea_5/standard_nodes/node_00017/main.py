import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from library.model_trainer import EnergyPredictor
from library.config import TARGET_COLS


def main():
    # 1. Initialize the predictor
    predictor = EnergyPredictor()

    # 2. Train the models
    # We use all available data (max_samples=None) as the dataset is small (~2k rows)
    # and XGBoost training is efficient.
    predictor.train_model(max_samples=None, load_cached_data=True)

    # 3. Validation Assessment
    print("\n" + "=" * 40)
    print(" VALIDATION ASSESSMENT ")
    print("=" * 40)

    # Load validation data manually to compute metrics and perform analysis
    X_val, y_val, _ = predictor.prepare_data(
        "val", max_samples=None, load_cached_data=True
    )

    rmsle_scores = []
    # DataFrame to store errors for analysis
    error_df = pd.DataFrame(index=X_val.index)

    for target in TARGET_COLS:
        model = predictor.models[target]
        y_true = y_val[target]

        # Predict on validation set
        # The model was trained on log1p(y), so it predicts log1p(y)
        y_pred_log = model.predict(X_val)

        # Ground truth in log space for RMSLE calculation
        y_true_log = np.log1p(y_true)

        # Calculate RMSLE (which is RMSE of log-transformed values)
        mse = mean_squared_error(y_true_log, y_pred_log)
        rmsle = np.sqrt(mse)
        rmsle_scores.append(rmsle)

        # Store absolute error for failure analysis
        error_df[f"{target}_error"] = np.abs(y_true_log - y_pred_log)

        print(f"Target: {target} | RMSLE: {rmsle:.10f}")

    # Compute Final Metric (Mean of Column-wise RMSLE)
    final_metric = np.mean(rmsle_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS ")
    print("=" * 40)

    # Calculate mean error across all targets for each sample
    mean_error = error_df.mean(axis=1)

    # Select numeric features for correlation analysis
    X_val_numeric = X_val.select_dtypes(include=[np.number])

    # Compute correlation between features and prediction error
    correlations = X_val_numeric.corrwith(mean_error)

    # Sort by absolute correlation strength
    sorted_corrs = correlations.abs().sort_values(ascending=False)

    print("Top 10 features correlated with prediction error:")
    print(sorted_corrs.head(10))

    # 5. Submission Generation
    THRESHOLD = 0.06278041684313306

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} is lower than threshold {THRESHOLD}.")
        print("Generating submission file...")
        predictor.predict_and_submit(max_samples=None, load_cached_data=True)
    else:
        print(f"\nMetric {final_metric} is NOT lower than threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
