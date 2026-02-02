import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import from provided libraries
import library.config as config
import library.data as data
import library.model as model


def main():
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)

    print("Starting runfile.py execution...")

    # 1. Prepare Data
    # Note: prepare_train_test_data returns log-transformed targets (log1p)
    X_train, y_train_log_dict, X_val, y_val_log_dict, X_test, test_ids = (
        data.prepare_train_test_data(load_cached_data=True)
    )

    # 2. Train Models
    # The LogTransformedXGBoost class in library.model expects RAW targets and applies log1p internally.
    # Therefore, we must inverse-transform the data we got from library.data.

    # Define fast training parameters to ensure baseline execution within time limits
    # We override n_estimators to a lower value for speed, relying on early stopping
    fast_params = config.XGB_PARAMS.copy()
    fast_params["n_estimators"] = 1000
    fast_params["n_jobs"] = 4  # Use available cores

    models = {}
    val_preds_log = {}

    print("\n--- Model Training ---")
    for target in config.TARGET_COLS:
        print(f"Training for target: {target}")

        # Inverse transform to get raw values for the model wrapper
        y_train_raw = np.expm1(y_train_log_dict[target])
        y_val_raw = np.expm1(y_val_log_dict[target])

        trained_model = model.train_model(
            X_train,
            y_train_raw,
            X_val,
            y_val_raw,
            params=fast_params,
            target_name=target,
            early_stopping_rounds=50,
            verbose=False,
        )

        models[target] = trained_model

        # Generate predictions on validation set (model.predict returns raw values)
        raw_preds = trained_model.predict(X_val)
        # Store log predictions for RMSLE calculation
        val_preds_log[target] = np.log1p(raw_preds)

    # 3. Validation Assessment
    print("\n--- Validation Assessment ---")
    rmsle_scores = []
    for target in config.TARGET_COLS:
        # y_val_log_dict is already log1p(true)
        # val_preds_log is log1p(pred)
        mse = mean_squared_error(y_val_log_dict[target], val_preds_log[target])
        rmsle = np.sqrt(mse)
        rmsle_scores.append(rmsle)
        print(f"{target} RMSLE: {rmsle:.6f}")

    # The metric is Column-wise root mean squared logarithmic error.
    # Usually averaged over columns for a single score.
    final_metric = np.mean(rmsle_scores)
    print(f"Final Validation Metric: {final_metric:.18f}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Identify features most correlated with error magnitude
    # We analyze the error in log space since the metric is RMSLE

    # Concatenate errors for analysis
    error_correlations = {}

    # We'll compute the mean absolute error in log space across targets for a general error metric
    # or analyze per target. Let's do per target to be specific.

    for target in config.TARGET_COLS:
        print(f"\nAnalyzing errors for {target}:")
        # Calculate absolute error in log space
        abs_log_error = np.abs(val_preds_log[target] - y_val_log_dict[target])

        # Create a dataframe with features and error
        analysis_df = X_val.copy()
        analysis_df["log_error_magnitude"] = abs_log_error

        # Compute correlation
        corr = analysis_df.corr()["log_error_magnitude"].drop("log_error_magnitude")

        # Get top 5 positive and negative correlations
        top_positive = corr.sort_values(ascending=False).head(5)

        print("Top features associated with high error:")
        print(top_positive)

    # 5. Submission Generation
    # Threshold from requirements
    THRESHOLD = 0.056919346405286564

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        model.generate_submission(
            models["formation_energy_ev_natom"],
            models["bandgap_energy_ev"],
            X_test,
            test_ids,
            config.SUBMISSION_PATH,
        )
    else:
        print(
            f"\nValidation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
