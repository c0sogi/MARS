import os
import numpy as np
import pandas as pd
import warnings

# Import from provided libraries
from library.data_factory import CrystalDataHandler
from library.model_factory import train_models, predict_models
from library.utils import save_submission, calculate_rmsle

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Data Loading & Feature Generation
    # The handler will load metadata and generate/load features from cache
    print("Initializing Data Handler...")
    handler = CrystalDataHandler()

    print("Loading data...")
    # y_train and y_val are already log1p transformed by the handler
    (X_train, y_train), (X_val, y_val), X_test, test_ids = handler.load_data(
        load_cached_data=True
    )

    # 2. Model Training
    print("Training models...")
    # train_models returns a dictionary of trained XGBoostRegressorWrapper objects
    models = train_models(X_train, y_train, X_val, y_val)

    # 3. Validation & Metric Calculation
    print("Validating models...")
    val_preds_log = {}
    val_metrics = {}
    targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    total_rmsle = 0.0

    for target in targets:
        model = models[target]
        # Predict in log space using the wrapper's predict_model method
        # This method handles feature pruning automatically
        log_pred = model.predict_model(X_val)
        val_preds_log[target] = log_pred

        # Calculate RMSE in log space, which is equivalent to RMSLE in original space
        # RMSLE = sqrt(mean((log(1+y) - log(1+pred))^2))
        # y_val is already log(1+y), and log_pred is log(1+pred)
        mse = np.mean((y_val[target].values - log_pred) ** 2)
        rmsle = np.sqrt(mse)
        val_metrics[target] = rmsle
        total_rmsle += rmsle

        print(f"Target: {target}, RMSLE: {rmsle:.6f}")

    # Calculate final column-wise mean RMSLE
    final_metric = total_rmsle / len(targets)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # We analyze correlation between absolute error (in log space) and features
    analysis_df = X_val.copy()

    for target in targets:
        # Calculate absolute error
        error = np.abs(y_val[target].values - val_preds_log[target])
        error_col_name = f"error_{target}"
        analysis_df[error_col_name] = error

        # Calculate correlations with the error
        # We drop the error columns themselves to avoid self-correlation
        drop_cols = [
            f"error_{t}" for t in targets if f"error_{t}" in analysis_df.columns
        ]
        correlations = (
            analysis_df.corrwith(analysis_df[error_col_name])
            .abs()
            .sort_values(ascending=False)
        )

        print(f"\nTop 5 features correlated with error in {target}:")
        # Filter out the error columns from the top list just in case
        print(correlations.drop(labels=drop_cols, errors="ignore").head(5))

    # 5. Submission
    # Threshold defined in the prompt
    THRESHOLD = 0.06278041684313306

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric:.6f} is below threshold {THRESHOLD}. Generating submission..."
        )

        # Generate predictions for test set
        # predict_models returns a DataFrame with columns 'formation_energy_ev_natom' and 'bandgap_energy_ev'
        # The values are already inverse transformed (expm1)
        submission_df = predict_models(models, X_test)

        # Add the ID column
        submission_df["id"] = test_ids.values

        # Ensure correct column order
        cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        submission_df = submission_df[cols]

        # Save submission
        output_path = "./submission/submission.csv"
        save_submission(
            submission_df["id"],
            submission_df["formation_energy_ev_natom"],
            submission_df["bandgap_energy_ev"],
            output_path,
        )
    else:
        print(
            f"\nValidation metric {final_metric:.6f} is NOT below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
