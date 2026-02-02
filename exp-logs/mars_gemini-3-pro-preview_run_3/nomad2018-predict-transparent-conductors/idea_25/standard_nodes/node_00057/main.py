import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Import from provided library files
from library.config import SUBMISSION_PATH, RANDOM_SEED
from library.preprocessor import get_preprocessed_data, TargetTransformer
from library.regressor import EnergyModel


def main():
    # Set random seed
    np.random.seed(RANDOM_SEED)

    print("Starting pipeline...")

    # 1. Load and Preprocess Training Data
    # We use load_cached_data=True to utilize any existing parquet files
    print("Loading Training Data...")
    train_df, cleaner = get_preprocessed_data(split="train", load_cached_data=True)

    # 2. Load and Preprocess Validation Data
    # We use the cleaner fitted on training data
    print("Loading Validation Data...")
    val_df, _ = get_preprocessed_data(
        split="val", cleaner=cleaner, load_cached_data=True
    )

    # Define feature columns
    exclude_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev", "file_path"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]

    X_val = val_df[feature_cols]
    y_val = val_df[target_cols]

    print(f"Training features: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # 3. Train Model
    # Using a reasonable number of estimators for a baseline that fits within time limits
    # The default in config is 3000, which is fine for this dataset size.
    model = EnergyModel(n_estimators=2000)
    model.fit(X_train, y_train, X_val, y_val)

    # 4. Validation Assessment
    print("\n--- Validation Assessment ---")
    predictions_val = model.predict(X_val)

    # Calculate RMSLE for each target
    # Metric is Column-wise Root Mean Squared Logarithmic Error
    # RMSLE = sqrt(mean( (log1p(y) - log1p(y_pred))^2 ))

    # Formation Energy
    y_val_form = y_val["formation_energy_ev_natom"]
    pred_val_form = predictions_val["formation_energy_ev_natom"]
    rmsle_form = np.sqrt(
        mean_squared_error(np.log1p(y_val_form), np.log1p(pred_val_form))
    )

    # Bandgap Energy
    y_val_band = y_val["bandgap_energy_ev"]
    pred_val_band = predictions_val["bandgap_energy_ev"]
    rmsle_band = np.sqrt(
        mean_squared_error(np.log1p(y_val_band), np.log1p(pred_val_band))
    )

    # Final Metric (Average of column-wise RMSLEs)
    final_metric = (rmsle_form + rmsle_band) / 2

    print(f"Formation Energy RMSLE: {rmsle_form}")
    print(f"Bandgap Energy RMSLE:   {rmsle_band}")
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude per sample (average log error)
    # Error_i = 0.5 * (|log1p(y_f) - log1p(p_f)| + |log1p(y_b) - log1p(p_b)|)
    error_form = np.abs(np.log1p(y_val_form) - np.log1p(pred_val_form))
    error_band = np.abs(np.log1p(y_val_band) - np.log1p(pred_val_band))
    mean_log_error = (error_form + error_band) / 2

    # Correlate error with features
    # Ensure X_val is all numeric for correlation
    X_val_numeric = X_val.select_dtypes(include=[np.number])

    correlations = (
        X_val_numeric.corrwith(mean_log_error).abs().sort_values(ascending=False)
    )

    print("Top 10 features correlated with model error:")
    print(correlations.head(10))

    # 6. Submission Generation
    THRESHOLD = 0.05500532306811823

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} < Threshold {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        print("Loading Test Data...")
        test_df, _ = get_preprocessed_data(
            split="test", cleaner=cleaner, load_cached_data=True
        )
        X_test = test_df[feature_cols]

        # Predict
        print("Predicting on Test Set...")
        predictions_test = model.predict(X_test)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "id": test_df["id"],
                "formation_energy_ev_natom": predictions_test[
                    "formation_energy_ev_natom"
                ],
                "bandgap_energy_ev": predictions_test["bandgap_energy_ev"],
            }
        )

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} >= Threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
