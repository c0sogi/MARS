import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import feature_engineering
from library import model_rf
from library import model_mlp


def main():
    # 1. Setup and Reproducibility
    utils.set_seed(config.RANDOM_STATE)

    # 2. Load Data
    print("Loading datasets...")
    train_df, val_df, test_df = data_loader.load_dataset(load_cached_data=True)

    # 3. Feature Engineering
    print("Running feature engineering pipeline...")
    fe = feature_engineering.FeatureEngineer()

    # process_data returns two tuples: one for RF features, one for MLP features
    rf_data, mlp_data = fe.process_data(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Unpack RF Data (Stream A)
    X_train_rf, y_train_rf, X_val_rf, y_val_rf, X_test_rf = rf_data

    # Unpack MLP Data (Stream B)
    X_train_mlp, y_train_mlp, X_val_mlp, y_val_mlp, X_test_mlp = mlp_data

    # Targets are identical, but we use the unpacked ones for consistency
    y_train = y_train_rf
    y_val = y_val_rf

    # 4. Train Models

    # --- Stream A: Random Forest ---
    print("\n--- Training Random Forest (Stream A) ---")
    rf_model_path = os.path.join(config.WORKING_DIR, "best_rf.pkl")
    rf_model, rf_val_auc = model_rf.train_rf_model(
        X_train_rf, y_train, X_val_rf, y_val, save_path=rf_model_path
    )
    print(f"Random Forest Validation AUC: {rf_val_auc}")

    # --- Stream B: Topology-Aware MLP ---
    print("\n--- Training Topology-Aware MLP (Stream B) ---")
    mlp_model_path = os.path.join(config.WORKING_DIR, "best_mlp.pth")
    mlp_model, mlp_val_auc = model_mlp.train_mlp_model(
        X_train_mlp, y_train, X_val_mlp, y_val, save_path=mlp_model_path
    )
    print(f"MLP Validation AUC: {mlp_val_auc}")

    # 5. Ensemble Evaluation
    print("\n--- Ensemble Evaluation ---")

    # Generate predictions on validation set
    rf_val_probs = model_rf.predict_rf_model(rf_model, X_val_rf)
    mlp_val_probs = model_mlp.predict_mlp_model(mlp_model, X_val_mlp)

    # Weighted Average Ensemble
    w_rf = config.ENSEMBLE_WEIGHT_RF
    w_mlp = config.ENSEMBLE_WEIGHT_MLP

    ensemble_val_probs = (w_rf * rf_val_probs) + (w_mlp * mlp_val_probs)

    # Compute Final Metric
    final_val_auc = roc_auc_score(y_val, ensemble_val_probs)
    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude (absolute difference)
    # y_val is 0 or 1, probs are 0..1
    errors = np.abs(y_val - ensemble_val_probs)

    # Correlate errors with numerical features in the validation set
    # We exclude the target and any non-numeric columns
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    if "requester_received_pizza" in numeric_cols:
        numeric_cols.remove("requester_received_pizza")

    correlations = {}
    for col in numeric_cols:
        # Fill NaNs with 0 for correlation calculation to avoid errors
        feat_values = val_df[col].fillna(0).values

        # Ensure there is variance
        if np.std(feat_values) > 0:
            corr = np.corrcoef(errors, feat_values)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_correlations = sorted(
        correlations.items(), key=lambda x: abs(x[1]), reverse=True
    )

    print("Top 5 Features Correlated with Prediction Error:")
    for feature, corr in sorted_correlations[:5]:
        print(f"{feature}: {corr:.4f}")

    # 7. Submission Generation
    threshold = 0.7135451153926904

    if final_val_auc > threshold:
        print(
            f"\nValidation metric exceeds threshold ({threshold}). Generating submission..."
        )

        # Generate predictions on test set
        rf_test_probs = model_rf.predict_rf_model(rf_model, X_test_rf)
        mlp_test_probs = model_mlp.predict_mlp_model(mlp_model, X_test_mlp)

        # Ensemble
        ensemble_test_probs = (w_rf * rf_test_probs) + (w_mlp * mlp_test_probs)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": ensemble_test_probs,
            }
        )

        # Verify format
        try:
            utils.verify_submission_format(submission_df)
        except Exception as e:
            print(f"Warning: Submission format verification check: {e}")

        # Save submission
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {final_val_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
