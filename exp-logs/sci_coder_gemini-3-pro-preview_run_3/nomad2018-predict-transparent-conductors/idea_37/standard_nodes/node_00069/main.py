import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from library.regression_model import LogTransformedXGBoost, prepare_data

# Constants
SUBMISSION_DIR = "./submission"
THRESHOLD = 0.05095


def main():
    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print("--- Data Preparation ---")
    # Load data using the library function which handles caching and cleaning
    # We load cached data to save time as per instructions
    # Cite debug_lesson_3: Force regeneration of features to avoid loading stale/empty cache
    train_ids, X_train, y_train_dict = prepare_data("train", load_cached_data=False)
    val_ids, X_val, y_val_dict = prepare_data("val", load_cached_data=False)
    test_ids, X_test, _ = prepare_data("test", load_cached_data=False)

    # Align columns: Train is the source of truth
    train_cols = X_train.columns.tolist()

    # Align Validation
    for col in train_cols:
        if col not in X_val.columns:
            X_val[col] = 0.0
    X_val = X_val[train_cols]

    # Align Test
    for col in train_cols:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_test = X_test[train_cols]

    print(f"Training features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")
    print(f"Test features shape: {X_test.shape}")

    print("\n--- Model Training ---")
    # Initialize models
    # Using settings optimized for generalization on this dataset size
    # n_jobs=-1 uses all cores for speed
    model_formation = LogTransformedXGBoost(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.6,
        colsample_bytree=0.6,
        n_jobs=-1,
        random_state=42,
    )

    model_bandgap = LogTransformedXGBoost(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.6,
        colsample_bytree=0.6,
        n_jobs=-1,
        random_state=42,
    )

    # Train Formation Energy
    print("Training Formation Energy Model...")
    rmsle_formation = model_formation.train(
        X_train,
        y_train_dict["formation_energy_ev_natom"],
        X_val,
        y_val_dict["formation_energy_ev_natom"],
        target_name="Formation Energy",
    )

    # Train Bandgap Energy
    print("Training Bandgap Energy Model...")
    rmsle_bandgap = model_bandgap.train(
        X_train,
        y_train_dict["bandgap_energy_ev"],
        X_val,
        y_val_dict["bandgap_energy_ev"],
        target_name="Bandgap Energy",
    )

    # Compute Final Metric
    final_metric = (rmsle_formation + rmsle_bandgap) / 2
    print(f"Final Validation Metric: {final_metric}")

    print("\n--- Failure Analysis ---")
    # Generate predictions on validation set for analysis
    # Note: The model.predict method returns values in original scale (inverse log applied)
    # But for RMSLE analysis we often look at log space errors.
    # Let's look at the error contribution in log space since the metric is RMSLE.

    # Get predictions in log space manually to compute error correlation
    # Accessing the underlying xgb model to get raw scores (which are log(1+y))
    z_pred_form = model_formation.model.predict(X_val)
    z_true_form = np.log1p(y_val_dict["formation_energy_ev_natom"])
    error_form = np.abs(z_pred_form - z_true_form)

    z_pred_band = model_bandgap.model.predict(X_val)
    z_true_band = np.log1p(y_val_dict["bandgap_energy_ev"])
    error_band = np.abs(z_pred_band - z_true_band)

    # Average error magnitude
    avg_error = (error_form + error_band) / 2

    # Correlation with features
    correlations = X_val.corrwith(avg_error).abs().sort_values(ascending=False)
    print("Top 10 features correlated with prediction error:")
    print(correlations.head(10))

    print("\n--- Submission Generation ---")
    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )

        # Predict on test set
        pred_formation = model_formation.predict(X_test)
        pred_bandgap = model_bandgap.predict(X_test)

        submission = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": pred_formation,
                "bandgap_energy_ev": pred_bandgap,
            }
        )

        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"Validation metric {final_metric} is NOT below threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
