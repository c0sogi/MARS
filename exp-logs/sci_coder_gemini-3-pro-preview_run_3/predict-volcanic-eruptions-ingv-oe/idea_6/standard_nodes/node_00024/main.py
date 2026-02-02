import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

from library import config, utils, data_processor, model_trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration
    # ---------------------------------------------------------
    # Set random seeds for reproducibility
    utils.seed_everything(config.SEED)

    # Adjust configuration for a fast baseline execution
    # Reducing n_estimators ensures the script completes within the time limit
    # while still leveraging the ensemble's power.
    config.LGBM_PARAMS["n_estimators"] = 5000
    config.XGB_PARAMS["n_estimators"] = 5000

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading datasets...")
    # Load training and validation data from cache
    X_train_split, y_train_split, X_val_split, y_val_split = (
        data_processor.get_train_val_datasets(load_cached_data=True)
    )

    # Load test data from cache
    X_test, test_ids = data_processor.get_test_dataset(load_cached_data=True)

    # Merge splits to perform full Stratified K-Fold CV
    X = pd.concat([X_train_split, X_val_split], axis=0).reset_index(drop=True)
    y = pd.concat([y_train_split, y_val_split], axis=0).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Stratified K-Fold Cross-Validation
    # ---------------------------------------------------------
    # Create bins for stratification based on the continuous target
    num_bins = 15
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Arrays to store Out-Of-Fold (OOF) predictions and accumulated test predictions
    oof_preds = np.zeros(len(y))
    test_preds_accumulator = np.zeros(len(X_test))

    print(f"Starting {config.N_FOLDS}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        # Split data for this fold
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        # --- Train LightGBM ---
        # model_trainer handles the specific parameters and early stopping
        lgbm_model = model_trainer.train_lgbm(X_tr, y_tr, X_va, y_va)
        p_lgbm_val = lgbm_model.predict(X_va)
        p_lgbm_test = lgbm_model.predict(X_test)

        # --- Train XGBoost ---
        # XGBoost is configured to use GPU if available via config.XGB_PARAMS
        xgb_model = model_trainer.train_xgboost(X_tr, y_tr, X_va, y_va)
        p_xgb_val = xgb_model.predict(X_va)
        p_xgb_test = xgb_model.predict(X_test)

        # --- Ensemble (Simple Average) ---
        p_val_ensemble = (p_lgbm_val + p_xgb_val) / 2.0
        p_test_ensemble = (p_lgbm_test + p_xgb_test) / 2.0

        # Store results
        oof_preds[val_idx] = p_val_ensemble
        test_preds_accumulator += p_test_ensemble

    # ---------------------------------------------------------
    # 4. Validation Assessment & Failure Analysis
    # ---------------------------------------------------------
    # Calculate Global Mean Absolute Error
    final_mae = mean_absolute_error(y, oof_preds)

    # Print the required metric string
    print(f"Final Validation Metric: {final_mae}")

    print("\n--- Failure Analysis ---")
    # Calculate absolute error magnitude for each sample
    errors = np.abs(y - oof_preds)
    error_series = pd.Series(errors, index=X.index)

    # Calculate correlation between input features and error magnitude
    # This helps identify which features are associated with high prediction errors
    correlations = X.corrwith(error_series).abs().sort_values(ascending=False)

    print("Top 10 Features correlated with Error Magnitude:")
    print(correlations.head(10))

    # ---------------------------------------------------------
    # 5. Conditional Submission
    # ---------------------------------------------------------
    threshold = 2739761.2592384242

    if final_mae < threshold:
        print(
            f"\nValidation metric {final_mae} is better than threshold {threshold}. Generating submission..."
        )

        # Average the accumulated test predictions
        final_test_preds = test_preds_accumulator / config.N_FOLDS

        # Save submission
        utils.save_submission(final_test_preds, test_ids)
    else:
        print(
            f"\nValidation metric {final_mae} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
