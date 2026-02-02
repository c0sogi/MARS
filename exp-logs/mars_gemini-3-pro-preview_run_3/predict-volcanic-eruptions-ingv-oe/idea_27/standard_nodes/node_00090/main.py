import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import os
import sys

# Import provided library modules
from library import config, utils, features, model

# Set seeds for reproducibility
np.random.seed(config.SEED)


def run():
    # ==========================================
    # 1. Data Loading & Feature Generation
    # ==========================================
    print("Loading metadata...")
    train_meta = utils.load_metadata("train")
    val_meta = utils.load_metadata("val")

    print("Generating features for training set...")
    # Generate features for the training set
    # This will use caching if available (train_features.parquet)
    df_train = features.create_feature_matrix(train_meta, split_name="train")

    print("Generating features for validation set...")
    # Generate features for the hold-out validation set (val_features.parquet)
    df_val = features.create_feature_matrix(val_meta, split_name="val")

    # Prepare X and y for Training
    target_col = "time_to_eruption"
    drop_cols = ["segment_id", "time_to_eruption"]

    # Ensure target exists
    if target_col not in df_train.columns:
        raise ValueError("Target column missing in training data")

    X_train_full = df_train.drop(columns=drop_cols)
    y_train_full = df_train[target_col].values

    # Prepare X and y for Validation (Hold-out)
    X_val_holdout = df_val.drop(columns=drop_cols)
    y_val_holdout = df_val[target_col].values

    # ==========================================
    # 2. Model Training (Ensemble)
    # ==========================================
    # We use the EnsembleManager class to hold the models,
    # but we implement the training loop here to strictly respect the train/val split.
    ensemble = model.EnsembleManager()

    # Stratified K-Fold on the Training Set
    # We bin the target to allow for stratified splitting
    num_bins = 10
    # Adjust bins if dataset is small (safety check)
    if len(y_train_full) < num_bins:
        num_bins = max(2, len(y_train_full) // 2)

    y_bins = pd.qcut(y_train_full, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    print(
        f"Starting training on {len(X_train_full)} samples using {config.N_FOLDS}-Fold CV..."
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_bins)):
        print(f"\n--- Training Fold {fold + 1} / {config.N_FOLDS} ---")

        X_tr = X_train_full.iloc[train_idx]
        y_tr = y_train_full[train_idx]
        X_va = X_train_full.iloc[val_idx]
        y_va = y_train_full[val_idx]

        # Train model for this fold
        # Note: We use the fold's validation set for early stopping
        trained_model, _, _ = model.train_lgbm_fold(
            X_tr, y_tr, X_va, y_va, ensemble.params
        )

        ensemble.models.append(trained_model)

    # ==========================================
    # 3. Validation & Metrics
    # ==========================================
    print("\nEvaluating on Hold-out Validation Set...")

    if len(ensemble.models) == 0:
        raise RuntimeError("No models were trained.")

    # Generate predictions on the hold-out validation set
    val_preds = np.zeros(len(X_val_holdout))

    for i, m in enumerate(ensemble.models):
        # Predict using the best iteration found during training
        p = m.predict(X_val_holdout, num_iteration=m.best_iteration)
        val_preds += p

    # Average predictions
    val_preds /= len(ensemble.models)

    # Calculate and Print Metric
    mae = utils.calculate_mae(y_val_holdout, val_preds)
    print(f"Final Validation Metric: {mae}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Calculate absolute errors
    errors = np.abs(y_val_holdout - val_preds)

    # Create a DataFrame for correlation analysis
    analysis_df = X_val_holdout.copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlation between features and error magnitude
    # We drop columns with NaN correlations (constant features)
    correlations = (
        analysis_df.corrwith(analysis_df["error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 10 features correlated with Error Magnitude:")
    print(correlations.head(10))

    # ==========================================
    # 5. Submission
    # ==========================================
    threshold = 2617304.0647319085

    if mae < threshold:
        print(
            f"\nValidation MAE ({mae}) is below threshold ({threshold}). Generating submission..."
        )
        # Use the ensemble's method to generate submission
        # This handles test metadata loading, feature generation, and saving
        ensemble.predict_average()
    else:
        print(
            f"\nValidation MAE ({mae}) is NOT below threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
