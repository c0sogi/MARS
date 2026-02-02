import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided library files
from library.config import SEED, N_FOLDS, VAL_PATH, TARGET_COL
from library.utils import set_seed
from library.models import ModelRegistry
from library.pipeline import StackingTrainer


def main():
    # 1. Setup
    set_seed(SEED)
    print("Initializing Stacking Trainer...")
    trainer = StackingTrainer()

    # 2. Load Data
    # We load cached data to save time as per instructions
    print("Loading data...")
    trainer.load_data(load_cached_data=True)

    # Access data components
    data = trainer.data
    y_train = data["y_train"]
    y_val = data["y_val"]

    # Map models to their feature views
    model_map = trainer.model_feature_map
    model_names = list(model_map.keys())

    # 3. Validation Phase (Manual implementation to ensure strict hold-out evaluation)
    # We will:
    #   a. Generate OOF predictions on TRAIN set (via CV) to train the Meta-Learner
    #   b. Train Base Models on the full TRAIN set
    #   c. Predict on the VAL set using Base Models -> Meta-Learner

    print("\n--- Starting Validation Phase ---")

    # 3a. Generate Train OOFs
    print(f"Generating OOF predictions on Training set ({len(y_train)} samples)...")
    train_oof = np.zeros((len(y_train), len(model_names)))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Pre-retrieve train features to avoid repeated dictionary lookups
    X_train_views = {name: data[f"X_train_{model_map[name]}"] for name in model_names}

    for fold, (t_idx, v_idx) in enumerate(skf.split(np.zeros(len(y_train)), y_train)):
        y_t, y_v = y_train[t_idx], y_train[v_idx]

        for i, name in enumerate(model_names):
            # Get fresh model
            model = ModelRegistry.create_base_models()[name]

            # Get features for this view
            X_view = X_train_views[name]
            X_t, X_v = X_view[t_idx], X_view[v_idx]

            # Fit and Predict
            model.fit(X_t, y_t)
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_v)[:, 1]
            else:
                preds = model.predict(X_v)

            train_oof[v_idx, i] = preds

    # 3b. Train Meta-Learner on Train OOFs
    print("Training Meta-Learner on Train OOFs...")
    meta_learner = ModelRegistry.get_meta_learner()
    meta_learner.fit(train_oof, y_train)

    # 3c. Train Base Models on Full Train and Predict Val
    print("Training Base Models on Full Train and Predicting Validation set...")
    val_L1_preds = np.zeros((len(y_val), len(model_names)))

    X_val_views = {name: data[f"X_val_{model_map[name]}"] for name in model_names}

    for i, name in enumerate(model_names):
        model = ModelRegistry.create_base_models()[name]
        X_t = X_train_views[name]
        X_v = X_val_views[name]

        # Fit on full training set
        model.fit(X_t, y_train)

        # Predict on validation set
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_v)[:, 1]
        else:
            preds = model.predict(X_v)

        val_L1_preds[:, i] = preds

    # 3d. Final Validation Prediction
    val_final_preds = meta_learner.predict_proba(val_L1_preds)[:, 1]

    # 4. Metric Calculation
    final_metric = roc_auc_score(y_val, val_final_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Load raw validation data to get feature names for correlation
    val_df = pd.read_parquet(VAL_PATH)

    # Calculate error
    errors = np.abs(y_val - val_final_preds)

    # Select numerical columns for correlation analysis
    # We use the raw dataframe columns that are numerical
    numeric_cols = val_df.select_dtypes(include=["number"]).columns.tolist()
    # Exclude target if present
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]

    print("Top correlations between Error and Features:")
    correlations = []
    for col in numeric_cols:
        # Handle NaNs just in case, though parquet should be clean or we use imputed logic
        # Simple dropna for correlation check
        valid_idx = ~val_df[col].isna()
        if valid_idx.sum() < 2:
            continue

        feat_vals = val_df.loc[valid_idx, col]
        err_vals = errors[valid_idx]

        if len(np.unique(feat_vals)) < 2:
            continue

        corr, _ = pearsonr(feat_vals, err_vals)
        correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Logic
    THRESHOLD = 0.7085870249842536

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) > Threshold ({THRESHOLD}). Proceeding to submission."
        )

        # Retrain on Train + Val (handled by pipeline's retrain_final)
        # Note: pipeline.retrain_final handles the specific logic for XGB (Train w/ Val stopping) vs others (Train+Val)
        trainer.retrain_final()

        # Generate Submission
        trainer.generate_submission()

    else:
        print(
            f"\nValidation metric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
