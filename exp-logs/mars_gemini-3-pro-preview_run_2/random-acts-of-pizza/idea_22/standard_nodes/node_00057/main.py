import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import library modules
import library.config as config
from library.utils import set_seed
from library.data_loader import (
    load_labeled_data,
    extract_text_data,
    extract_numeric_data,
)
from library.feature_engineering import generate_embeddings, prepare_design_matrix
from library.model_factory import build_ensemble_pipeline
from library.trainer import tune_component
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    set_seed(config.SEED)
    print("Starting execution...")

    # 2. Data Loading & Feature Engineering
    print("Loading labeled data...")
    df_train = load_labeled_data(load_cached_data=True)
    y = df_train["requester_received_pizza"].values.astype(int)

    print("Extracting features...")
    # Text Features (SBERT Embeddings)
    text_data = extract_text_data(df_train)
    embeddings = generate_embeddings(text_data, "train", load_cached_data=True)

    # Numeric Features
    numeric_data = extract_numeric_data(df_train)

    # Prepare Design Matrix
    X, metadata_start_idx = prepare_design_matrix(embeddings, numeric_data)
    print(f"Design Matrix Shape: {X.shape}")

    # 3. Cross-Validation Loop
    print(f"Starting {config.N_FOLDS}-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    oof_preds = np.zeros(len(y))

    # Create a template pipeline to access the preprocessor for tuning
    dummy_pipeline = build_ensemble_pipeline(metadata_start_idx)
    preprocessor_template = dummy_pipeline.named_steps["preprocessor"]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Processing Fold {fold + 1}/{config.N_FOLDS}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- Hyperparameter Tuning ---
        # Tune Logistic Regression (Log-Likelihood)
        best_lr_params, _ = tune_component(
            X_train, y_train, preprocessor_template, "lr", config.LR_GRID
        )

        # Tune SVM (Hinge Loss)
        best_svm_params, _ = tune_component(
            X_train, y_train, preprocessor_template, "svm", config.SVM_GRID
        )

        # Tune Ridge Classifier (Squared Error)
        best_ridge_params, _ = tune_component(
            X_train, y_train, preprocessor_template, "ridge", config.RIDGE_GRID
        )

        # --- Model Training ---
        # Build the Heterogeneous Linear Ensemble with optimized parameters
        fold_pipeline = build_ensemble_pipeline(
            metadata_start_idx,
            lr_params=best_lr_params,
            svm_params=best_svm_params,
            ridge_params=best_ridge_params,
        )

        # Fit on the fold's training data
        fold_pipeline.fit(X_train, y_train)

        # Save the trained model for inference
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        model_path = os.path.join(config.CACHE_DIR, f"model_fold_{fold}.joblib")
        joblib.dump(fold_pipeline, model_path)

        # --- Validation Prediction ---
        # Predict probabilities for the validation set
        y_pred_proba = fold_pipeline.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = y_pred_proba

    # 4. Metric Calculation
    final_auc = roc_auc_score(y, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error magnitude
    errors = np.abs(y - oof_preds)

    # Identify valid numeric columns (handling potential missing ones)
    valid_cols = [c for c in config.NUMERIC_FEATURES if c in df_train.columns]

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(numeric_data, columns=valid_cols)
    analysis_df["error_magnitude"] = errors

    # Calculate correlations between features and error magnitude
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")
    correlations_abs = correlations.abs().sort_values(ascending=False)

    print("Top 5 Features Correlated with Error Magnitude:")
    print(correlations_abs.head(5))

    # 6. Submission
    threshold = 0.7141749705260098
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({threshold}). Proceeding to inference."
        )
        run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({final_auc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
