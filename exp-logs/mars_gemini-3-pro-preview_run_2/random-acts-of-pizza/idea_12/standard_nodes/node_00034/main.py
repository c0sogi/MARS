import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import logging

# Import from provided library files
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_dataset
from library.feature_engineering import prepare_feature_matrices, TabularProcessor
from library.execution import train_model, generate_submission


def main():
    # 1. Setup
    # Ensure reproducibility
    set_seed(Config.RANDOM_SEED)
    logger = setup_logger("runfile")

    logger.info("Starting execution of runfile.py...")

    # 2. Load Data
    # load_dataset returns (df_train, df_val, df_test) containing metadata and raw features.
    # We use load_cached_data=True to leverage any pre-processed files.
    df_train, df_val, df_test = load_dataset(load_cached_data=True)

    # 3. Feature Engineering
    # prepare_feature_matrices returns numpy arrays for features and targets.
    # It handles text embedding generation (cached) and tabular scaling.
    # X_train/y_train correspond to the training set metadata.
    # X_val/y_val correspond to the hold-out validation set metadata.
    X_train, y_train, X_val, y_val, X_test = prepare_feature_matrices(
        df_train, df_val, df_test, load_cached_data=True
    )

    # 4. Train Model
    # We train on the Training set (X_train).
    # train_model performs Stratified 5-fold CV internally with Grid Search.
    # It returns a list of the best pipeline models from each fold.
    # We do NOT merge X_train and X_val here, ensuring X_val remains a strict hold-out
    # for the required "Final Validation Metric" calculation.
    logger.info(f"Training ensemble on Training set ({len(X_train)} samples)...")
    models = train_model(X_train, y_train)

    # 5. Inference on Hold-out Validation Set
    logger.info(
        f"Evaluating ensemble on Hold-out Validation set ({len(X_val)} samples)..."
    )

    # Collect predictions from all fold models (CV-Bagging)
    val_preds_matrix = np.zeros((X_val.shape[0], len(models)))
    for i, model in enumerate(models):
        # Predict probability of the positive class (1)
        # The pipeline handles differential scaling internally
        val_preds_matrix[:, i] = model.predict_proba(X_val)[:, 1]

    # Average predictions across the ensemble
    avg_val_preds = val_preds_matrix.mean(axis=1)

    # 6. Compute and Print Metric
    # This metric is computed on the strictly held-out validation set.
    val_auc = roc_auc_score(y_val, avg_val_preds)

    # REQUIRED OUTPUT FORMAT: Print full precision
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")

    # Calculate absolute error magnitude
    errors = np.abs(y_val - avg_val_preds)

    # Retrieve tabular feature names for interpretation.
    # We instantiate a temporary processor to get the list of numeric columns used.
    tp = TabularProcessor()
    feature_names = tp._get_numeric_cols(df_val)

    # Calculate correlations between tabular features and prediction error.
    # Tabular features start after the text embeddings in the concatenated matrix.
    tabular_start_idx = Config.EMBEDDING_DIM

    if X_val.shape[1] > tabular_start_idx:
        X_val_tabular = X_val[:, tabular_start_idx:]

        # Verify shape alignment between matrix columns and feature names
        if X_val_tabular.shape[1] == len(feature_names):
            correlations = []
            for i, name in enumerate(feature_names):
                feat_values = X_val_tabular[:, i]

                # Calculate correlation if variance exists
                if np.std(feat_values) > 1e-9 and np.std(errors) > 1e-9:
                    corr = np.corrcoef(feat_values, errors)[0, 1]
                    correlations.append((name, corr))
                else:
                    correlations.append((name, 0.0))

            # Sort by magnitude of correlation (descending)
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            print(
                "\n--- Failure Analysis: Top 10 Feature Correlations with Prediction Error ---"
            )
            for name, corr in correlations[:10]:
                print(f"{name:<50} {corr:.4f}")
        else:
            logger.warning(
                f"Feature count mismatch: X_val_tabular has {X_val_tabular.shape[1]} columns, but found {len(feature_names)} names."
            )
    else:
        logger.info("No tabular features found for failure analysis.")

    # 8. Submission
    # Generate submission only if the validation metric exceeds the specified threshold.
    threshold = 0.7141749705260098

    if val_auc > threshold:
        logger.info(
            f"Validation AUC ({val_auc:.6f}) exceeds threshold ({threshold:.6f}). Generating submission..."
        )
        generate_submission(models, X_test, df_test)
    else:
        logger.warning(
            f"Validation AUC ({val_auc:.6f}) does not exceed threshold ({threshold:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
