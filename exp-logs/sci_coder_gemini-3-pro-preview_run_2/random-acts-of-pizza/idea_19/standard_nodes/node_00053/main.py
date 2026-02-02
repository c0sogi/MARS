import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import get_logger, set_seed
from library.data_manager import load_dataset
from library.feature_extractor import generate_sbert_embeddings
from library.pipeline_manager import LPADFPipelineManager


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    logger = get_logger("RunFile")

    # 2. Load Data
    # load_dataset handles caching internally
    df_train, df_val, df_test = load_dataset()

    # 3. Generate/Load Embeddings
    # generate_sbert_embeddings handles caching internally
    train_emb, val_emb, test_emb = generate_sbert_embeddings(df_train, df_val, df_test)

    # 4. Initialize Pipeline Manager
    manager = LPADFPipelineManager()

    # 5. Merge Features
    logger.info("Merging metadata, text embeddings, and user history...")
    X_train_full = manager.merge_features(df_train, train_emb)
    y_train_full = df_train["requester_received_pizza"].values.astype(int)

    X_val_full = manager.merge_features(df_val, val_emb)
    y_val_full = df_val["requester_received_pizza"].values.astype(int)

    X_test_full = manager.merge_features(df_test, test_emb)

    # 6. Combine Train and Val for Stratified CV
    # We combine them to perform a fresh 5-fold split on the entire labeled dataset
    X_all = pd.concat([X_train_full, X_val_full], axis=0).reset_index(drop=True)
    y_all = np.concatenate([y_train_full, y_val_full], axis=0)

    logger.info(f"Combined Labeled Data Shape: {X_all.shape}")

    # 7. Stratified Cross-Validation Setup
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
    )

    oof_preds = np.zeros(len(X_all))
    test_preds = np.zeros(len(X_test_full))

    # Parameter Grid for GridSearchCV
    param_grid = {
        "classifier__estimator__C": Config.LR_C_RANGE,
        "classifier__estimator__class_weight": Config.LR_CLASS_WEIGHTS,
    }

    logger.info("Starting Cross-Validation Loop...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        # Split Data
        X_tr, X_va = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_va = y_all[train_idx], y_all[val_idx]

        # Create fresh pipeline for this fold
        pipeline = manager.create_pipeline()

        # Grid Search
        # Using 3-fold inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=4, verbose=0
        )

        grid_search.fit(X_tr, y_tr)

        best_model = grid_search.best_estimator_

        # Validation Prediction
        val_probs = best_model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_probs

        # Test Prediction (Accumulate)
        test_probs = best_model.predict_proba(X_test_full)[:, 1]
        test_preds += test_probs / Config.N_FOLDS

    # 8. Overall Evaluation
    final_metric = roc_auc_score(y_all, oof_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 9. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(y_all - oof_preds)

    # Select numerical features for correlation analysis
    # We use the columns defined in Config + derived text lengths if available
    analysis_df = X_all[Config.NUMERICAL_COLS].copy()
    analysis_df["error"] = errors

    # Compute correlations
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )

    print("\nTop Feature Correlations with Prediction Error:")
    print(correlations.head(5))

    # 10. Conditional Submission
    threshold = 0.7141749705260098
    if final_metric > threshold:
        logger.info(
            f"Metric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )
        manager.save_submission(df_test, test_preds)
    else:
        logger.warning(
            f"Metric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
