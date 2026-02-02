"""
Implementation of the Modality-Balanced Bagged Linear Ensemble execution script.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import PizzaDataLoader
from library.feature_engineering import TextEmbedder, TabularProcessor, FeatureFuser
from library.model_factory import ModelFactory


def main():
    # 1. Setup
    logger = setup_logger("runfile")
    set_seed(Config.SEED)

    logger.info("Initializing Modality-Balanced Bagged Linear Ensemble Pipeline...")

    # 2. Load Data
    # We load cached data if available to speed up execution
    data_loader = PizzaDataLoader()
    train_df, val_df, test_df = data_loader.load_data(load_cached_data=True)

    # 3. Feature Engineering
    # 3.1 Text Embeddings
    text_embedder = TextEmbedder()
    # These methods handle caching internally
    X_train_text = text_embedder.get_embeddings(train_df, "train")
    X_val_text = text_embedder.get_embeddings(val_df, "val")
    X_test_text = text_embedder.get_embeddings(test_df, "test")

    # 3.2 Numeric Metadata (RankGauss Transformed)
    tabular_processor = TabularProcessor()
    X_train_tab, X_val_tab, X_test_tab = tabular_processor.process_numeric_features(
        train_df, val_df, test_df
    )

    # 3.3 Targets
    y_train = train_df["requester_received_pizza"].values.astype(int)
    y_val = val_df["requester_received_pizza"].values.astype(int)

    # 4. CV-Bagging & Hyperparameter Tuning
    # We implement CV-Bagging: Tuning on CV and averaging predictions from fold models.
    # Cite solution_lesson_node_00027

    # Combine Train and Val for Stratified K-Fold
    X_dev_text = np.vstack([X_train_text, X_val_text])
    X_dev_tab = np.vstack([X_train_tab, X_val_tab])
    y_dev = np.hstack([y_train, y_val])

    # Create Dev DataFrame for Failure Analysis
    dev_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED
    )

    best_cv_auc = -1.0
    best_params = {}
    best_models = []  # Stores the list of models from the best CV run
    best_oof_preds = np.zeros(len(y_dev))

    Cs = Config.GRID_SEARCH_PARAMS["C"]
    class_weights = Config.GRID_SEARCH_PARAMS["class_weight"]

    # Fixed alpha=1.0 (Cite solution_lesson_node_00035)
    alpha = 1.0
    X_dev_fused = FeatureFuser.fuse(X_dev_text, X_dev_tab, alpha)
    X_test_fused = FeatureFuser.fuse(X_test_text, X_test_tab, alpha)

    logger.info(
        f"Starting CV Grid Search over {len(Cs)*len(class_weights)} combinations..."
    )

    for C in Cs:
        for cw in class_weights:
            fold_models = []
            oof_preds = np.zeros(len(y_dev))

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev_fused, y_dev)):
                X_fold_train, X_fold_val = X_dev_fused[train_idx], X_dev_fused[val_idx]
                y_fold_train, y_fold_val = y_dev[train_idx], y_dev[val_idx]

                model = ModelFactory.create_bagged_ensemble(
                    C=C,
                    class_weight=cw,
                    n_estimators=Config.BAGGING_N_ESTIMATORS,
                    random_state=Config.SEED,
                )

                model.fit(X_fold_train, y_fold_train)

                val_preds = model.predict_proba(X_fold_val)[:, 1]
                oof_preds[val_idx] = val_preds
                fold_models.append(model)

            # Calculate Overall CV AUC
            cv_auc = roc_auc_score(y_dev, oof_preds)

            if cv_auc > best_cv_auc:
                best_cv_auc = cv_auc
                best_params = {"C": C, "class_weight": cw}
                best_models = fold_models
                best_oof_preds = oof_preds

    # 5. Reporting
    logger.info("Grid Search Complete.")
    print(f"Final Validation Metric: {best_cv_auc}")
    logger.info(f"Best Parameters: {best_params}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis on Dev Set OOF Predictions...")

    # Calculate absolute error
    errors = np.abs(y_dev - best_oof_preds)

    # Correlate error with raw numeric features
    analysis_df = dev_df[Config.NUMERIC_FEATURES].copy()
    analysis_df = analysis_df.fillna(analysis_df.median())
    analysis_df["error"] = errors

    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )

    print("Correlation between Error and Features:")
    print(correlations)

    # 7. Submission Generation
    SUBMISSION_THRESHOLD = 0.7141749705260098

    if best_cv_auc > SUBMISSION_THRESHOLD:
        logger.info(
            f"CV AUC ({best_cv_auc}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Average predictions from all fold models (CV-Bagging)
        test_preds_sum = np.zeros(len(test_df))
        for model in best_models:
            test_preds_sum += model.predict_proba(X_test_fused)[:, 1]

        avg_test_preds = test_preds_sum / Config.N_SPLITS

        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": avg_test_preds,
            }
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.warning(
            f"CV AUC ({best_cv_auc}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
