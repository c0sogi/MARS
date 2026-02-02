import os
import sys
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_dataset, NUMERIC_COLS
from library.feature_text import SBERTEmbedder
from library.feature_topic import LDATopicExtractor
from library.feature_meta import MetadataScaler
from library.model_builder import create_classifier


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    logger = setup_logger("runfile")

    # 2. Load Data
    logger.info("Loading datasets...")
    # Load cached data if available to save time
    df_train_split, df_val_split, df_test = load_dataset(load_cached_data=True)

    # Combine train and val for Stratified K-Fold
    # We reset index to ensure iloc indexing works correctly during CV
    df_full_train = pd.concat([df_train_split, df_val_split], ignore_index=True)
    y_full = df_full_train["requester_received_pizza"].values.astype(int)

    logger.info(f"Full training set shape: {df_full_train.shape}")
    logger.info(f"Test set shape: {df_test.shape}")

    # 3. Pre-compute Static Embeddings (View 1: Semantic Content)
    logger.info("Retrieving SBERT embeddings...")
    embedder = SBERTEmbedder()

    emb_train_part = embedder.process_and_cache(
        df_train_split, Config.TRAIN_EMBEDDINGS_PATH, load_cached_data=True
    )
    emb_val_part = embedder.process_and_cache(
        df_val_split, Config.VAL_EMBEDDINGS_PATH, load_cached_data=True
    )
    emb_test = embedder.process_and_cache(
        df_test, Config.TEST_EMBEDDINGS_PATH, load_cached_data=True
    )

    # Stack train and val embeddings to match df_full_train
    X_text_full = np.vstack([emb_train_part, emb_val_part])

    # 4. Stratified Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
    )

    # Arrays to store predictions
    test_preds_sum = np.zeros(len(df_test))
    oof_preds = np.zeros(len(df_full_train))

    logger.info(f"Starting {Config.N_FOLDS}-Fold Stratified CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_text_full, y_full)):
        logger.info(f"--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # A. Data Slicing
        # Text Embeddings (Static)
        X_text_train = X_text_full[train_idx]
        X_text_val = X_text_full[val_idx]

        # DataFrame slices (Dynamic)
        df_fold_train = df_full_train.iloc[train_idx]
        df_fold_val = df_full_train.iloc[val_idx]
        y_fold_train = y_full[train_idx]
        y_fold_val = y_full[val_idx]

        # B. Dynamic Feature Extraction
        # View 2: User Persona (LDA)
        # Fit on fold train, transform fold val and test
        lda_extractor = LDATopicExtractor(
            n_components=Config.LDA_N_COMPONENTS,
            min_df=Config.LDA_MIN_DF,
            random_state=Config.LDA_RANDOM_STATE,
        )
        X_topic_train = lda_extractor.fit_transform(df_fold_train)
        X_topic_val = lda_extractor.transform(df_fold_val)
        X_topic_test = lda_extractor.transform(df_test)

        # View 3: Robust Metadata (RankGauss)
        meta_scaler = MetadataScaler(random_state=Config.RANDOM_SEED)
        X_meta_train = meta_scaler.fit_transform(df_fold_train)
        X_meta_val = meta_scaler.transform(df_fold_val)
        X_meta_test = meta_scaler.transform(df_test)

        # C. Feature Fusion
        X_train_fold = np.hstack([X_text_train, X_topic_train, X_meta_train])
        X_val_fold = np.hstack([X_text_val, X_topic_val, X_meta_val])
        X_test_fold = np.hstack([emb_test, X_topic_test, X_meta_test])

        # D. Hyperparameter Tuning (Grid Search)
        best_fold_auc = -1.0
        best_fold_model = None

        keys = Config.PARAM_GRID.keys()
        values = Config.PARAM_GRID.values()
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for params in param_combinations:
            clf = create_classifier(
                C=params["C"],
                class_weight=params["class_weight"],
                n_estimators=Config.BAGGING_N_ESTIMATORS,
                random_state=Config.RANDOM_SEED,
            )

            clf.fit(X_train_fold, y_fold_train)

            # Evaluate on validation fold
            y_pred_val = clf.predict_proba(X_val_fold)[:, 1]
            auc = roc_auc_score(y_fold_val, y_pred_val)

            if auc > best_fold_auc:
                best_fold_auc = auc
                best_fold_model = clf

        logger.info(f"Fold {fold + 1} Best AUC: {best_fold_auc:.6f}")

        # E. Inference
        # OOF Predictions
        oof_preds[val_idx] = best_fold_model.predict_proba(X_val_fold)[:, 1]

        # Test Predictions (Accumulate)
        fold_test_preds = best_fold_model.predict_proba(X_test_fold)[:, 1]
        test_preds_sum += fold_test_preds

    # 5. Global Evaluation
    global_auc = roc_auc_score(y_full, oof_preds)
    print(f"Final Validation Metric: {global_auc}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_full - oof_preds)

    # Correlate error with numerical features
    # Ensure we only use numeric columns present in the dataframe
    valid_numeric_cols = [c for c in NUMERIC_COLS if c in df_full_train.columns]

    if valid_numeric_cols:
        # Create a dataframe for correlation
        analysis_df = df_full_train[valid_numeric_cols].copy()
        analysis_df["error"] = errors

        correlations = (
            analysis_df.corr()["error"]
            .drop("error")
            .sort_values(ascending=False, key=abs)
        )

        print("\n--- Failure Analysis: Correlation with Error ---")
        print(correlations.head(5))
        print("------------------------------------------------\n")
    else:
        logger.warning("No numerical columns available for failure analysis.")

    # 7. Submission
    threshold = 0.7141749705260098
    if global_auc > threshold:
        logger.info(
            f"Validation metric ({global_auc}) > threshold ({threshold}). Generating submission..."
        )

        avg_test_preds = test_preds_sum / Config.N_FOLDS

        submission_df = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": avg_test_preds,
            }
        )

        Config.ensure_directories()
        submission_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
    else:
        logger.warning(
            f"Validation metric ({global_auc}) <= threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
