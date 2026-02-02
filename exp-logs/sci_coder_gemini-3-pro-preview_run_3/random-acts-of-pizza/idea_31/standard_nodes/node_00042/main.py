import sys
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_and_process_data
from library.models import (
    LexicalBagger,
    CommunityBagger,
    SemanticBooster,
    SemanticBagger,
    MetadataAnchor,
    StackingMetaLearner,
)


def main():
    # 1. Setup
    logger = setup_logger("RunFile")
    set_seed(Config.RANDOM_SEED)

    # 2. Load Data
    # Use cached data for speed
    data = load_and_process_data(load_cached_data=True)

    # Extract Train Data
    X_train_lex = data["train"]["lexical"]
    X_train_beh = data["train"]["behavioral"]
    X_train_sem = data["train"]["semantic"]
    X_train_meta = data["train"]["metadata"]
    y_train = data["train"]["y"]

    # Extract Val Data
    X_val_lex = data["val"]["lexical"]
    X_val_beh = data["val"]["behavioral"]
    X_val_sem = data["val"]["semantic"]
    X_val_meta = data["val"]["metadata"]
    y_val = data["val"]["y"]

    # Extract Test Data
    X_test_lex = data["test"]["lexical"]
    X_test_beh = data["test"]["behavioral"]
    X_test_sem = data["test"]["semantic"]
    X_test_meta = data["test"]["metadata"]

    # 3. Level 1: OOF Generation on Train
    logger.info("Generating OOF predictions on Training set...")
    n_train = y_train.shape[0]
    n_folds = 5
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=Config.RANDOM_SEED
    )

    # Placeholders for OOF predictions
    # Columns: [Lexical, Behavioral, SemanticXGB, SemanticRF, Metadata]
    oof_preds = np.zeros((n_train, 5))

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_train), y_train)):
        # Slice data for this fold
        X_tr_lex, X_va_lex = X_train_lex[train_idx], X_train_lex[val_idx]
        X_tr_beh, X_va_beh = X_train_beh[train_idx], X_train_beh[val_idx]
        X_tr_sem, X_va_sem = X_train_sem[train_idx], X_train_sem[val_idx]
        X_tr_meta, X_va_meta = X_train_meta[train_idx], X_train_meta[val_idx]
        y_tr = y_train[train_idx]

        # Train & Predict Base Learners

        # 1. Lexical Bagger
        model_lex = LexicalBagger().fit(X_tr_lex, X_tr_meta, y_tr)
        oof_preds[val_idx, 0] = model_lex.predict_proba(X_va_lex, X_va_meta)

        # 2. Community Bagger
        model_comm = CommunityBagger().fit(X_tr_beh, X_tr_meta, y_tr)
        oof_preds[val_idx, 1] = model_comm.predict_proba(X_va_beh, X_va_meta)

        # 3. Semantic Booster (XGB)
        # Use internal fold validation for early stopping
        model_sem_xgb = SemanticBooster().fit(
            X_tr_sem, X_tr_meta, y_tr, eval_set=(X_va_sem, X_va_meta, y_train[val_idx])
        )
        oof_preds[val_idx, 2] = model_sem_xgb.predict_proba(X_va_sem, X_va_meta)

        # 4. Semantic Bagger (RF)
        model_sem_rf = SemanticBagger().fit(X_tr_sem, X_tr_meta, y_tr)
        oof_preds[val_idx, 3] = model_sem_rf.predict_proba(X_va_sem, X_va_meta)

        # 5. Metadata Anchor
        model_meta = MetadataAnchor().fit(X_tr_meta, y_tr)
        oof_preds[val_idx, 4] = model_meta.predict_proba(X_va_meta)

    # 4. Train Meta-Learner
    logger.info("Training Meta-Learner on OOF predictions...")
    meta_learner = StackingMetaLearner().fit(oof_preds, y_train)

    # 5. Validation on Hold-out Set
    logger.info("Validating on Hold-out Validation set...")

    # Train Base Learners on full Train set to predict on Val
    # For XGBoost, split Train slightly to provide an unbiased eval_set for early stopping
    X_tr_xgb, X_es_xgb, y_tr_xgb, y_es_xgb = train_test_split(
        np.arange(n_train),
        y_train,
        test_size=0.1,
        stratify=y_train,
        random_state=Config.RANDOM_SEED,
    )

    val_level1_preds = np.zeros((y_val.shape[0], 5))

    # 1. Lexical
    model_lex_val = LexicalBagger().fit(X_train_lex, X_train_meta, y_train)
    val_level1_preds[:, 0] = model_lex_val.predict_proba(X_val_lex, X_val_meta)

    # 2. Community
    model_comm_val = CommunityBagger().fit(X_train_beh, X_train_meta, y_train)
    val_level1_preds[:, 1] = model_comm_val.predict_proba(X_val_beh, X_val_meta)

    # 3. Semantic XGB (using internal split for ES)
    model_sem_xgb_val = SemanticBooster().fit(
        X_train_sem[X_tr_xgb],
        X_train_meta[X_tr_xgb],
        y_train[X_tr_xgb],
        eval_set=(X_train_sem[X_es_xgb], X_train_meta[X_es_xgb], y_train[X_es_xgb]),
    )
    val_level1_preds[:, 2] = model_sem_xgb_val.predict_proba(X_val_sem, X_val_meta)

    # 4. Semantic RF
    model_sem_rf_val = SemanticBagger().fit(X_train_sem, X_train_meta, y_train)
    val_level1_preds[:, 3] = model_sem_rf_val.predict_proba(X_val_sem, X_val_meta)

    # 5. Metadata
    model_meta_val = MetadataAnchor().fit(X_train_meta, y_train)
    val_level1_preds[:, 4] = model_meta_val.predict_proba(X_val_meta)

    # Meta Prediction
    val_final_probs = meta_learner.predict_proba(val_level1_preds)

    # Metric
    val_auc = roc_auc_score(y_val, val_final_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")
    residuals = np.abs(y_val - val_final_probs)
    feature_names = Config.NUMERICAL_ALLOW_LIST + ["text_history_similarity"]

    print("Failure Analysis (Correlation with Error):")
    for i, feature in enumerate(feature_names):
        if i < X_val_meta.shape[1]:
            feat_values = X_val_meta[:, i]
            if np.std(feat_values) > 0:
                corr, _ = pearsonr(residuals, feat_values)
                print(f"  {feature}: {corr:.4f}")
            else:
                print(f"  {feature}: 0.0000 (Constant)")

    # 7. Submission
    threshold = 0.7085870249842536
    if val_auc > threshold:
        logger.info(
            f"Validation score {val_auc} > {threshold}. Proceeding to submission..."
        )

        # Prepare Full Data (Train + Val) for RF/Linear models
        X_full_lex = sp.vstack([X_train_lex, X_val_lex])
        X_full_beh = sp.vstack([X_train_beh, X_val_beh])
        X_full_sem = np.vstack([X_train_sem, X_val_sem])
        X_full_meta = np.vstack([X_train_meta, X_val_meta])
        y_full = np.concatenate([y_train, y_val])

        n_test = X_test_meta.shape[0]
        test_level1_preds = np.zeros((n_test, 5))

        # Retrain Base Learners

        # 1. Lexical (Full Data)
        model_lex_full = LexicalBagger().fit(X_full_lex, X_full_meta, y_full)
        test_level1_preds[:, 0] = model_lex_full.predict_proba(X_test_lex, X_test_meta)

        # 2. Community (Full Data)
        model_comm_full = CommunityBagger().fit(X_full_beh, X_full_meta, y_full)
        test_level1_preds[:, 1] = model_comm_full.predict_proba(X_test_beh, X_test_meta)

        # 3. Semantic XGB ("The Fix": Train on Train, Stop on Val)
        model_sem_xgb_full = SemanticBooster().fit(
            X_train_sem, X_train_meta, y_train, eval_set=(X_val_sem, X_val_meta, y_val)
        )
        test_level1_preds[:, 2] = model_sem_xgb_full.predict_proba(
            X_test_sem, X_test_meta
        )

        # 4. Semantic RF (Full Data)
        model_sem_rf_full = SemanticBagger().fit(X_full_sem, X_full_meta, y_full)
        test_level1_preds[:, 3] = model_sem_rf_full.predict_proba(
            X_test_sem, X_test_meta
        )

        # 5. Metadata (Full Data)
        model_meta_full = MetadataAnchor().fit(X_full_meta, y_full)
        test_level1_preds[:, 4] = model_meta_full.predict_proba(X_test_meta)

        # Meta Prediction
        test_final_probs = meta_learner.predict_proba(test_level1_preds)

        # Save Submission
        df_test = pd.read_parquet(Config.TEST_PATH)
        submission_df = pd.DataFrame(
            {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: test_final_probs}
        )
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.warning(
            f"Validation score {val_auc} <= {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
