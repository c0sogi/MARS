import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

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


def run_training(debug: bool = False, debug_size: int = 100):
    """
    Executes the full training pipeline:
    1. Loads and processes data.
    2. Performs 5-Fold Stratified CV on the training set to generate OOF predictions.
    3. Trains the Level 2 Meta-Learner on OOF predictions.
    4. Retrains Level 1 Base Learners using Validation-Guided strategies.
    5. Generates predictions for the Test set and creates the submission file.

    Args:
        debug (bool): If True, runs on a subset of data for quick verification.
        debug_size (int): Number of samples to use in debug mode.
    """
    logger = setup_logger("TrainingPipeline")
    set_seed(Config.RANDOM_SEED)

    logger.info(f"Starting training run. Debug={debug}")

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    # The data_loader handles caching and preprocessing
    data = load_and_process_data(
        load_cached_data=True, debug=debug, debug_size=debug_size
    )

    X_train_lex = data["train"]["lexical"]
    X_train_beh = data["train"]["behavioral"]
    X_train_sem = data["train"]["semantic"]
    X_train_meta = data["train"]["metadata"]
    y_train = data["train"]["y"]

    # -------------------------------------------------------------------------
    # 2. Level 1: OOF Generation (5-Fold CV on Train Split)
    # -------------------------------------------------------------------------
    logger.info("Starting Level 1 OOF Generation (5-Fold CV)...")

    n_train = y_train.shape[0]
    n_folds = 5
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=Config.RANDOM_SEED
    )

    # Placeholders for OOF predictions
    # Columns: [Lexical, Behavioral, SemanticXGB, SemanticRF, Metadata]
    oof_preds = np.zeros((n_train, 5))

    # Iterate folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_train), y_train)):
        logger.info(f"Processing Fold {fold + 1}/{n_folds}...")

        # Slice data for this fold
        # Lexical (Sparse)
        X_tr_lex = X_train_lex[train_idx]
        X_va_lex = X_train_lex[val_idx]

        # Behavioral (Sparse)
        X_tr_beh = X_train_beh[train_idx]
        X_va_beh = X_train_beh[val_idx]

        # Semantic (Dense)
        X_tr_sem = X_train_sem[train_idx]
        X_va_sem = X_train_sem[val_idx]

        # Metadata (Dense)
        X_tr_meta = X_train_meta[train_idx]
        X_va_meta = X_train_meta[val_idx]

        # Targets
        y_tr = y_train[train_idx]
        # y_va = y_train[val_idx] # Not needed for prediction, only for scoring if desired

        # --- Train Base Learners ---

        # 1. Lexical Bagger
        model_lex = LexicalBagger()
        model_lex.fit(X_tr_lex, X_tr_meta, y_tr)
        oof_preds[val_idx, 0] = model_lex.predict_proba(X_va_lex, X_va_meta)

        # 2. Community Bagger
        model_comm = CommunityBagger()
        model_comm.fit(X_tr_beh, X_tr_meta, y_tr)
        oof_preds[val_idx, 1] = model_comm.predict_proba(X_va_beh, X_va_meta)

        # 3. Semantic Booster (XGB)
        # Note: Inside CV, we use the fold's validation set for early stopping if desired,
        # but to keep OOF consistent with standard stacking, we often just fit.
        # However, since we have a 'scale_pos_weight' and fixed rounds in config,
        # we can fit without internal early stopping for the OOF generation
        # OR use the fold validation set. Using fold val set is safer.
        model_sem_xgb = SemanticBooster()
        model_sem_xgb.fit(
            X_tr_sem, X_tr_meta, y_tr, eval_set=(X_va_sem, X_va_meta, y_train[val_idx])
        )
        oof_preds[val_idx, 2] = model_sem_xgb.predict_proba(X_va_sem, X_va_meta)

        # 4. Semantic Bagger (RF)
        model_sem_rf = SemanticBagger()
        model_sem_rf.fit(X_tr_sem, X_tr_meta, y_tr)
        oof_preds[val_idx, 3] = model_sem_rf.predict_proba(X_va_sem, X_va_meta)

        # 5. Metadata Anchor
        model_meta = MetadataAnchor()
        model_meta.fit(X_tr_meta, y_tr)
        oof_preds[val_idx, 4] = model_meta.predict_proba(X_va_meta)

    # Print OOF Scores
    model_names = [
        "LexicalBagger",
        "CommunityBagger",
        "SemanticBooster",
        "SemanticBagger",
        "MetadataAnchor",
    ]
    logger.info("--- Level 1 OOF AUC Scores ---")
    for i, name in enumerate(model_names):
        auc = roc_auc_score(y_train, oof_preds[:, i])
        print(f"{name}: {auc}")

    # -------------------------------------------------------------------------
    # 3. Level 2: Train Meta-Learner
    # -------------------------------------------------------------------------
    logger.info("Training Level 2 Meta-Learner on OOF predictions...")
    meta_learner = StackingMetaLearner()
    meta_learner.fit(oof_preds, y_train)

    # Check Meta-Learner OOF performance (approximate, usually biased on training data)
    meta_oof_probs = meta_learner.predict_proba(oof_preds)
    meta_auc = roc_auc_score(y_train, meta_oof_probs)
    print(f"Meta-Learner OOF AUC: {meta_auc}")

    # -------------------------------------------------------------------------
    # 4. Final Retraining (Validation-Guided) & Test Prediction
    # -------------------------------------------------------------------------
    logger.info("Retraining Base Learners for Test Prediction...")

    # Prepare Test Data
    X_test_lex = data["test"]["lexical"]
    X_test_beh = data["test"]["behavioral"]
    X_test_sem = data["test"]["semantic"]
    X_test_meta = data["test"]["metadata"]
    n_test = X_test_meta.shape[0]

    # Prepare Full Training Data (Train + Val) for RF/Linear models
    # We concatenate the 'train' and 'val' splits from metadata
    X_full_lex = sp.vstack([data["train"]["lexical"], data["val"]["lexical"]])
    X_full_beh = sp.vstack([data["train"]["behavioral"], data["val"]["behavioral"]])
    X_full_sem = np.vstack([data["train"]["semantic"], data["val"]["semantic"]])
    X_full_meta = np.vstack([data["train"]["metadata"], data["val"]["metadata"]])
    y_full = np.concatenate([data["train"]["y"], data["val"]["y"]])

    # Prepare Data for XGBoost (Train split for training, Val split for Early Stopping)
    # We use the raw 'train' and 'val' dictionaries directly
    X_train_sem_xgb = data["train"]["semantic"]
    X_train_meta_xgb = data["train"]["metadata"]
    y_train_xgb = data["train"]["y"]

    eval_set_xgb = (data["val"]["semantic"], data["val"]["metadata"], data["val"]["y"])

    # Matrix to store Level 1 Test Predictions
    test_level1_preds = np.zeros((n_test, 5))

    # --- Retrain and Predict ---

    # 1. Lexical Bagger (Full Data)
    logger.info("Retraining LexicalBagger on Full Data...")
    model_lex_full = LexicalBagger()
    model_lex_full.fit(X_full_lex, X_full_meta, y_full)
    test_level1_preds[:, 0] = model_lex_full.predict_proba(X_test_lex, X_test_meta)

    # 2. Community Bagger (Full Data)
    logger.info("Retraining CommunityBagger on Full Data...")
    model_comm_full = CommunityBagger()
    model_comm_full.fit(X_full_beh, X_full_meta, y_full)
    test_level1_preds[:, 1] = model_comm_full.predict_proba(X_test_beh, X_test_meta)

    # 3. Semantic Booster (Train Data + Val Early Stopping)
    logger.info("Retraining SemanticBooster with Global Validation Stopping...")
    model_sem_xgb_full = SemanticBooster()
    model_sem_xgb_full.fit(
        X_train_sem_xgb, X_train_meta_xgb, y_train_xgb, eval_set=eval_set_xgb
    )
    test_level1_preds[:, 2] = model_sem_xgb_full.predict_proba(X_test_sem, X_test_meta)

    # 4. Semantic Bagger (Full Data)
    logger.info("Retraining SemanticBagger on Full Data...")
    model_sem_rf_full = SemanticBagger()
    model_sem_rf_full.fit(X_full_sem, X_full_meta, y_full)
    test_level1_preds[:, 3] = model_sem_rf_full.predict_proba(X_test_sem, X_test_meta)

    # 5. Metadata Anchor (Full Data)
    logger.info("Retraining MetadataAnchor on Full Data...")
    model_meta_full = MetadataAnchor()
    model_meta_full.fit(X_full_meta, y_full)
    test_level1_preds[:, 4] = model_meta_full.predict_proba(X_test_meta)

    # -------------------------------------------------------------------------
    # 5. Generate Final Submission
    # -------------------------------------------------------------------------
    logger.info("Generating Final Predictions with Meta-Learner...")
    final_probs = meta_learner.predict_proba(test_level1_preds)

    # Load Test IDs for submission file
    df_test = pd.read_parquet(Config.TEST_PATH)
    if debug:
        df_test = df_test.iloc[:debug_size]

    submission_df = pd.DataFrame(
        {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: final_probs}
    )

    # Save
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
    logger.info("Training pipeline completed successfully.")
