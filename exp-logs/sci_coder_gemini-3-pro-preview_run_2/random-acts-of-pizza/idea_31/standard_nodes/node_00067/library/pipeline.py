import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_datasets
from library.feature_engineering import FusionTransformer
from library.model import tune_ensemble_hyperparameters


def run_stratified_cv(debug_sample_size=None, load_cached_data=True):
    """
    Executes the Stratified Cross-Validation pipeline for the Tri-Backbone architecture.

    Args:
        debug_sample_size (int, optional): Number of samples to use for debugging.
        load_cached_data (bool): Whether to load embeddings from cache.
    """
    # 1. Setup
    set_seed(Config.SEED)
    logger = setup_logger("Pipeline", os.path.join(Config.WORKING_DIR, "pipeline.log"))

    logger.info("Initializing Tri-Backbone Asymmetric Early Fusion Pipeline...")

    # 2. Load Data
    # Returns dictionary with df_train, df_test, y_train, meta_train, meta_test,
    # train_embeddings (dict), test_embeddings (dict)
    data = load_datasets(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )

    y = data["y_train"]
    df_test = data["df_test"]

    # Prepare Raw Test Dictionary (static structure, transformed per fold)
    raw_test_dict = {
        "anchor": data["test_embeddings"]["anchor"],
        "aux1": data["test_embeddings"]["aux1"],
        "aux2": data["test_embeddings"]["aux2"],
        "meta": data["meta_test"],
    }

    # Access Raw Train Data for slicing
    raw_train_embeddings = data["train_embeddings"]
    raw_train_meta = data["meta_train"]

    # 3. Cross-Validation Initialization
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(y))
    test_preds_accum = np.zeros(len(df_test))

    logger.info(f"Starting {Config.N_FOLDS}-Fold Stratified CV...")

    # 4. CV Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        logger.info(f"--- Fold {fold+1}/{Config.N_FOLDS} ---")

        # A. Slice Data for this Fold
        X_train_dict = {
            "anchor": raw_train_embeddings["anchor"][train_idx],
            "aux1": raw_train_embeddings["aux1"][train_idx],
            "aux2": raw_train_embeddings["aux2"][train_idx],
            "meta": raw_train_meta[train_idx],
        }

        X_val_dict = {
            "anchor": raw_train_embeddings["anchor"][val_idx],
            "aux1": raw_train_embeddings["aux1"][val_idx],
            "aux2": raw_train_embeddings["aux2"][val_idx],
            "meta": raw_train_meta[val_idx],
        }

        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # B. Feature Engineering (Fit on Train, Transform All)
        logger.info("Applying FusionTransformer...")
        transformer = FusionTransformer()
        transformer.fit(X_train_dict)

        X_train_fused = transformer.transform(X_train_dict)
        X_val_fused = transformer.transform(X_val_dict)
        X_test_fused = transformer.transform(raw_test_dict)

        # C. Hyperparameter Tuning & Training
        # tune_ensemble_hyperparameters handles the grid search and returns the best fitted model
        best_model, best_params, best_score = tune_ensemble_hyperparameters(
            X_train=X_train_fused,
            y_train=y_train_fold,
            X_val=X_val_fused,
            y_val=y_val_fold,
            param_grid=Config.GRID_PARAMS,
            n_estimators=Config.N_BAGGING_ESTIMATORS,
            random_state=Config.SEED,
        )

        logger.info(f"Fold {fold+1} Best Validation AUC: {best_score}")

        # D. Inference
        # OOF Predictions
        val_probs = best_model.predict_proba(X_val_fused)[:, 1]
        oof_preds[val_idx] = val_probs

        # Test Predictions (Accumulate)
        test_probs = best_model.predict_proba(X_test_fused)[:, 1]
        test_preds_accum += test_probs

    # 5. Final Evaluation
    overall_auc = roc_auc_score(y, oof_preds)
    logger.info(f"Overall OOF AUC: {overall_auc}")

    # 6. Submission
    avg_test_preds = test_preds_accum / Config.N_FOLDS

    submission_df = pd.DataFrame(
        {
            "request_id": df_test["request_id"],
            "requester_received_pizza": avg_test_preds,
        }
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to {sub_path}")
