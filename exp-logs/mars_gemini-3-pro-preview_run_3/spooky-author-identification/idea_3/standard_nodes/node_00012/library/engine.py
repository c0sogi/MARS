import numpy as np
import pandas as pd
import torch
import gc
from library.config import Config
from library.utils import set_seed, save_submission, calculate_log_loss
from library.data_loader import load_data, get_cv_folds
from library.models_statistical import StatisticalExpert
from library.models_neural import TransformerExpert
from library.stacking import MetaLearner


def run_full_pipeline(debug=Config.DEBUG, n_folds=Config.N_FOLDS):
    """
    Orchestrates the end-to-end training and inference pipeline.

    1. Loads data.
    2. Performs Stratified K-Fold Cross Validation.
    3. Trains Base Models (Statistical, DeBERTa, RoBERTa) on each fold.
    4. Collects Out-Of-Fold (OOF) predictions.
    5. Trains a Meta-Learner on OOF predictions + Meta Features.
    6. Generates final predictions on the Test set.
    7. Saves the submission file.

    Args:
        debug (bool): If True, runs on a small subset of data.
        n_folds (int): Number of cross-validation folds.
    """
    # 1. Setup
    set_seed(Config.SEED)
    print("--- Starting Pipeline ---")
    print(f"Debug Mode: {debug}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    # We load cached data if available.
    train_df, val_df, test_df = load_data(load_cached_data=True, debug=debug)

    # Combine provided train and val splits for K-Fold CV to maximize data usage
    # Reset index is crucial for correct iloc indexing during CV
    full_train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    print(f"Full Training Data Shape: {full_train_df.shape}")
    print(f"Test Data Shape: {test_df.shape}")

    # 3. Initialization for Stacking
    n_samples = len(full_train_df)
    n_classes = 3

    # Arrays to store Out-Of-Fold predictions
    oof_preds_stat = np.zeros((n_samples, n_classes))
    oof_preds_deberta = np.zeros((n_samples, n_classes))
    oof_preds_roberta = np.zeros((n_samples, n_classes))

    # Lists to store Test predictions from each fold (to be averaged)
    test_preds_stat_folds = []
    test_preds_deberta_folds = []
    test_preds_roberta_folds = []

    # Target values for OOF evaluation
    y_target = full_train_df["target"].values

    # 4. Stratified K-Fold Loop
    skf = get_cv_folds(n_splits=n_folds, random_state=Config.SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, y_target)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")

        # Split Data
        X_train_fold = full_train_df.iloc[train_idx]
        X_val_fold = full_train_df.iloc[val_idx]

        # Extract Texts and Targets
        train_texts = X_train_fold["text"].values
        train_targets = X_train_fold["target"].values
        val_texts = X_val_fold["text"].values
        val_targets = X_val_fold["target"].values

        test_texts = test_df["text"].values

        # --- A. Statistical Expert ---
        print("Training Statistical Expert...")
        model_stat = StatisticalExpert()
        model_stat.fit(train_texts, train_targets)

        # OOF Inference
        val_probs_stat = model_stat.predict_proba(val_texts)
        oof_preds_stat[val_idx] = val_probs_stat

        # Test Inference
        test_probs_stat = model_stat.predict_proba(test_texts)
        test_preds_stat_folds.append(test_probs_stat)

        score_stat = calculate_log_loss(val_targets, val_probs_stat)
        print(f"Fold {fold + 1} Statistical LogLoss: {score_stat}")

        # --- B. Neural Expert: DeBERTa ---
        print(f"Training Neural Expert ({Config.MODEL_DEBERTA})...")
        model_deberta = TransformerExpert(model_name=Config.MODEL_DEBERTA)
        model_deberta.fit(train_texts, train_targets, val_texts, val_targets)

        # OOF Inference
        val_probs_deberta = model_deberta.predict_proba(val_texts)
        oof_preds_deberta[val_idx] = val_probs_deberta

        # Test Inference
        test_probs_deberta = model_deberta.predict_proba(test_texts)
        test_preds_deberta_folds.append(test_probs_deberta)

        score_deberta = calculate_log_loss(val_targets, val_probs_deberta)
        print(f"Fold {fold + 1} DeBERTa LogLoss: {score_deberta}")

        # Cleanup DeBERTa
        del model_deberta
        gc.collect()
        torch.cuda.empty_cache()

        # --- C. Neural Expert: RoBERTa ---
        print(f"Training Neural Expert ({Config.MODEL_ROBERTA})...")
        model_roberta = TransformerExpert(model_name=Config.MODEL_ROBERTA)
        model_roberta.fit(train_texts, train_targets, val_texts, val_targets)

        # OOF Inference
        val_probs_roberta = model_roberta.predict_proba(val_texts)
        oof_preds_roberta[val_idx] = val_probs_roberta

        # Test Inference
        test_probs_roberta = model_roberta.predict_proba(test_texts)
        test_preds_roberta_folds.append(test_probs_roberta)

        score_roberta = calculate_log_loss(val_targets, val_probs_roberta)
        print(f"Fold {fold + 1} RoBERTa LogLoss: {score_roberta}")

        # Cleanup RoBERTa
        del model_roberta
        gc.collect()
        torch.cuda.empty_cache()

    # 5. Aggregate Test Predictions (Level 0)
    print("\nAggregating Level 0 Test Predictions...")
    avg_test_stat = np.mean(test_preds_stat_folds, axis=0)
    avg_test_deberta = np.mean(test_preds_deberta_folds, axis=0)
    avg_test_roberta = np.mean(test_preds_roberta_folds, axis=0)

    # 6. Meta-Learner Training (Level 1)
    print("\nTraining Meta-Learner...")
    meta_learner = MetaLearner()

    # Prepare Training Data for Meta-Learner (OOFs + Meta Features)
    # Meta feature: log_char_len
    train_meta_features = full_train_df["log_char_len"].values
    X_meta_train = meta_learner.prepare_level1_features(
        [oof_preds_stat, oof_preds_deberta, oof_preds_roberta], train_meta_features
    )

    # Fit Meta-Learner
    meta_learner.fit(X_meta_train, y_target)

    # Evaluate Meta-Learner on OOF (Approximation of performance)
    oof_meta_probs = meta_learner.predict_proba(X_meta_train)
    oof_score = calculate_log_loss(y_target, oof_meta_probs)
    print(f"Meta-Learner OOF LogLoss: {oof_score}")

    # 7. Final Inference
    print("Generating Final Predictions...")
    test_meta_features = test_df["log_char_len"].values
    X_meta_test = meta_learner.prepare_level1_features(
        [avg_test_stat, avg_test_deberta, avg_test_roberta], test_meta_features
    )

    final_test_probs = meta_learner.predict_proba(X_meta_test)

    # 8. Submission
    print(f"Saving submission to {Config.SUBMISSION_FILE}...")
    save_submission(test_df["id"].values, final_test_probs, Config.SUBMISSION_FILE)
    print("Pipeline Complete.")
