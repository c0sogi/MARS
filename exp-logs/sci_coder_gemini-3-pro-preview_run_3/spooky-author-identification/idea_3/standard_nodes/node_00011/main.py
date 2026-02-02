import sys
import os
import numpy as np
import pandas as pd
import torch
import gc
import warnings
from sklearn.metrics import log_loss

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import library modules
from library.config import Config
from library.utils import set_seed, save_submission, calculate_log_loss
from library.data_loader import load_data, get_cv_folds
from library.models_statistical import StatisticalExpert
from library.models_neural import TransformerExpert
from library.stacking import MetaLearner

# --- Monkey Patch Config for Fast Baseline ---
# We reduce epochs and folds to ensure execution within the time limit
Config.EPOCHS = 1
Config.N_FOLDS = 3
Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("--- Starting Runfile Execution ---")
    print(f"Device: {Config.DEVICE}")
    print(f"Configuration: Epochs={Config.EPOCHS}, Folds={Config.N_FOLDS}")

    # 2. Load Data
    # load_data handles caching and meta-feature generation (log_char_len)
    # We use train_df for CV/Training and val_df as the strict hold-out set
    train_df, val_df, test_df = load_data(load_cached_data=True, debug=False)

    print(
        f"Train set: {train_df.shape}, Val set: {val_df.shape}, Test set: {test_df.shape}"
    )

    # Prepare arrays for Stacking
    n_train = len(train_df)
    n_classes = 3

    # OOF predictions on train_df (Level 1 features for Meta-Learner training)
    oof_train_stat = np.zeros((n_train, n_classes))
    oof_train_deberta = np.zeros((n_train, n_classes))
    oof_train_roberta = np.zeros((n_train, n_classes))

    # Predictions on hold-out val_df (averaged across folds)
    val_preds_stat_folds = []
    val_preds_deberta_folds = []
    val_preds_roberta_folds = []

    # Predictions on test_df (averaged across folds)
    test_preds_stat_folds = []
    test_preds_deberta_folds = []
    test_preds_roberta_folds = []

    # Targets
    y_train = train_df["target"].values
    y_val = val_df["target"].values

    # 3. Cross-Validation Loop on train_df
    skf = get_cv_folds(n_splits=Config.N_FOLDS, random_state=Config.SEED)

    for fold, (train_idx, fold_val_idx) in enumerate(skf.split(train_df, y_train)):
        print(f"\n=== Fold {fold + 1}/{Config.N_FOLDS} ===")

        # Split Fold Data
        X_fold_train = train_df.iloc[train_idx]
        X_fold_val = train_df.iloc[fold_val_idx]

        train_texts = X_fold_train["text"].values
        train_targets = X_fold_train["target"].values
        fold_val_texts = X_fold_val["text"].values
        fold_val_targets = X_fold_val["target"].values

        # Hold-out Val and Test texts
        holdout_val_texts = val_df["text"].values
        test_texts = test_df["text"].values

        # --- A. Statistical Expert ---
        print("Training Statistical Expert...")
        model_stat = StatisticalExpert()
        model_stat.fit(train_texts, train_targets)

        # Predict
        oof_train_stat[fold_val_idx] = model_stat.predict_proba(fold_val_texts)
        val_preds_stat_folds.append(model_stat.predict_proba(holdout_val_texts))
        test_preds_stat_folds.append(model_stat.predict_proba(test_texts))

        # --- B. Neural Expert: DeBERTa ---
        print(f"Training Neural Expert ({Config.MODEL_DEBERTA})...")
        model_deberta = TransformerExpert(model_name=Config.MODEL_DEBERTA)
        model_deberta.fit(train_texts, train_targets, fold_val_texts, fold_val_targets)

        # Predict
        oof_train_stat_deb = model_deberta.predict_proba(fold_val_texts)
        oof_train_deberta[fold_val_idx] = oof_train_stat_deb
        val_preds_deberta_folds.append(model_deberta.predict_proba(holdout_val_texts))
        test_preds_deberta_folds.append(model_deberta.predict_proba(test_texts))

        # Cleanup
        del model_deberta
        gc.collect()
        torch.cuda.empty_cache()

        # --- C. Neural Expert: RoBERTa ---
        print(f"Training Neural Expert ({Config.MODEL_ROBERTA})...")
        model_roberta = TransformerExpert(model_name=Config.MODEL_ROBERTA)
        model_roberta.fit(train_texts, train_targets, fold_val_texts, fold_val_targets)

        # Predict
        oof_train_stat_rob = model_roberta.predict_proba(fold_val_texts)
        oof_train_roberta[fold_val_idx] = oof_train_stat_rob
        val_preds_roberta_folds.append(model_roberta.predict_proba(holdout_val_texts))
        test_preds_roberta_folds.append(model_roberta.predict_proba(test_texts))

        # Cleanup
        del model_roberta
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Aggregate Predictions (Level 0)
    print("\nAggregating Level 0 Predictions...")
    # Average predictions on hold-out val and test
    avg_val_stat = np.mean(val_preds_stat_folds, axis=0)
    avg_val_deberta = np.mean(val_preds_deberta_folds, axis=0)
    avg_val_roberta = np.mean(val_preds_roberta_folds, axis=0)

    avg_test_stat = np.mean(test_preds_stat_folds, axis=0)
    avg_test_deberta = np.mean(test_preds_deberta_folds, axis=0)
    avg_test_roberta = np.mean(test_preds_roberta_folds, axis=0)

    # 5. Meta-Learner Training (Level 1)
    print("\nTraining Meta-Learner...")
    meta_learner = MetaLearner()

    # Prepare Features for Meta-Learner Training (from OOF on train_df)
    train_meta_features = train_df["log_char_len"].values
    X_meta_train = meta_learner.prepare_level1_features(
        [oof_train_stat, oof_train_deberta, oof_train_roberta], train_meta_features
    )

    meta_learner.fit(X_meta_train, y_train)

    # 6. Validation on Hold-Out Set
    print("Performing Validation on Hold-Out Set...")
    val_meta_features = val_df["log_char_len"].values
    X_meta_val = meta_learner.prepare_level1_features(
        [avg_val_stat, avg_val_deberta, avg_val_roberta], val_meta_features
    )

    val_final_probs = meta_learner.predict_proba(X_meta_val)

    # Calculate Metric
    final_metric = calculate_log_loss(y_val, val_final_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude per sample (Negative Log Likelihood of the true class)
    # y_val contains indices 0, 1, 2
    row_indices = np.arange(len(y_val))
    true_class_probs = val_final_probs[row_indices, y_val]

    # Clip to avoid log(0)
    eps = 1e-15
    true_class_probs = np.clip(true_class_probs, eps, 1 - eps)
    error_magnitudes = -np.log(true_class_probs)

    # Calculate correlation with log_char_len
    correlation = np.corrcoef(val_meta_features, error_magnitudes)[0, 1]
    print(f"Correlation between Error Magnitude and log_char_len: {correlation:.10f}")

    # 8. Submission
    THRESHOLD = 0.2665362892717963
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_meta_features = test_df["log_char_len"].values
        X_meta_test = meta_learner.prepare_level1_features(
            [avg_test_stat, avg_test_deberta, avg_test_roberta], test_meta_features
        )

        test_final_probs = meta_learner.predict_proba(X_meta_test)

        save_submission(test_df["id"].values, test_final_probs, Config.SUBMISSION_FILE)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
