import pandas as pd
import numpy as np
import os
import torch
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.dataset import clean_text
from library.lexical import train_lexical_fold
from library.engine import train_fold, predict
from library.optimization import optimize_thresholds, apply_thresholds


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.create_dirs()

    # Override Config for Fast Baseline
    Config.EPOCHS = 1

    print(f"--- Configuration ---")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Device: {Config.DEVICE}")
    print(f"Output Dir: {Config.OUTPUT_DIR}")

    # 2. Load Data
    print("\n--- Loading Data ---")
    df_train_full = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val_holdout = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Apply cleaning
    df_train_full["full_text"] = df_train_full["full_text"].apply(clean_text)
    df_val_holdout["full_text"] = df_val_holdout["full_text"].apply(clean_text)
    df_test["full_text"] = df_test["full_text"].apply(clean_text)

    # 3. Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for OOF
    oof_preds_sem = np.zeros(len(df_train_full))
    oof_preds_lex = np.zeros(len(df_train_full))

    # Storage for Holdout Predictions (Ensemble)
    holdout_preds_sem = np.zeros((len(df_val_holdout), Config.N_FOLDS))
    holdout_preds_lex = np.zeros((len(df_val_holdout), Config.N_FOLDS))

    # Storage for Test Predictions
    test_preds_sem = np.zeros((len(df_test), Config.N_FOLDS))
    test_preds_lex = np.zeros((len(df_test), Config.N_FOLDS))

    print(f"\n--- Starting {Config.N_FOLDS}-Fold Cross-Validation ---")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["score"])
    ):
        print(f"\n[Fold {fold}]")

        df_train_fold = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df_train_full.iloc[val_idx].reset_index(drop=True)

        # --- A. Lexical Branch ---
        print(f"Training Lexical Branch...")
        # Prepare combined evaluation set (Holdout + Test) to get predictions in one go
        # This is a workaround because train_lexical_fold accepts only one test_df
        df_combined_eval = pd.concat([df_val_holdout, df_test], axis=0).reset_index(
            drop=True
        )

        # Train and Predict
        # We pass df_val_fold as 'val_df' to get OOF predictions (returned as val_preds)
        # We pass df_combined_eval as 'test_df' to get holdout+test predictions (returned as test_preds)
        _, lex_oof_fold, combined_lex_preds = train_lexical_fold(
            df_train_fold,
            df_val_fold,
            df_combined_eval,
            fold_idx=fold,
            load_cached_data=True,
        )

        # Store OOF
        oof_preds_lex[val_idx] = lex_oof_fold

        # Split and Store Combined Predictions
        len_holdout = len(df_val_holdout)
        holdout_preds_lex[:, fold] = combined_lex_preds[:len_holdout]
        test_preds_lex[:, fold] = combined_lex_preds[len_holdout:]

        # --- B. Semantic Branch ---
        print(f"Training Semantic Branch...")
        # Train (Saves model to Config.OUTPUT_DIR)
        train_fold(fold, df_train_fold, df_val_fold)

        # Load best model for this fold
        checkpoint_path = os.path.join(Config.OUTPUT_DIR, f"deberta_fold_{fold}.bin")

        # Predict on OOF (Validation part of fold)
        print("Generating Semantic OOF predictions...")
        oof_preds_sem[val_idx] = predict(df_val_fold, checkpoint_path)

        # Predict on Holdout
        print("Generating Semantic Holdout predictions...")
        holdout_preds_sem[:, fold] = predict(df_val_holdout, checkpoint_path)

        # Predict on Test
        print("Generating Semantic Test predictions...")
        test_preds_sem[:, fold] = predict(df_test, checkpoint_path)

        # Cleanup to free GPU memory
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Meta-Learner (Stacking)
    print("\n--- Training Meta-Learner ---")
    # Features: [Semantic, Lexical]
    X_meta_train = np.column_stack([oof_preds_sem, oof_preds_lex])
    y_meta_train = df_train_full["score"].values

    meta_model = Ridge(alpha=1.0, random_state=Config.SEED)
    meta_model.fit(X_meta_train, y_meta_train)

    # 5. Threshold Optimization
    print("\n--- Optimizing Thresholds ---")
    # Predict on OOF using Meta
    meta_oof_preds = meta_model.predict(X_meta_train)

    # Optimize
    opt_thresholds = optimize_thresholds(y_meta_train, meta_oof_preds)
    print(f"Optimized Thresholds: {opt_thresholds}")

    # 6. Validation Evaluation (Hold-out)
    print("\n--- Final Validation ---")
    # Average predictions across folds
    avg_sem_holdout = holdout_preds_sem.mean(axis=1)
    avg_lex_holdout = holdout_preds_lex.mean(axis=1)

    # Meta Predict
    X_meta_holdout = np.column_stack([avg_sem_holdout, avg_lex_holdout])
    val_raw_preds = meta_model.predict(X_meta_holdout)

    # Apply Thresholds
    val_final_preds = apply_thresholds(val_raw_preds, opt_thresholds)

    # Metric
    val_qwk = compute_qwk(df_val_holdout["score"].values, val_final_preds)
    print(f"Final Validation Metric: {val_qwk}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate residuals
    residuals = np.abs(df_val_holdout["score"].values - val_final_preds)

    # Calculate word counts
    word_counts = (
        df_val_holdout["full_text"].apply(lambda x: len(str(x).split())).values
    )

    # Correlation
    corr = np.corrcoef(residuals, word_counts)[0, 1]
    print(f"Correlation between Error Magnitude and Word Count: {corr}")

    # 8. Submission
    SUBMISSION_THRESHOLD = 0.8307992749024942
    if val_qwk > SUBMISSION_THRESHOLD:
        print("\n--- Generating Submission ---")

        # Average Test Predictions
        avg_sem_test = test_preds_sem.mean(axis=1)
        avg_lex_test = test_preds_lex.mean(axis=1)

        # Meta Predict
        X_meta_test = np.column_stack([avg_sem_test, avg_lex_test])
        test_raw_preds = meta_model.predict(X_meta_test)

        # Apply Thresholds
        test_final_preds = apply_thresholds(test_raw_preds, opt_thresholds)

        # Create DataFrame
        submission = pd.DataFrame(
            {"essay_id": df_test["essay_id"], "score": test_final_preds}
        )

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission.head())
    else:
        print(
            f"Validation metric {val_qwk} did not meet threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
