import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

# Import from provided library
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.features import MechanicsFeatureExtractor
from library.dataset import get_dataloaders, get_test_dataloader
from library.model_semantic import DebertaV3Regressor, predict_semantic
from library.model_lexical import train_lexical_fold, predict_lexical
from library.trainer import run_fold
from library.meta_learner import MetaLearner, optimize_thresholds, apply_thresholds


def main():
    # --- 1. Configuration & Setup ---
    # Override Config for fast baseline execution
    Config.EPOCHS = 2
    Config.N_FOLDS = 3  # Reduced from 5 to 3 for speed
    Config.setup()

    seed_everything(Config.SEED)
    print(f"Starting execution with Device: {Config.DEVICE}")
    print(f"Config: {Config.EPOCHS} Epochs, {Config.N_FOLDS} Folds")

    # --- 2. Load Data ---
    print("\n--- Loading Data ---")
    df_train_full = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_holdout = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # --- 3. Feature Extraction (Mechanics Branch) ---
    print("\n--- Extracting Mechanics Features ---")
    extractor = MechanicsFeatureExtractor()

    # Extract features for all sets
    mech_train = extractor.extract_features(df_train_full, "train")
    mech_holdout = extractor.extract_features(df_holdout, "val")
    mech_test = extractor.extract_features(df_test, "test")

    # --- 4. Level 1: Cross-Validation Loop ---
    print("\n--- Starting Level 1 Stacking (Base Models) ---")

    # Storage for OOF predictions (aligned with df_train_full)
    oof_semantic = np.zeros(len(df_train_full))
    oof_lexical = np.zeros(len(df_train_full))

    # Storage for Holdout predictions (accumulate to average later)
    holdout_semantic_accum = np.zeros(len(df_holdout))
    holdout_lexical_accum = np.zeros(len(df_holdout))

    # Storage for Test predictions (accumulate to average later)
    test_semantic_accum = np.zeros(len(df_test))
    test_lexical_accum = np.zeros(len(df_test))

    # Tokenizer for Semantic Model
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_BACKBONE)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Prepare Holdout and Test Loaders (Semantic) once
    # We treat holdout as a test set here (no labels needed for prediction loop)
    holdout_loader = get_test_dataloader(df_holdout, tokenizer)
    test_loader = get_test_dataloader(df_test, tokenizer)

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["score"])
    ):
        print(f"\nProcessing Fold {fold_idx + 1}/{Config.N_FOLDS}")

        # Split Data
        df_fold_train = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_fold_val = df_train_full.iloc[val_idx].reset_index(drop=True)

        # --- A. Lexical Branch ---
        # Train and get validation score
        _, lex_val_preds = train_lexical_fold(fold_idx, df_fold_train, df_fold_val)

        # Save OOF
        oof_lexical[val_idx] = lex_val_preds

        # Predict on Holdout and Test
        lex_model_path = os.path.join(
            Config.MODEL_DIR, f"lexical_fold_{fold_idx}.joblib"
        )
        holdout_lexical_accum += predict_lexical(lex_model_path, df_holdout)
        test_lexical_accum += predict_lexical(lex_model_path, df_test)

        # --- B. Semantic Branch ---
        # Get DataLoaders
        train_loader, val_loader = get_dataloaders(
            df_fold_train, df_fold_val, tokenizer
        )

        # Train (saves best model to disk)
        _ = run_fold(fold_idx, train_loader, val_loader)

        # Load Best Model for Inference
        model_path = os.path.join(Config.MODEL_DIR, f"deberta_fold_{fold_idx}.bin")
        model = DebertaV3Regressor()
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)
        model.eval()

        # Predict OOF
        # Note: val_loader is shuffled=False, so order matches df_fold_val
        sem_val_preds = predict_semantic(model, val_loader, Config.DEVICE)
        oof_semantic[val_idx] = sem_val_preds

        # Predict on Holdout and Test
        holdout_semantic_accum += predict_semantic(model, holdout_loader, Config.DEVICE)
        test_semantic_accum += predict_semantic(model, test_loader, Config.DEVICE)

        # Cleanup
        del model, train_loader, val_loader
        torch.cuda.empty_cache()

    # Average predictions
    holdout_semantic_avg = holdout_semantic_accum / Config.N_FOLDS
    holdout_lexical_avg = holdout_lexical_accum / Config.N_FOLDS
    test_semantic_avg = test_semantic_accum / Config.N_FOLDS
    test_lexical_avg = test_lexical_accum / Config.N_FOLDS

    # --- 5. Level 2: Meta-Learner ---
    print("\n--- Training Meta-Learner ---")

    # Prepare Meta-Features
    # Train: OOFs + Mechanics
    X_meta_train = pd.DataFrame({"semantic": oof_semantic, "lexical": oof_lexical})
    X_meta_train = pd.concat([X_meta_train, mech_train.reset_index(drop=True)], axis=1)
    y_meta_train = df_train_full["score"].values

    # Holdout: Averaged Preds + Mechanics
    X_meta_holdout = pd.DataFrame(
        {"semantic": holdout_semantic_avg, "lexical": holdout_lexical_avg}
    )
    X_meta_holdout = pd.concat(
        [X_meta_holdout, mech_holdout.reset_index(drop=True)], axis=1
    )
    y_meta_holdout = df_holdout["score"].values

    # Train Meta-Learner
    meta_learner = MetaLearner()
    meta_learner.fit(X_meta_train, y_meta_train, X_meta_holdout, y_meta_holdout)

    # Get Continuous Predictions on Holdout
    holdout_preds_raw = meta_learner.predict(X_meta_holdout)

    # --- 6. Optimization & Evaluation ---
    print("\n--- Optimizing Thresholds ---")
    best_thresholds = optimize_thresholds(y_meta_holdout, holdout_preds_raw)

    # Apply Thresholds
    holdout_preds_final = apply_thresholds(holdout_preds_raw, best_thresholds)

    # Calculate Final Metric
    final_qwk = quadratic_weighted_kappa(y_meta_holdout, holdout_preds_final)
    print(f"Final Validation Metric: {final_qwk}")

    # --- 7. Failure Analysis ---
    print("\n--- Failure Analysis ---")
    residuals = np.abs(y_meta_holdout - holdout_preds_final)

    # Correlate residuals with mechanics features
    print("Correlation between Error Magnitude and Mechanics Features:")
    analysis_df = mech_holdout.copy()
    analysis_df["residual"] = residuals
    correlations = analysis_df.corr()["residual"].sort_values(ascending=False)
    print(correlations)

    # --- 8. Submission ---
    THRESHOLD_QWK = 0.8307992749024942

    if final_qwk > THRESHOLD_QWK:
        print(
            f"\nValidation Metric ({final_qwk}) > Threshold ({THRESHOLD_QWK}). Generating Submission..."
        )

        # Prepare Test Meta-Features
        X_meta_test = pd.DataFrame(
            {"semantic": test_semantic_avg, "lexical": test_lexical_avg}
        )
        X_meta_test = pd.concat([X_meta_test, mech_test.reset_index(drop=True)], axis=1)

        # Predict
        test_preds_raw = meta_learner.predict(X_meta_test)
        test_preds_final = apply_thresholds(test_preds_raw, best_thresholds)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"essay_id": df_test["essay_id"], "score": test_preds_final}
        )

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission.head())
    else:
        print(
            f"\nValidation Metric ({final_qwk}) did not meet threshold ({THRESHOLD_QWK}). Skipping submission."
        )


if __name__ == "__main__":
    main()
