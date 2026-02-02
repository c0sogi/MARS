import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
import joblib

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_data
from library.embeddings import get_text_embeddings
from library.features import UserPersonaTransformer, MetadataTransformer

# Initialize logger
logger = setup_logger(name="trainer")


def run_training_pipeline(load_cached_data=True):
    """
    Executes the full training pipeline:
    1. Loads and merges data (Train + Val) for Cross-Validation.
    2. Loads pre-computed SBERT embeddings (View 1).
    3. Iterates through Stratified K-Folds.
    4. Inside each fold:
       - Fits UserPersonaTransformer (View 2) and MetadataTransformer (View 3) on training data.
       - Fuses the three feature views.
       - Performs GridSearchCV to tune a Bagged Logistic Regression model.
       - Generates predictions for the validation set (OOF) and test set.
    5. Aggregates predictions and saves the submission file.

    Args:
        load_cached_data (bool): Whether to use cached data/embeddings.
    """
    set_seed(Config.SEED)

    # ==========================================
    # 1. Data Loading & Preparation
    # ==========================================
    logger.info("Starting data loading...")
    df_train_split, df_val_split, df_test = load_data(load_cached_data=load_cached_data)

    # Load SBERT Embeddings (View 1)
    emb_train_split, emb_val_split, emb_test = get_text_embeddings(
        df_train_split, df_val_split, df_test, load_cached_data=load_cached_data
    )

    # Merge Train and Validation splits to form the full Development set for CV
    # We do this to maximize data usage with 5-fold CV
    logger.info("Merging provided train/val splits for 5-Fold Cross-Validation...")

    # Concatenate DataFrames
    df_dev = pd.concat([df_train_split, df_val_split], axis=0).reset_index(drop=True)

    # Concatenate Embeddings
    emb_dev = np.vstack([emb_train_split, emb_val_split])

    # Extract Targets
    y_dev = df_dev["requester_received_pizza"].astype(int).values

    # Prepare Test IDs for submission
    test_ids = df_test["request_id"].values

    logger.info(f"Development Set Shape: {df_dev.shape}")
    logger.info(f"Test Set Shape: {df_test.shape}")

    # ==========================================
    # 2. Cross-Validation Loop
    # ==========================================
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Arrays to store results
    oof_preds = np.zeros(len(df_dev))
    test_preds_accum = np.zeros(len(df_test))
    fold_scores = []

    logger.info(f"Starting {Config.N_FOLDS}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_dev, y_dev)):
        logger.info(f"\n=== Fold {fold + 1}/{Config.N_FOLDS} ===")

        # ---------------------------
        # A. Data Splitting
        # ---------------------------
        # Split DataFrames (for metadata/persona features)
        X_train_df = df_dev.iloc[train_idx].copy()
        X_val_df = df_dev.iloc[val_idx].copy()
        y_train, y_val = y_dev[train_idx], y_dev[val_idx]

        # Split Embeddings (View 1 - Pre-computed)
        X_train_emb = emb_dev[train_idx]
        X_val_emb = emb_dev[val_idx]

        # ---------------------------
        # B. Feature Engineering (Fit on Train)
        # ---------------------------
        logger.info("Fitting dynamic feature transformers...")

        # View 2: User Persona (LSA)
        persona_transformer = UserPersonaTransformer(
            n_components=Config.LSA_N_COMPONENTS, random_state=Config.SEED
        )
        # Fit on train, transform all
        X_train_persona = persona_transformer.fit_transform(X_train_df)
        X_val_persona = persona_transformer.transform(X_val_df)
        X_test_persona = persona_transformer.transform(df_test)

        # View 3: Robust Metadata
        meta_transformer = MetadataTransformer(
            numerical_cols=Config.NUMERICAL_COLS, random_state=Config.SEED
        )
        # Fit on train, transform all
        X_train_meta = meta_transformer.fit_transform(X_train_df)
        X_val_meta = meta_transformer.transform(X_val_df)
        X_test_meta = meta_transformer.transform(df_test)

        # ---------------------------
        # C. Feature Fusion
        # ---------------------------
        # Concatenate: [SBERT (384), Persona (16), Metadata (~10)]
        X_train_fused = np.hstack([X_train_emb, X_train_persona, X_train_meta])
        X_val_fused = np.hstack([X_val_emb, X_val_persona, X_val_meta])
        X_test_fused = np.hstack([emb_test, X_test_persona, X_test_meta])

        # ---------------------------
        # D. Model Training (Grid Search)
        # ---------------------------
        logger.info("Tuning Bagged Logistic Regression...")

        # Define Base Estimator
        base_lr = LogisticRegression(
            penalty=Config.LR_PENALTY,
            solver="lbfgs",
            max_iter=Config.LR_MAX_ITER,
            random_state=Config.SEED,
        )

        # Define Bagging Wrapper
        # Note: We tune the inner LR parameters via the bagging wrapper
        bagging_clf = BaggingClassifier(
            estimator=base_lr,
            n_estimators=Config.BAGGING_N_ESTIMATORS,
            max_samples=Config.BAGGING_MAX_SAMPLES,
            random_state=Config.SEED,
            n_jobs=1,  # Parallelism handled by GridSearchCV
        )

        # Parameter Grid
        # We target the 'estimator' (LR) parameters using double underscore
        param_grid = {
            "estimator__C": Config.LR_C_CANDIDATES,
            "estimator__class_weight": Config.LR_CLASS_WEIGHTS,
        }

        # Grid Search with internal CV (3-fold)
        grid_search = GridSearchCV(
            bagging_clf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0
        )

        grid_search.fit(X_train_fused, y_train)

        best_model = grid_search.best_estimator_
        logger.info(f"Best Params: {grid_search.best_params_}")

        # ---------------------------
        # E. Prediction & Evaluation
        # ---------------------------
        # Validation Prediction
        val_probs = best_model.predict_proba(X_val_fused)[:, 1]
        oof_preds[val_idx] = val_probs

        fold_auc = roc_auc_score(y_val, val_probs)
        fold_scores.append(fold_auc)
        logger.info(f"Fold {fold + 1} ROC AUC: {fold_auc}")

        # Test Prediction (Accumulate)
        test_probs = best_model.predict_proba(X_test_fused)[:, 1]
        test_preds_accum += test_probs

        # Optional: Save model for this fold
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")
        joblib.dump(best_model, model_path)

    # ==========================================
    # 3. Aggregation & Submission
    # ==========================================
    logger.info("\n=== Training Complete ===")

    # Calculate OOF Score
    overall_auc = roc_auc_score(y_dev, oof_preds)
    mean_fold_auc = np.mean(fold_scores)
    std_fold_auc = np.std(fold_scores)

    logger.info(f"Mean Fold AUC: {mean_fold_auc}")
    logger.info(f"Std Fold AUC: {std_fold_auc}")
    logger.info(f"Overall OOF AUC: {overall_auc}")

    # Average Test Predictions
    avg_test_preds = test_preds_accum / Config.N_FOLDS

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"request_id": test_ids, "requester_received_pizza": avg_test_preds}
    )

    # Save Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    logger.info(f"Submission saved to: {submission_path}")

    return overall_auc
