import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

from library.config import Config
from library.utils import get_logger
from library.features import preprocess_data

logger = get_logger("model")


class XGBoostTrainer:
    """
    Wrapper for XGBoost training to handle parameters, probability output, and early stopping.
    """

    def __init__(self, num_class: int, device: str = "cuda"):
        self.params = Config.XGB_PARAMS.copy()
        self.params["device"] = device
        self.params["num_class"] = num_class
        # Override objective to ensure we get probability outputs for Soft Voting
        self.params["objective"] = "multi:softprob"

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model with Early Stopping.
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        evals = [(dtrain, "train"), (dval, "val")]

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=Config.NUM_BOOST_ROUND,
            evals=evals,
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=Config.VERBOSE_EVAL,
        )
        return model

    @staticmethod
    def predict_proba(model, X):
        """
        Generates class probabilities.
        """
        dtest = xgb.DMatrix(X)
        return model.predict(dtest)


def run_experiment(debug: bool = Config.DEBUG):
    """
    Main execution pipeline:
    1. Loads and preprocesses data.
    2. Merges Train and Val sets for full 5-Fold CV.
    3. Runs Stratified 5-Fold CV with dynamic, leakage-free k-NN injection.
    4. Generates predictions for the Test set using Soft Voting.
    5. Saves the submission file.
    """
    logger.info(f"Starting experiment (Debug={debug})...")

    # --- 1. Load Data ---
    # Load processed data (Physics features, Dense indices, Scaled cols)
    train_df, val_df, test_df = preprocess_data(load_cached_data=True, debug=debug)

    # Merge Train and Val to use 100% data for Cross-Validation
    logger.info("Merging Train and Val sets for full Cross-Validation...")
    train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    del val_df
    gc.collect()

    logger.info(f"Combined Training Data Shape: {train_df.shape}")

    # --- 2. Encode Target ---
    le = LabelEncoder()
    train_df[Config.TARGET_COL] = le.fit_transform(train_df[Config.TARGET_COL])
    num_classes = len(le.classes_)
    logger.info(f"Target Encoded. Num Classes: {num_classes}")

    # --- 3. Define Feature Exclusion ---
    # Exclude ID, Target, and the Scaled columns used purely for KNN search
    ignore_cols = [Config.ID_COL, Config.TARGET_COL] + [
        f"{c}_scaled" for c in Config.KNN_FEATURES
    ]

    # --- 4. Stratified K-Fold CV ---
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Arrays to store predictions
    oof_preds = np.zeros((len(train_df), num_classes), dtype=np.float32)
    test_preds_sum = np.zeros((len(test_df), num_classes), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df[Config.TARGET_COL])
    ):
        logger.info(f"\n=== Fold {fold + 1} / {Config.N_FOLDS} ===")

        # Split Data
        X_tr_fold = train_df.iloc[train_idx].copy()
        X_val_fold = train_df.iloc[val_idx].copy()

        # --- Dynamic k-NN Injection ---
        logger.info("Injecting k-NN features...")

        # A. Train Set: Reference = Self, Exclude Self = True (Prevent Leakage)
        X_tr_aug = inject_knn_features(
            ref_df=X_tr_fold,
            query_df=X_tr_fold,
            knn_cols=Config.KNN_FEATURES,
            exclude_self=True,
        )

        # B. Val Set: Reference = Train Fold, Exclude Self = False
        X_val_aug = inject_knn_features(
            ref_df=X_tr_fold,
            query_df=X_val_fold,
            knn_cols=Config.KNN_FEATURES,
            exclude_self=False,
        )

        # C. Test Set: Reference = Train Fold, Exclude Self = False
        # Each model in the ensemble predicts using its specific training data as the manifold reference
        X_test_aug = inject_knn_features(
            ref_df=X_tr_fold,
            query_df=test_df,
            knn_cols=Config.KNN_FEATURES,
            exclude_self=False,
        )

        # Identify final feature columns (Base + Injected KNN features)
        train_cols = [c for c in X_tr_aug.columns if c not in ignore_cols]
        if fold == 0:
            logger.info(f"Final Feature Count: {len(train_cols)}")
            logger.info(f"Features: {train_cols}")

        # Prepare Data for XGBoost
        X_train_np = X_tr_aug[train_cols]
        y_train_np = X_tr_aug[Config.TARGET_COL]
        X_val_np = X_val_aug[train_cols]
        y_val_np = X_val_aug[Config.TARGET_COL]
        X_test_np = X_test_aug[train_cols]

        # Train
        trainer = XGBoostTrainer(num_class=num_classes)
        model = trainer.fit(X_train_np, y_train_np, X_val_np, y_val_np)

        # Predict
        val_probs = trainer.predict_proba(model, X_val_np)
        test_probs = trainer.predict_proba(model, X_test_np)

        # Accumulate Results
        oof_preds[val_idx] = val_probs
        test_preds_sum += test_probs

        # Cleanup to free memory
        del X_tr_fold, X_val_fold, X_tr_aug, X_val_aug, X_test_aug
        del X_train_np, y_train_np, X_val_np, y_val_np, X_test_np
        del model, trainer
        gc.collect()

    # --- 5. Evaluation ---
    logger.info("\n=== CV Evaluation ===")
    oof_labels = np.argmax(oof_preds, axis=1)
    acc = accuracy_score(train_df[Config.TARGET_COL], oof_labels)
    loss = log_loss(train_df[Config.TARGET_COL], oof_preds)

    logger.info(f"Overall OOF Accuracy: {acc:.6f}")
    logger.info(f"Overall OOF Log Loss: {loss:.6f}")

    # --- 6. Submission ---
    logger.info("Generating Submission...")
    # Soft Voting: Average probabilities
    avg_test_probs = test_preds_sum / Config.N_FOLDS
    test_pred_labels = np.argmax(avg_test_probs, axis=1)

    # Inverse Transform Labels to original class IDs
    final_preds = le.inverse_transform(test_pred_labels)

    submission = pd.DataFrame(
        {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: final_preds}
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
