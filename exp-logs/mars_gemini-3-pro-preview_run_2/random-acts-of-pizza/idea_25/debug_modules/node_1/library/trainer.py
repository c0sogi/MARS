import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_manager import DataManager
from library.feature_engine import TextEmbedder, HomophilyTargetEncoder
from library.model_factory import ModelFactory

logger = setup_logger("trainer")


class Trainer:
    """
    Orchestrates the training pipeline:
    1. Loads data.
    2. Generates/Loads global features (Text).
    3. Runs Stratified CV.
    4. Generates fold-specific features (Homophily OOF) and scales data.
    5. Trains Bagging Ensemble.
    6. Evaluates and generates Test predictions.
    """

    def __init__(self):
        self.data_manager = DataManager()
        self.text_embedder = TextEmbedder()
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def run_cross_validation(self, load_cached_data: bool = True):
        set_seed(Config.RANDOM_SEED)
        logger.info("Starting Cross-Validation Pipeline...")

        # ---------------------------------------------------------
        # 1. Load Data
        # ---------------------------------------------------------
        df_train = self.data_manager.load_dataset(
            split="train", load_cached_data=load_cached_data
        )
        df_test = self.data_manager.load_dataset(
            split="test", load_cached_data=load_cached_data
        )

        target = df_train[Config.TARGET_COL].values
        request_ids_test = df_test["request_id"].values

        # ---------------------------------------------------------
        # 2. Global Feature Preparation (View 1 & 3)
        # ---------------------------------------------------------
        # View 1: Text Embeddings (Semantic)
        logger.info("Generating/Loading Text Embeddings...")
        train_text_emb = self.text_embedder.generate_embeddings(
            df_train,
            save_path=Config.TRAIN_EMBEDDINGS_PATH,
            load_cached=load_cached_data,
        )
        test_text_emb = self.text_embedder.generate_embeddings(
            df_test,
            save_path=Config.TEST_EMBEDDINGS_PATH,
            load_cached=load_cached_data,
        )

        # View 3: Numeric Metadata (Robust)
        # Extract numeric columns as numpy arrays
        train_numeric = df_train[Config.NUMERIC_COLS].values.astype(np.float32)
        test_numeric = df_test[Config.NUMERIC_COLS].values.astype(np.float32)

        # ---------------------------------------------------------
        # 3. Cross-Validation Loop
        # ---------------------------------------------------------
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
        )

        oof_preds = np.zeros(len(df_train))
        test_preds_accum = np.zeros(len(df_test))

        logger.info(f"Starting {Config.N_FOLDS}-Fold CV...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, target)):
            logger.info(f"--- Fold {fold + 1}/{Config.N_FOLDS} ---")

            # A. Split Data
            # Raw DataFrames (needed for Homophily Encoder which uses subreddit lists)
            fold_df_train = df_train.iloc[train_idx].copy()
            fold_df_val = df_train.iloc[val_idx].copy()
            fold_y_train = target[train_idx]
            fold_y_val = target[val_idx]

            # Pre-computed Features
            fold_text_train = train_text_emb[train_idx]
            fold_text_val = train_text_emb[val_idx]

            fold_num_train = train_numeric[train_idx]
            fold_num_val = train_numeric[val_idx]

            # B. View 2: Homophily Features (Target Encoding)
            # We use OOF encoding for the training part of the fold to prevent leakage
            # The encoder is automatically refitted on the full fold_df_train at the end of fit_transform_oof
            homophily_encoder = HomophilyTargetEncoder(smoothing=10.0)

            # 1. Train: OOF generation
            fold_homo_train = homophily_encoder.fit_transform_oof(
                fold_df_train,
                target_col=Config.TARGET_COL,
                n_folds=Config.INNER_CV_FOLDS,
            )

            # 2. Val: Transform using encoder fitted on fold_df_train
            fold_homo_val = homophily_encoder.transform(fold_df_val)

            # 3. Test: Transform using encoder fitted on fold_df_train
            fold_homo_test = homophily_encoder.transform(df_test)

            # C. Feature Fusion
            # Concatenate: [Text (384), Homophily (2), Numeric (10)]
            X_train_raw = np.hstack([fold_text_train, fold_homo_train, fold_num_train])
            X_val_raw = np.hstack([fold_text_val, fold_homo_val, fold_num_val])
            X_test_raw = np.hstack([test_text_emb, fold_homo_test, test_numeric])

            # D. Scaling (RankGauss)
            # Fit on training fold only
            scaler = QuantileTransformer(
                output_distribution="normal", random_state=Config.RANDOM_SEED
            )
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
            X_test = scaler.transform(X_test_raw)

            # E. Model Training
            model = ModelFactory.get_classifier()
            model.fit(X_train, fold_y_train)

            # F. Inference
            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]

            # G. Store Results
            oof_preds[val_idx] = val_probs
            test_preds_accum += test_probs

            # H. Evaluation
            fold_auc = roc_auc_score(fold_y_val, val_probs)
            logger.info(f"Fold {fold + 1} AUC: {fold_auc}")

            # I. Save Artifacts
            joblib.dump(
                model, os.path.join(self.models_dir, f"model_fold_{fold}.joblib")
            )
            joblib.dump(
                scaler, os.path.join(self.models_dir, f"scaler_fold_{fold}.joblib")
            )
            # We can't easily pickle the homophily encoder if it contains lambda functions,
            # but our implementation uses standard methods.
            joblib.dump(
                homophily_encoder,
                os.path.join(self.models_dir, f"homophily_encoder_fold_{fold}.joblib"),
            )

        # ---------------------------------------------------------
        # 4. Final Evaluation & Submission
        # ---------------------------------------------------------
        overall_auc = roc_auc_score(target, oof_preds)
        logger.info(f"Overall OOF AUC: {overall_auc}")

        # Average Test Predictions
        avg_test_preds = test_preds_accum / Config.N_FOLDS

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"request_id": request_ids_test, "requester_received_pizza": avg_test_preds}
        )

        # Save Submission
        logger.info(f"Saving submission to {Config.SUBMISSION_PATH}")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

        logger.info("Training pipeline completed successfully.")
