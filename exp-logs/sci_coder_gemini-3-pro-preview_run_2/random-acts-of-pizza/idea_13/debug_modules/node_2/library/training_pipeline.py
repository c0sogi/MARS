import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import PizzaDataLoader
from library.feature_engineering import TextEmbedder, TabularProcessor, FeatureFuser
from library.model_factory import ModelFactory


class TrainingPipeline:
    """
    Orchestrates the training, tuning, and evaluation of the Modality-Balanced
    Bagged Linear Ensemble.
    """

    def __init__(self):
        self.logger = setup_logger("training_pipeline")
        set_seed(Config.SEED)

    def run(self):
        """
        Executes the full pipeline:
        1. Data Loading & Feature Extraction
        2. 5-Fold Stratified CV with Nested Grid Search
        3. CV-Bagging Inference
        4. Submission Generation
        """
        self.logger.info("Starting Training Pipeline...")

        # ==========================================
        # 1. Load Data
        # ==========================================
        data_loader = PizzaDataLoader()
        train_df, val_df, test_df = data_loader.load_data(load_cached_data=True)

        # ==========================================
        # 2. Feature Extraction
        # ==========================================
        # Text Embeddings (L2 Normalized)
        text_embedder = TextEmbedder()
        X_train_text = text_embedder.get_embeddings(train_df, "train")
        X_val_text = text_embedder.get_embeddings(val_df, "val")
        X_test_text = text_embedder.get_embeddings(test_df, "test")

        # Numeric Metadata (Imputed & RankGauss Transformed)
        tabular_processor = TabularProcessor()
        X_train_tab, X_val_tab, X_test_tab = tabular_processor.process_numeric_features(
            train_df, val_df, test_df
        )

        # Targets
        y_train = train_df["requester_received_pizza"].values
        y_val = val_df["requester_received_pizza"].values

        # ==========================================
        # 3. Prepare Development Set for CV
        # ==========================================
        # Combine provided train and val splits to maximize data for 5-Fold CV
        X_dev_text = np.vstack([X_train_text, X_val_text])
        X_dev_tab = np.vstack([X_train_tab, X_val_tab])
        y_dev = np.hstack([y_train, y_val])

        self.logger.info(
            f"Combined Dev Set: Text={X_dev_text.shape}, Tab={X_dev_tab.shape}, y={y_dev.shape}"
        )

        # ==========================================
        # 4. Stratified K-Fold CV with Grid Search
        # ==========================================
        skf = StratifiedKFold(
            n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED
        )

        fold_models = []
        oof_preds = np.zeros(len(y_dev))
        grid_params = Config.GRID_SEARCH_PARAMS

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev_text, y_dev)):
            self.logger.info(f"=== Starting Fold {fold + 1}/{Config.N_SPLITS} ===")

            # Fold Data Split
            X_fold_train_text = X_dev_text[train_idx]
            X_fold_val_text = X_dev_text[val_idx]
            X_fold_train_tab = X_dev_tab[train_idx]
            X_fold_val_tab = X_dev_tab[val_idx]
            y_fold_train = y_dev[train_idx]
            y_fold_val = y_dev[val_idx]

            best_fold_auc = -1.0
            best_fold_model = None
            best_fold_params = {}

            # --- Nested Grid Search ---
            # Outer Loop: Modality Balance Factor (alpha)
            for alpha in grid_params["alpha"]:
                # Fuse features: Scale metadata by alpha, then concat with text
                X_fused_train = FeatureFuser.fuse(
                    X_fold_train_text, X_fold_train_tab, alpha
                )
                X_fused_val = FeatureFuser.fuse(X_fold_val_text, X_fold_val_tab, alpha)

                # Inner Loop: Regularization (C) and Class Weights
                for C in grid_params["C"]:
                    for cw in grid_params["class_weight"]:

                        # Train Bagged Ensemble
                        model = ModelFactory.create_bagged_ensemble(
                            C=C,
                            class_weight=cw,
                            n_estimators=Config.BAGGING_N_ESTIMATORS,
                            random_state=Config.SEED,
                        )
                        model.fit(X_fused_train, y_fold_train)

                        # Evaluate
                        y_pred_proba = model.predict_proba(X_fused_val)[:, 1]
                        auc = roc_auc_score(y_fold_val, y_pred_proba)

                        # Track Best
                        if auc > best_fold_auc:
                            best_fold_auc = auc
                            best_fold_model = model
                            best_fold_params = {
                                "alpha": alpha,
                                "C": C,
                                "class_weight": cw,
                            }

            self.logger.info(f"Fold {fold + 1} Best AUC: {best_fold_auc}")
            self.logger.info(f"Fold {fold + 1} Best Params: {best_fold_params}")

            # Store best model and its configuration
            fold_models.append(
                {"model": best_fold_model, "alpha": best_fold_params["alpha"]}
            )

            # Generate OOF predictions for overall CV score
            # Re-fuse validation data with the winning alpha
            X_fused_val_best = FeatureFuser.fuse(
                X_fold_val_text, X_fold_val_tab, best_fold_params["alpha"]
            )
            oof_preds[val_idx] = best_fold_model.predict_proba(X_fused_val_best)[:, 1]

        # ==========================================
        # 5. Overall Evaluation
        # ==========================================
        total_auc = roc_auc_score(y_dev, oof_preds)
        self.logger.info(f"Overall CV AUC: {total_auc}")

        # ==========================================
        # 6. Inference (CV-Bagging)
        # ==========================================
        self.logger.info("Generating Test Predictions...")

        test_preds_sum = np.zeros(len(test_df))

        for item in fold_models:
            model = item["model"]
            alpha = item["alpha"]

            # Fuse test features using the specific alpha optimized for this fold
            X_test_fused = FeatureFuser.fuse(X_test_text, X_test_tab, alpha)

            # Predict
            preds = model.predict_proba(X_test_fused)[:, 1]
            test_preds_sum += preds

        # Average predictions across folds
        avg_test_preds = test_preds_sum / Config.N_SPLITS

        # ==========================================
        # 7. Save Submission
        # ==========================================
        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": avg_test_preds,
            }
        )

        submission_path = Config.SUBMISSION_PATH
        submission_df.to_csv(submission_path, index=False)
        self.logger.info(f"Submission saved to {submission_path}")
