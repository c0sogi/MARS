import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, QuantileTransformer
from sklearn.decomposition import PCA

from library.utils import setup_logger, save_object, load_object, set_seed, WORKING_DIR
from library.data_loader import get_data_splits
from library.feature_extractor import EmbeddingGenerator, extract_metadata_features
from library.pipeline_builder import (
    build_adrsf_pipeline,
    combine_features,
    DIM_HIGH_RES,
    DIM_LOW_RES,
    DIM_METADATA,
)


class Trainer:
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.logger = setup_logger("Trainer", os.path.join(WORKING_DIR, "trainer.log"))
        self.models_dir = os.path.join(WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def _build_search_pipeline(self, pca_components=32):
        """
        Builds a non-bagged pipeline for efficient GridSearch of the base learner.
        Replicates the preprocessing logic of build_adrsf_pipeline.
        """
        slice_high = slice(0, DIM_HIGH_RES)
        slice_low = slice(DIM_HIGH_RES, DIM_HIGH_RES + DIM_LOW_RES)
        slice_meta = slice(
            DIM_HIGH_RES + DIM_LOW_RES, DIM_HIGH_RES + DIM_LOW_RES + DIM_METADATA
        )

        transformer_high = Normalizer(norm="l2")
        transformer_low = Pipeline(
            steps=[
                (
                    "pca",
                    PCA(n_components=pca_components, random_state=self.random_state),
                ),
                ("norm", Normalizer(norm="l2")),
            ]
        )
        transformer_meta = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("view_high", transformer_high, slice_high),
                ("view_low", transformer_low, slice_low),
                ("view_meta", transformer_meta, slice_meta),
            ],
            n_jobs=-1,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        return pipeline

    def run_cross_validation(self):
        set_seed(self.random_state)
        self.logger.info("Starting Cross-Validation Pipeline...")

        # 1. Load Data
        train_df, val_df, test_df = get_data_splits(load_cached_data=True)

        # Merge train and val for full CV usage
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)
        self.logger.info(f"Merged Train+Val shape: {full_train_df.shape}")

        # 2. Feature Extraction
        emb_gen = EmbeddingGenerator()

        # Process Training Data
        self.logger.info("Generating features for training data...")
        high_res_train, low_res_train = emb_gen.process_split(
            full_train_df, "full_train"
        )
        meta_train = extract_metadata_features(full_train_df)
        X = combine_features(high_res_train, low_res_train, meta_train)
        y = full_train_df["requester_received_pizza"].values.astype(int)

        # Process Test Data (Pre-compute for later)
        self.logger.info("Generating features for test data...")
        high_res_test, low_res_test = emb_gen.process_split(test_df, "test")
        meta_test = extract_metadata_features(test_df)
        X_test = combine_features(high_res_test, low_res_test, meta_test)
        test_ids = test_df["request_id"].values

        # 3. Cross-Validation Loop
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        oof_preds = np.zeros(len(y))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.logger.info(f"--- Fold {fold} ---")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # A. Grid Search for Base Learner
            self.logger.info("Running GridSearchCV for base learner...")
            search_pipeline = self._build_search_pipeline(pca_components=32)

            # Param Grid: C up to 10.0, Class Weights
            param_grid = {
                "classifier__C": np.logspace(-4, 1, 6),  # 0.0001 to 10.0
                "classifier__class_weight": ["balanced", None],
            }

            grid_search = GridSearchCV(
                search_pipeline,
                param_grid,
                cv=3,  # Internal CV for hyperparam tuning
                scoring="roc_auc",
                n_jobs=-1,
            )
            grid_search.fit(X_train_fold, y_train_fold)

            best_params = grid_search.best_params_
            best_C = best_params["classifier__C"]
            best_cw = best_params["classifier__class_weight"]
            self.logger.info(f"Best Params: C={best_C}, Class Weight={best_cw}")

            # B. Train Final Bagged Ensemble
            self.logger.info("Training final bagged ensemble with best params...")
            final_pipeline = build_adrsf_pipeline(
                pca_components=32,
                n_estimators=20,
                C=best_C,
                class_weight=best_cw,
                random_state=self.random_state,
            )

            final_pipeline.fit(X_train_fold, y_train_fold)

            # C. Evaluate
            y_pred_proba = final_pipeline.predict_proba(X_val_fold)[:, 1]
            oof_preds[val_idx] = y_pred_proba

            score = roc_auc_score(y_val_fold, y_pred_proba)
            fold_scores.append(score)
            self.logger.info(f"Fold {fold} ROC AUC: {score}")

            # D. Save Model
            model_path = os.path.join(self.models_dir, f"model_fold_{fold}.joblib")
            save_object(final_pipeline, model_path)

        # 4. Overall Evaluation
        overall_auc = roc_auc_score(y, oof_preds)
        self.logger.info(f"Overall CV ROC AUC: {overall_auc}")
        print(f"Overall CV ROC AUC: {overall_auc}")

        # 5. Generate Submission
        self.logger.info("Generating submission...")
        test_preds = np.zeros(len(X_test))

        for fold in range(self.n_folds):
            model_path = os.path.join(self.models_dir, f"model_fold_{fold}.joblib")
            model = load_object(model_path)
            test_preds += model.predict_proba(X_test)[:, 1]

        test_preds /= self.n_folds

        submission_df = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": test_preds}
        )

        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        self.logger.info(f"Submission saved to {submission_path}")
