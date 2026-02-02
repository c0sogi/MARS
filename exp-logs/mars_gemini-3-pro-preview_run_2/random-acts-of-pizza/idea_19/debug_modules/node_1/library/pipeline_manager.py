import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, QuantileTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import get_logger, set_seed
from library.data_manager import load_dataset
from library.feature_extractor import generate_sbert_embeddings

logger = get_logger("PipelineManager")


class LPADFPipelineManager:
    """
    Manages the Latent Persona Augmented Dense Fusion (LPADF) pipeline.
    Handles feature merging, pipeline construction, cross-validation, and inference.
    """

    def __init__(self):
        self.sbert_cols = [f"sbert_{i}" for i in range(384)]
        set_seed(Config.RANDOM_SEED)

    def merge_features(self, df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
        """
        Merges the base DataFrame with SBERT embeddings.

        Args:
            df (pd.DataFrame): Base data containing metadata and text.
            embeddings (np.ndarray): Pre-computed SBERT embeddings (N, 384).

        Returns:
            pd.DataFrame: Merged DataFrame with 'sbert_X' columns.
        """
        # Create a DataFrame for embeddings
        emb_df = pd.DataFrame(embeddings, columns=self.sbert_cols, index=df.index)

        # Concatenate horizontally
        merged_df = pd.concat([df, emb_df], axis=1)
        return merged_df

    def create_pipeline(self) -> Pipeline:
        """
        Constructs the Scikit-Learn pipeline for LPADF.

        Structure:
            - ColumnTransformer:
                - Subreddits: TF-IDF -> SVD(16) -> L2 Norm
                - Metadata: Imputer -> RankGauss
                - SBERT: Passthrough
            - Classifier: Bagging(LogisticRegression)

        Returns:
            Pipeline: Unfitted sklearn pipeline.
        """
        # 1. User Persona View (Subreddits)
        # Input: 'subreddit_string' column (Series)
        subreddit_pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b"),
                ),
                (
                    "svd",
                    TruncatedSVD(
                        n_components=Config.LSA_N_COMPONENTS,
                        random_state=Config.RANDOM_SEED,
                    ),
                ),
                ("norm", Normalizer(norm="l2")),
            ]
        )

        # 2. Robust Metadata View
        # Input: Numerical columns (DataFrame)
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                (
                    "scaler",
                    QuantileTransformer(
                        output_distribution="normal", random_state=Config.RANDOM_SEED
                    ),
                ),
            ]
        )

        # 3. Feature Union (ColumnTransformer)
        preprocessor = ColumnTransformer(
            [
                ("subreddits", subreddit_pipeline, "subreddit_string"),
                ("metadata", numeric_pipeline, Config.NUMERICAL_COLS),
                ("sbert", "passthrough", self.sbert_cols),
            ]
        )

        # 4. Classifier
        # Base Estimator: Logistic Regression
        lr = LogisticRegression(
            solver="lbfgs", max_iter=2000, random_state=Config.RANDOM_SEED
        )

        # Ensemble: Bagging
        # Note: We set n_jobs=1 here to avoid over-subscription when running GridSearch in parallel
        bagging = BaggingClassifier(
            estimator=lr,
            n_estimators=Config.N_ESTIMATORS,
            random_state=Config.RANDOM_SEED,
            n_jobs=1,
        )

        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", bagging)])

        return pipeline

    def run_cv_and_predict(self):
        """
        Executes the full training and inference workflow:
        1. Loads data.
        2. Merges features.
        3. Runs Stratified CV with nested Grid Search.
        4. Generates submission.
        """
        logger.info("Starting LPADF Pipeline execution...")

        # 1. Load Data
        df_train, df_val, df_test = load_dataset()
        train_emb, val_emb, test_emb = generate_sbert_embeddings(
            df_train, df_val, df_test
        )

        # 2. Merge Features
        logger.info("Merging features...")
        X_train_full = self.merge_features(df_train, train_emb)
        y_train_full = df_train["requester_received_pizza"].values

        X_val_full = self.merge_features(df_val, val_emb)
        y_val_full = df_val["requester_received_pizza"].values

        X_test_full = self.merge_features(df_test, test_emb)

        # Combine Train and Val for full Cross-Validation
        # We ignore the fixed train/val split from metadata for the final CV
        # to maximize data usage, as we are doing 5-fold CV anyway.
        X_all = pd.concat([X_train_full, X_val_full], axis=0).reset_index(drop=True)
        y_all = np.concatenate([y_train_full, y_val_full], axis=0)

        logger.info(f"Combined Training Data Shape: {X_all.shape}")

        # 3. Stratified Cross-Validation
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
        )

        oof_preds = np.zeros(len(X_all))
        test_preds = np.zeros(len(X_test_full))

        fold_aucs = []

        # Define Grid Search Space
        # Note: Parameters target the estimator inside BaggingClassifier
        param_grid = {
            "classifier__estimator__C": Config.LR_C_RANGE,
            "classifier__estimator__class_weight": Config.LR_CLASS_WEIGHTS,
        }

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            logger.info(f"Processing Fold {fold + 1}/{Config.N_FOLDS}...")

            X_tr, X_va = X_all.iloc[train_idx], X_all.iloc[val_idx]
            y_tr, y_va = y_all[train_idx], y_all[val_idx]

            # Create Pipeline
            pipeline = self.create_pipeline()

            # Grid Search (Inner Loop)
            # We use a small inner CV (3-fold) or just fit on training data if data is scarce.
            # Given dataset size (~3k), 3-fold inner CV is appropriate.
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,
                scoring="roc_auc",
                n_jobs=4,  # Parallelize grid search
                verbose=0,
            )

            grid_search.fit(X_tr, y_tr)

            best_model = grid_search.best_estimator_
            logger.info(f"  Best Params: {grid_search.best_params_}")

            # Predict on Validation
            val_probs = best_model.predict_proba(X_va)[:, 1]
            oof_preds[val_idx] = val_probs

            # Calculate Fold AUC
            fold_auc = roc_auc_score(y_va, val_probs)
            fold_aucs.append(fold_auc)
            logger.info(f"  Fold {fold + 1} AUC: {fold_auc:.10f}")

            # Predict on Test
            test_probs = best_model.predict_proba(X_test_full)[:, 1]
            test_preds += test_probs / Config.N_FOLDS

        # 4. Overall Evaluation
        overall_auc = roc_auc_score(y_all, oof_preds)
        mean_auc = np.mean(fold_aucs)

        logger.info("-" * 40)
        logger.info(f"Overall OOF AUC: {overall_auc:.10f}")
        logger.info(f"Mean Fold AUC:   {mean_auc:.10f}")
        logger.info("-" * 40)

        # 5. Save Submission
        self.save_submission(df_test, test_preds)

    def save_submission(self, df_test: pd.DataFrame, predictions: np.ndarray):
        """
        Saves predictions to the submission file.

        Args:
            df_test (pd.DataFrame): Test dataframe containing request_id.
            predictions (np.ndarray): Predicted probabilities.
        """
        submission = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": predictions,
            }
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_lpadf_pipeline():
    """
    Entry point to run the pipeline manager.
    """
    manager = LPADFPipelineManager()
    manager.run_cv_and_predict()
