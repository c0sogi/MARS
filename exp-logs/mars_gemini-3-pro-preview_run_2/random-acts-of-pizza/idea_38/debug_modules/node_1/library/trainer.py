import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_and_process_data
from library.embedding_manager import EmbeddingService
from library.pipeline_builder import PipelineBuilder


class Trainer:
    """
    Orchestrates the training and submission generation for the DAADBE model.
    """

    def __init__(self):
        self.logger = setup_logger("trainer")
        self.embedding_service = EmbeddingService()
        set_seed(Config.SEED)

    def _prepare_feature_matrix(
        self, df: pd.DataFrame, anchor_emb: np.ndarray, aux_emb: np.ndarray
    ) -> tuple[pd.DataFrame, list, list]:
        """
        Combines metadata dataframe with embedding arrays into a single DataFrame
        suitable for the ColumnTransformer.

        Returns:
            pd.DataFrame: Combined feature matrix.
            list: Column names for anchor embeddings.
            list: Column names for aux embeddings.
        """
        # Create column names for embeddings
        anchor_cols = [f"anchor_{i}" for i in range(anchor_emb.shape[1])]
        aux_cols = [f"aux_{i}" for i in range(aux_emb.shape[1])]

        # Create DataFrames for embeddings
        df_anchor = pd.DataFrame(anchor_emb, columns=anchor_cols, index=df.index)
        df_aux = pd.DataFrame(aux_emb, columns=aux_cols, index=df.index)

        # Concatenate horizontally
        # We assume df, df_anchor, df_aux share the same index (ensured by reset_index in caller)
        X = pd.concat([df, df_anchor, df_aux], axis=1)

        return X, anchor_cols, aux_cols

    def train(self, debug: bool = False, load_cached_data: bool = True):
        """
        Executes the Stratified K-Fold Cross-Validation training loop.
        """
        self.logger.info("Starting training process...")

        # 1. Load Data
        df_train_split, df_val_split, _ = load_and_process_data(
            debug=debug, load_cached_data=load_cached_data
        )

        # 2. Get Embeddings for splits
        # We need to get embeddings before concatenating to ensure alignment with cache keys
        anchor_train = self.embedding_service.get_embeddings(
            df_train_split, "train", "anchor", load_cached_data
        )
        aux_train = self.embedding_service.get_embeddings(
            df_train_split, "train", "aux", load_cached_data
        )

        anchor_val = self.embedding_service.get_embeddings(
            df_val_split, "val", "anchor", load_cached_data
        )
        aux_val = self.embedding_service.get_embeddings(
            df_val_split, "val", "aux", load_cached_data
        )

        # 3. Combine Train and Val for CV
        # Reset indices to ensure clean concatenation
        df_full = pd.concat([df_train_split, df_val_split], axis=0).reset_index(
            drop=True
        )
        anchor_full = np.vstack([anchor_train, anchor_val])
        aux_full = np.vstack([aux_train, aux_val])

        y = df_full[Config.TARGET_COL].values

        # 4. Construct Feature Matrix
        X, anchor_cols, aux_cols = self._prepare_feature_matrix(
            df_full, anchor_full, aux_full
        )

        # 5. Stratified K-Fold CV
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.logger.info(f"--- Starting Fold {fold} ---")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Build Pipeline
            # We pass the column names dynamically
            pipeline = PipelineBuilder.build_daadbe_pipeline(
                anchor_cols=anchor_cols,
                aux_cols=aux_cols,
                continuous_cols=Config.NUMERICAL_COLS,
                discrete_cols=Config.DISCRETE_COLS,
                pca_components=Config.PCA_COMPONENTS,
                n_bins=Config.N_BINS,
                bin_strategy=Config.BIN_STRATEGY,
                n_estimators=Config.N_ESTIMATORS,
                random_state=Config.SEED,
            )

            # Adjust Grid Search Params for the Pipeline structure
            # Pipeline: [('preprocessor', ...), ('classifier', BaggingClassifier)]
            # BaggingClassifier: estimator=LogisticRegression
            # Target: classifier__estimator__<param>
            grid_params = {
                f"classifier__estimator__{k.split('__')[1]}": v
                for k, v in Config.GRID_SEARCH_PARAMS.items()
            }

            # Grid Search
            self.logger.info("Running GridSearchCV...")
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=grid_params,
                scoring="roc_auc",
                cv=3,  # Internal CV for tuning
                n_jobs=-1,
                verbose=0,
            )

            grid.fit(X_train_fold, y_train_fold)

            best_model = grid.best_estimator_
            self.logger.info(f"Fold {fold} Best Params: {grid.best_params_}")

            # Evaluate
            y_pred_proba = best_model.predict_proba(X_val_fold)[:, 1]
            score = roc_auc_score(y_val_fold, y_pred_proba)
            self.logger.info(f"Fold {fold} ROC AUC: {score}")
            fold_scores.append(score)

            # Save Model
            model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")
            joblib.dump(best_model, model_path)
            self.logger.info(f"Saved model to {model_path}")

        avg_score = np.mean(fold_scores)
        self.logger.info(f"Average CV ROC AUC: {avg_score}")

    def generate_submission(self, debug: bool = False, load_cached_data: bool = True):
        """
        Generates predictions for the test set using the trained fold models.
        Averages the predictions (CV-Bagging) and saves to CSV.
        """
        self.logger.info("Generating submission...")

        # 1. Load Test Data
        _, _, df_test = load_and_process_data(
            debug=debug, load_cached_data=load_cached_data
        )

        # 2. Get Embeddings
        anchor_test = self.embedding_service.get_embeddings(
            df_test, "test", "anchor", load_cached_data
        )
        aux_test = self.embedding_service.get_embeddings(
            df_test, "test", "aux", load_cached_data
        )

        # 3. Construct Feature Matrix
        # Reset index to ensure alignment
        df_test = df_test.reset_index(drop=True)
        X_test, _, _ = self._prepare_feature_matrix(df_test, anchor_test, aux_test)

        # 4. Load Models and Predict
        fold_predictions = []

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")
            if not os.path.exists(model_path):
                self.logger.warning(
                    f"Model for fold {fold} not found at {model_path}. Skipping."
                )
                continue

            self.logger.info(f"Loading model from {model_path}...")
            model = joblib.load(model_path)

            # Predict
            preds = model.predict_proba(X_test)[:, 1]
            fold_predictions.append(preds)

        if not fold_predictions:
            raise RuntimeError("No models found to generate predictions!")

        # 5. Average Predictions
        avg_preds = np.mean(fold_predictions, axis=0)

        # 6. Save Submission
        submission_df = pd.DataFrame(
            {
                Config.ID_COL: df_test[Config.ID_COL],
                Config.TARGET_COL: avg_preds,
            }
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info(f"Submission shape: {submission_df.shape}")
