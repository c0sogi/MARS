import os
import joblib
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger
from library.data_loader import load_and_process_data
from library.embedding_manager import EmbeddingService


class InferenceService:
    """
    Manages the inference process for the DAADBE model.
    Loads trained fold models, generates predictions for the test set,
    and creates the submission file.
    """

    def __init__(self):
        self.logger = setup_logger("inference_service")
        self.embedding_service = EmbeddingService()

    def _prepare_feature_matrix(
        self, df: pd.DataFrame, anchor_emb: np.ndarray, aux_emb: np.ndarray
    ) -> pd.DataFrame:
        """
        Combines metadata dataframe with embedding arrays into a single DataFrame.
        Ensures column names match those expected by the trained pipeline.

        Args:
            df (pd.DataFrame): Metadata dataframe.
            anchor_emb (np.ndarray): Anchor embeddings.
            aux_emb (np.ndarray): Aux embeddings.

        Returns:
            pd.DataFrame: Combined feature matrix.
        """
        # Replicate column naming logic from Trainer
        anchor_cols = [f"anchor_{i}" for i in range(anchor_emb.shape[1])]
        aux_cols = [f"aux_{i}" for i in range(aux_emb.shape[1])]

        # Create DataFrames for embeddings
        df_anchor = pd.DataFrame(anchor_emb, columns=anchor_cols, index=df.index)
        df_aux = pd.DataFrame(aux_emb, columns=aux_cols, index=df.index)

        # Concatenate horizontally
        # Assumes indices are aligned (ensured by reset_index in predict method)
        X = pd.concat([df, df_anchor, df_aux], axis=1)

        return X

    def predict(self, debug: bool = False, load_cached_data: bool = True):
        """
        Executes the inference pipeline: loads data, prepares features,
        runs predictions across all fold models, averages results, and saves submission.

        Args:
            debug (bool): If True, uses a subset of data.
            load_cached_data (bool): If True, attempts to load features/embeddings from cache.
        """
        self.logger.info("Starting inference process...")

        # 1. Load Test Data
        # load_and_process_data returns (train, val, test), we only need test
        _, _, df_test = load_and_process_data(
            debug=debug, load_cached_data=load_cached_data
        )

        # 2. Get Embeddings
        # Retrieve embeddings for the test split
        anchor_test = self.embedding_service.get_embeddings(
            df_test, "test", "anchor", load_cached_data
        )
        aux_test = self.embedding_service.get_embeddings(
            df_test, "test", "aux", load_cached_data
        )

        # 3. Construct Feature Matrix
        # Reset index to ensure alignment between dataframe and numpy arrays
        df_test = df_test.reset_index(drop=True)
        X_test = self._prepare_feature_matrix(df_test, anchor_test, aux_test)

        # 4. Load Models and Predict
        fold_predictions = []
        models_loaded = 0

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")

            if not os.path.exists(model_path):
                self.logger.warning(
                    f"Model file not found: {model_path}. Skipping fold {fold}."
                )
                continue

            self.logger.info(f"Loading model for fold {fold} from {model_path}...")
            try:
                model = joblib.load(model_path)

                # Predict probability of success (Class 1)
                # The pipeline handles all preprocessing (Scaling, PCA, Discretization)
                preds = model.predict_proba(X_test)[:, 1]
                fold_predictions.append(preds)
                models_loaded += 1

            except Exception as e:
                self.logger.error(
                    f"Failed to load or predict with model fold {fold}: {e}"
                )

        if models_loaded == 0:
            raise RuntimeError("No valid models found. Cannot generate predictions.")

        # 5. Average Predictions (CV-Bagging)
        self.logger.info(f"Averaging predictions from {models_loaded} models...")
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
