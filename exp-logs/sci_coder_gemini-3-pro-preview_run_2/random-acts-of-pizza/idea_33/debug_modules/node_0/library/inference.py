import os
import numpy as np
import pandas as pd
import joblib
from library.utils import setup_logger, set_seed
from library.data_manager import DataManager
from library.embedding_engine import EmbeddingEngine

# Import custom transformers to ensure they are available in the namespace for joblib loading
from library.custom_transformers import TensorSlicer, GMMTransformer


class InferenceManager:
    def __init__(self, work_dir="./working/idea_33"):
        """
        Initialize the InferenceManager.

        Args:
            work_dir (str): Directory where models and cached data are stored.
        """
        self.work_dir = work_dir
        self.models_dir = os.path.join(work_dir, "models")
        self.submission_dir = "./submission"

        # Ensure directories exist
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        self.logger = setup_logger("InferenceManager")
        self.data_manager = DataManager(cache_dir=work_dir)
        self.embedding_engine = EmbeddingEngine(cache_dir=work_dir)
        self.seed = 42
        set_seed(self.seed)

    def _build_test_features(self, test_df):
        """
        Construct the feature matrix for the test set.
        Must match the structure used in training: [Anchor | Aux | Meta]
        """
        texts = test_df["text_combined"].tolist()

        # 1. Anchor Embeddings (MiniLM)
        anchor_emb = self.embedding_engine.get_anchor_embeddings(
            texts, "test", load_cached_data=True
        )

        # 2. Auxiliary Embeddings (MPNet)
        aux_emb = self.embedding_engine.get_auxiliary_embeddings(
            texts, "test", load_cached_data=True
        )

        # 3. Metadata
        meta_cols = self.data_manager.metadata_cols
        meta_data = test_df[meta_cols].fillna(0).values.astype(np.float32)

        # Concatenate features
        X = np.hstack([anchor_emb, aux_emb, meta_data])
        return X

    def predict(self, load_cached_data=True, n_folds=5):
        """
        Generate predictions for the test set using trained models.

        Args:
            load_cached_data (bool): If True, try to load pre-computed X_test from disk.
            n_folds (int): Number of fold models to use for bagging.

        Returns:
            pd.DataFrame: The submission dataframe.
        """
        cache_X_test = os.path.join(self.work_dir, "X_test.npy")
        cache_test_ids = os.path.join(self.work_dir, "test_ids.npy")

        X_test = None
        test_ids = None

        # 1. Load or Compute Test Features
        if (
            load_cached_data
            and os.path.exists(cache_X_test)
            and os.path.exists(cache_test_ids)
        ):
            self.logger.info("Loading test features from cache...")
            try:
                X_test = np.load(cache_X_test)
                test_ids = np.load(cache_test_ids, allow_pickle=True)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        if X_test is None:
            self.logger.info("Computing test features from scratch...")
            # Load test dataframe (ignoring train/val)
            _, _, test_df = self.data_manager.load_dataset(
                load_cached_data=load_cached_data
            )

            test_ids = test_df["request_id"].values
            X_test = self._build_test_features(test_df)

            # Save to cache
            self.logger.info("Saving test features to cache...")
            np.save(cache_X_test, X_test)
            np.save(cache_test_ids, test_ids)

        # 2. Inference Loop (Bagging)
        self.logger.info(f"Starting inference using {n_folds} folds...")
        fold_preds = []

        for fold in range(n_folds):
            model_path = os.path.join(self.models_dir, f"model_fold_{fold}.joblib")

            if not os.path.exists(model_path):
                self.logger.warning(
                    f"Model file {model_path} not found. Skipping fold."
                )
                continue

            try:
                self.logger.info(f"Loading model fold {fold}...")
                model = joblib.load(model_path)

                # Predict probability of success (class 1)
                preds = model.predict_proba(X_test)[:, 1]
                fold_preds.append(preds)
            except Exception as e:
                self.logger.error(f"Error predicting with fold {fold}: {e}")

        if not fold_preds:
            raise RuntimeError(
                "No predictions generated. Ensure models are trained and saved."
            )

        # Average predictions across folds
        avg_preds = np.mean(fold_preds, axis=0)

        # 3. Generate Submission File
        submission_df = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": avg_preds}
        )

        submission_path = os.path.join(self.submission_dir, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        self.logger.info(f"Submission saved to {submission_path}")

        return submission_df
