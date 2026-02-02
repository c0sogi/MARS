import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, setup_logger, save_to_cache, load_from_cache
from library.data_loader import DataLoader
from library.embedding_engine import EmbeddingEngine
from library.model_builder import get_model_pipeline

# Initialize Logger
logger = setup_logger("engine", os.path.join(Config.WORKING_DIR, "engine.log"))


class Engine:
    def __init__(self):
        """
        Initializes the Engine with DataLoader and EmbeddingEngine.
        """
        self.data_loader = DataLoader()
        self.embedding_engine = EmbeddingEngine()
        self.models = []  # List to store trained models from CV

    def _assemble_features(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ):
        """
        Generates and concatenates features for the OASF pipeline.
        Structure: [Anchor Embeddings (384) | Aux Embeddings (768) | Metadata]

        Args:
            df (pd.DataFrame): Dataframe containing text and metadata.
            split_name (str): Name of the split (e.g., 'train_full', 'test') for caching keys.
            load_cached_data (bool): Whether to use cached assembled features.

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target array (or None).
        """
        # Define cache paths for the assembled matrix
        cache_path_X = os.path.join(Config.WORKING_DIR, f"X_{split_name}_assembled.npy")
        cache_path_y = os.path.join(Config.WORKING_DIR, f"y_{split_name}_assembled.npy")

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path_X):
            X = load_from_cache(cache_path_X)
            # Check for y if it's expected (i.e., not test)
            if os.path.exists(cache_path_y):
                y = load_from_cache(cache_path_y)
            else:
                y = None

            # If we expect y (based on dataframe columns) but didn't get it, we might need to recompute.
            # However, for simplicity, if X is cached, we assume valid state or user intent.
            # We'll just check if df has target to decide if we should have loaded y.
            if Config.TARGET_COL in df.columns and y is None:
                logger.warning(
                    f"Cached X found for {split_name} but y is missing. Recomputing."
                )
            else:
                logger.info(f"Loaded assembled features for {split_name} from cache.")
                return X, y

        logger.info(f"Assembling features for {split_name}...")

        # 2. Generate/Load Embeddings
        texts = df["text_combined"].tolist()

        # Anchor Backbone (MiniLM)
        emb_anchor = self.embedding_engine.get_anchor_embeddings(
            texts, split_name, load_cached_data
        )

        # Auxiliary Backbone (MPNet)
        emb_aux = self.embedding_engine.get_aux_embeddings(
            texts, split_name, load_cached_data
        )

        # 3. Extract Metadata
        # Ensure strict column ordering as defined in Config
        meta_features = df[Config.METADATA_COLS].values.astype(np.float32)

        # 4. Concatenate Views
        # Order MUST be: [Anchor | Aux | Meta] for OASFPreprocessor
        X = np.hstack([emb_anchor, emb_aux, meta_features])

        # 5. Extract Target
        y = None
        if Config.TARGET_COL in df.columns:
            y = df[Config.TARGET_COL].values.astype(int)

        # 6. Save to Cache
        save_to_cache(X, cache_path_X)
        if y is not None:
            save_to_cache(y, cache_path_y)

        logger.info(f"Feature assembly complete. Shape: {X.shape}")
        return X, y

    def run_cross_validation(self, load_cached_data: bool = True):
        """
        Performs Stratified K-Fold Cross Validation.
        Merges 'train' and 'val' splits from metadata to use the full labeled dataset.
        """
        set_seed(Config.RANDOM_SEED)

        # 1. Load and Combine Data
        logger.info("Loading training data...")
        df_train_part = self.data_loader.load_dataset("train", load_cached_data)
        df_val_part = self.data_loader.load_dataset("val", load_cached_data)

        # Concatenate to form full training set
        df_full = pd.concat([df_train_part, df_val_part], axis=0).reset_index(drop=True)

        # 2. Assemble Features
        X, y = self._assemble_features(df_full, "train_full", load_cached_data)

        if y is None:
            raise ValueError("Target variable missing in training data.")

        # 3. Setup Cross-Validation
        skf = StratifiedKFold(
            n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_SEED
        )

        oof_preds = np.zeros(len(y))
        fold_aucs = []
        self.models = []  # Reset models list

        logger.info(f"Starting {Config.N_SPLITS}-Fold Stratified CV...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"--- Fold {fold + 1} / {Config.N_SPLITS} ---")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 4. Model & Hyperparameter Tuning
            pipeline, param_grid = get_model_pipeline(random_state=Config.RANDOM_SEED)

            # Nested Grid Search
            # We use a smaller inner CV (e.g., 3) to save time while tuning
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=3,
                n_jobs=Config.N_JOBS,
                verbose=0,
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            self.models.append(best_model)

            # Save Model
            model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")
            joblib.dump(best_model, model_path)

            # 5. Validation
            y_pred = best_model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = y_pred

            auc = roc_auc_score(y_val, y_pred)
            fold_aucs.append(auc)

            logger.info(f"Fold {fold + 1} Best Params: {grid.best_params_}")
            print(f"Fold {fold + 1} AUC: {auc}")

        # 6. Overall Metrics
        overall_auc = roc_auc_score(y, oof_preds)
        print(f"Overall CV AUC: {overall_auc}")

        # Save OOF predictions
        np.save(os.path.join(Config.WORKING_DIR, "oof_preds.npy"), oof_preds)
        logger.info("Cross-validation complete.")

    def predict_test(self, load_cached_data: bool = True):
        """
        Generates predictions for the test set using the ensemble of trained fold models.
        """
        logger.info("Starting inference on test set...")

        # 1. Load Test Data
        df_test = self.data_loader.load_dataset("test", load_cached_data)
        X_test, _ = self._assemble_features(df_test, "test", load_cached_data)

        # 2. Load Models (if not in memory)
        if not self.models:
            logger.info("Loading models from disk...")
            for fold in range(Config.N_SPLITS):
                model_path = os.path.join(
                    Config.WORKING_DIR, f"model_fold_{fold}.joblib"
                )
                if os.path.exists(model_path):
                    self.models.append(joblib.load(model_path))
                else:
                    raise FileNotFoundError(
                        f"Model for fold {fold} not found. Run CV first."
                    )

        # 3. Generate Predictions (Bagging)
        avg_preds = np.zeros(len(X_test))

        for i, model in enumerate(self.models):
            preds = model.predict_proba(X_test)[:, 1]
            avg_preds += preds

        avg_preds /= len(self.models)

        # 4. Create Submission File
        submission = pd.DataFrame(
            {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: avg_preds}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission generated with {len(self.models)} models.")
