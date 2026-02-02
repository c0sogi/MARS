import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.utils import setup_logger, set_seed
from library.data_manager import DataManager
from library.embedding_engine import EmbeddingEngine
from library.custom_transformers import build_feature_pipeline


class ModelTrainer:
    def __init__(self, work_dir="./working/idea_33"):
        """
        Initialize the ModelTrainer.

        Args:
            work_dir (str): Directory for working files (models, cache).
        """
        self.work_dir = work_dir
        self.models_dir = os.path.join(work_dir, "models")
        self.submission_dir = "./submission"

        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        self.logger = setup_logger("ModelTrainer")
        self.data_manager = DataManager(cache_dir=work_dir)
        self.embedding_engine = EmbeddingEngine(cache_dir=work_dir)
        self.seed = 42
        set_seed(self.seed)

    def _build_feature_matrix(self, df, split_name):
        """
        Internal helper to construct the [Anchor | Aux | Meta] matrix.
        """
        # 1. Get Texts and Embeddings
        texts = df["text_combined"].tolist()

        # Anchor (MiniLM) - 384d
        anchor_emb = self.embedding_engine.get_anchor_embeddings(
            texts, split_name, load_cached_data=True
        )

        # Aux (MPNet) - 768d
        aux_emb = self.embedding_engine.get_auxiliary_embeddings(
            texts, split_name, load_cached_data=True
        )

        # 2. Get Metadata - ~10d
        meta_cols = self.data_manager.metadata_cols
        # Ensure columns exist and are numeric
        meta_data = df[meta_cols].fillna(0).values.astype(np.float32)

        # 3. Concatenate
        # Expected order: Anchor, Aux, Meta
        X = np.hstack([anchor_emb, aux_emb, meta_data])
        return X

    def prepare_data(self, load_cached_data=True):
        """
        Prepare the training and testing data matrices.
        Merges DataManager's train and val splits into a single training set for CV.

        Args:
            load_cached_data (bool): Whether to load pre-computed matrices from disk.

        Returns:
            tuple: (X_train, y_train, X_test, test_ids)
        """
        cache_X_train = os.path.join(self.work_dir, "X_train_full.npy")
        cache_y_train = os.path.join(self.work_dir, "y_train_full.npy")
        cache_X_test = os.path.join(self.work_dir, "X_test.npy")
        cache_test_ids = os.path.join(self.work_dir, "test_ids.npy")

        # 1. Try Loading from Cache
        if load_cached_data:
            if (
                os.path.exists(cache_X_train)
                and os.path.exists(cache_y_train)
                and os.path.exists(cache_X_test)
                and os.path.exists(cache_test_ids)
            ):
                self.logger.info("Loading feature matrices from cache...")
                try:
                    X_train = np.load(cache_X_train)
                    y_train = np.load(cache_y_train)
                    X_test = np.load(cache_X_test)
                    test_ids = np.load(cache_test_ids, allow_pickle=True)
                    return X_train, y_train, X_test, test_ids
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load feature cache: {e}. Recomputing..."
                    )
            else:
                self.logger.info("Feature cache not found. Computing from scratch...")

        # 2. Compute from Scratch
        self.logger.info("Loading raw dataframes...")
        train_df, val_df, test_df = self.data_manager.load_dataset(
            load_cached_data=load_cached_data
        )

        self.logger.info("Constructing feature matrices...")

        # Process Train and Val separately then merge
        X_train_part = self._build_feature_matrix(train_df, "train")
        y_train_part = train_df["requester_received_pizza"].values.astype(int)

        X_val_part = self._build_feature_matrix(val_df, "val")
        y_val_part = val_df["requester_received_pizza"].values.astype(int)

        # Merge for Full CV
        X_train_full = np.vstack([X_train_part, X_val_part])
        y_train_full = np.concatenate([y_train_part, y_val_part])

        # Process Test
        X_test = self._build_feature_matrix(test_df, "test")
        test_ids = test_df["request_id"].values

        # 3. Save to Cache
        self.logger.info("Saving feature matrices to cache...")
        np.save(cache_X_train, X_train_full)
        np.save(cache_y_train, y_train_full)
        np.save(cache_X_test, X_test)
        np.save(cache_test_ids, test_ids)

        return X_train_full, y_train_full, X_test, test_ids

    def train_loop(self, n_folds=5):
        """
        Execute the Stratified K-Fold training loop with GridSearch.
        """
        X, y, _, _ = self.prepare_data(load_cached_data=True)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)

        oof_preds = np.zeros(len(y))
        fold_aucs = []

        # Memory for caching pipeline transformers during GridSearch
        memory = joblib.Memory(location=self.work_dir, verbose=0)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.logger.info(f"Starting Fold {fold + 1}/{n_folds}...")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Define Pipeline
            # 1. Feature Engineering (PCA, GMM, Scaling)
            # 2. Classifier (Bagged Logistic Regression)

            # Dimensions
            anchor_dim = 384
            aux_dim = 768
            meta_dim = len(self.data_manager.metadata_cols)

            # Adjust PCA components for small datasets (Cite debug_lesson_8)
            # Inner CV is 3, so we have approx 2/3 of X_train for training
            n_inner_train_samples = int(len(X_train) * (2 / 3))
            pca_components = min(50, n_inner_train_samples)
            # Ensure at least 1 component
            pca_components = max(1, pca_components)

            feature_pipeline = build_feature_pipeline(
                anchor_dim=anchor_dim,
                aux_dim=aux_dim,
                meta_dim=meta_dim,
                seed=self.seed,
                pca_components=pca_components,
            )

            # Bagging Classifier wrapping Logistic Regression
            # Note: We tune the inner LR parameters via the BaggingClassifier
            clf = BaggingClassifier(
                estimator=LogisticRegression(
                    solver="liblinear", random_state=self.seed
                ),
                n_estimators=20,
                random_state=self.seed,
                n_jobs=1,  # Sequential bagging, parallel grid search
            )

            pipeline = Pipeline(
                [("features", feature_pipeline), ("clf", clf)], memory=memory
            )

            # Grid Search
            # We tune the C and class_weight of the base estimator inside the bag
            param_grid = {
                "clf__estimator__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                "clf__estimator__class_weight": [None, "balanced"],
            }

            grid = GridSearchCV(
                pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=4, verbose=0
            )

            self.logger.info("Running Grid Search...")
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            self.logger.info(f"Best Params: {grid.best_params_}")

            # Evaluate
            y_pred_val = best_model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_val)
            fold_aucs.append(auc)
            oof_preds[val_idx] = y_pred_val

            self.logger.info(f"Fold {fold + 1} AUC: {auc:.8f}")

            # Save Model
            model_path = os.path.join(self.models_dir, f"model_fold_{fold}.joblib")
            joblib.dump(best_model, model_path)

        avg_auc = np.mean(fold_aucs)
        oof_auc = roc_auc_score(y, oof_preds)
        self.logger.info(f"Training Complete. Average AUC: {avg_auc:.8f}")
        self.logger.info(f"OOF AUC: {oof_auc:.8f}")

        # Clean up memory cache
        memory.clear(warn=False)

    def generate_submission(self, n_folds=5):
        """
        Generate predictions for the test set using trained models and save submission.
        """
        _, _, X_test, test_ids = self.prepare_data(load_cached_data=True)

        test_preds = np.zeros((len(X_test), n_folds))

        self.logger.info("Generating predictions for test set...")

        for fold in range(n_folds):
            model_path = os.path.join(self.models_dir, f"model_fold_{fold}.joblib")
            if not os.path.exists(model_path):
                self.logger.error(f"Model file {model_path} not found.")
                continue

            model = joblib.load(model_path)
            # Predict
            preds = model.predict_proba(X_test)[:, 1]
            test_preds[:, fold] = preds

        # Average predictions (Bagging)
        avg_preds = np.mean(test_preds, axis=1)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": avg_preds}
        )

        # Save
        submission_path = os.path.join(self.submission_dir, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        self.logger.info(f"Submission saved to {submission_path}")

    def run(self):
        """
        Main execution method.
        """
        self.train_loop()
        self.generate_submission()
