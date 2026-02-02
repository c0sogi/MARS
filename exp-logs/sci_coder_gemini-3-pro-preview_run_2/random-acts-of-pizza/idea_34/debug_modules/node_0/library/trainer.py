import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.utils import set_seed, save_submission
from library.data_loader import DataLoader
from library.embedding_manager import EmbeddingManager
from library.feature_engineering import build_jbpce_pipeline


class Trainer:
    """
    Orchestrates the training and prediction process using the JBPCE strategy.
    """

    def __init__(self, cache_dir="./working/idea_34"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        set_seed(42)

        self.data_loader = DataLoader(cache_dir=cache_dir)
        self.embedding_manager = EmbeddingManager(cache_dir=cache_dir)

    def prepare_data(self, load_cached_data=True, debug_limit=None):
        """
        Loads data, computes embeddings, and constructs feature matrices.
        """
        print("Loading raw data and metadata...")
        train_json, test_json, train_meta, val_meta, test_meta = (
            self.data_loader.load_data(debug_limit=debug_limit)
        )

        # Extract Texts
        train_texts = self.data_loader.extract_text_data(train_json)
        test_texts = self.data_loader.extract_text_data(test_json)

        # Extract Metadata
        train_meta_feats = self.data_loader.extract_metadata(train_json)
        test_meta_feats = self.data_loader.extract_metadata(test_json)

        # Compute/Load Embeddings
        print("Retrieving dual-backbone embeddings...")
        train_emb_a, train_emb_b = self.embedding_manager.get_dual_backbone_embeddings(
            train_texts, "train", load_cached=load_cached_data
        )
        test_emb_a, test_emb_b = self.embedding_manager.get_dual_backbone_embeddings(
            test_texts, "test", load_cached=load_cached_data
        )

        # Map data by request_id for alignment
        # We need to reconstruct the arrays based on the metadata splits
        # because train_json contains both train and val samples (it's the full input/train.json)

        # Create a lookup dictionary for training data (from input/train.json)
        train_lookup = {}
        for i, entry in enumerate(train_json):
            rid = entry["request_id"]
            train_lookup[rid] = {
                "emb_a": train_emb_a[i],
                "emb_b": train_emb_b[i],
                "meta": train_meta_feats[i],
                "y": int(entry["requester_received_pizza"]),
            }

        # Create a lookup dictionary for test data
        test_lookup = {}
        for i, entry in enumerate(test_json):
            rid = entry["request_id"]
            test_lookup[rid] = {
                "emb_a": test_emb_a[i],
                "emb_b": test_emb_b[i],
                "meta": test_meta_feats[i],
            }

        # Helper to build arrays from metadata dataframe
        def build_arrays(meta_df, lookup_dict, is_test=False):
            X_emb_a_list, X_emb_b_list, X_meta_list, y_list, ids_list = (
                [],
                [],
                [],
                [],
                [],
            )

            for _, row in meta_df.iterrows():
                rid = row["request_id"]
                if rid in lookup_dict:
                    data = lookup_dict[rid]
                    X_emb_a_list.append(data["emb_a"])
                    X_emb_b_list.append(data["emb_b"])
                    X_meta_list.append(data["meta"])
                    ids_list.append(rid)
                    if not is_test:
                        y_list.append(data["y"])

            X_emb_a = np.array(X_emb_a_list, dtype=np.float32)
            X_emb_b = np.array(X_emb_b_list, dtype=np.float32)
            X_meta = np.array(X_meta_list, dtype=np.float32)

            if is_test:
                return X_emb_a, X_emb_b, X_meta, None, ids_list
            else:
                return X_emb_a, X_emb_b, X_meta, np.array(y_list, dtype=int), ids_list

        # Build Train and Val arrays
        tr_emb_a, tr_emb_b, tr_meta, tr_y, _ = build_arrays(train_meta, train_lookup)
        val_emb_a, val_emb_b, val_meta, val_y, _ = build_arrays(val_meta, train_lookup)

        # Build Test arrays
        te_emb_a, te_emb_b, te_meta, _, te_ids = build_arrays(
            test_meta, test_lookup, is_test=True
        )

        # Merge Train and Val for Stratified CV
        X_full_emb_a = np.vstack([tr_emb_a, val_emb_a])
        X_full_emb_b = np.vstack([tr_emb_b, val_emb_b])
        X_full_meta = np.vstack([tr_meta, val_meta])
        y_full = np.hstack([tr_y, val_y])

        # Concatenate features: [Emb_A | Emb_B | Meta]
        # This structure is required by the ColumnTransformer in feature_engineering.py
        X_full = np.hstack([X_full_emb_a, X_full_emb_b, X_full_meta])
        X_test = np.hstack([te_emb_a, te_emb_b, te_meta])

        dims = {
            "emb_a": tr_emb_a.shape[1],
            "emb_b": tr_emb_b.shape[1],
            "meta": tr_meta.shape[1],
        }

        print(
            f"Data prepared. Full training shape: {X_full.shape}, Test shape: {X_test.shape}"
        )
        return X_full, y_full, X_test, te_ids, dims

    def run_stratified_cv(self, X, y, X_test, dims, n_splits=5):
        """
        Runs Stratified K-Fold CV with nested GridSearchCV.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Store test predictions from each fold
        test_preds_sum = np.zeros(len(X_test))
        oof_preds = np.zeros(len(X))
        fold_scores = []

        # Parameter Grid for Tuning
        param_grid = {
            # Tuning PCA components for the Joint Embedding Space
            "preprocessor__emb_joint_pca__n_components": [100, 150, 200],
            # Tuning Logistic Regression Regularization
            "clf__estimator__C": [0.1, 1.0, 10.0],
            # Tuning Class Weights
            "clf__estimator__class_weight": ["balanced", None],
            # Fixed Ensemble parameters
            "clf__n_estimators": [20],
            "clf__max_samples": [1.0],
            "clf__bootstrap": [True],
        }

        print(f"Starting {n_splits}-Fold Stratified CV...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Build base pipeline
            pipeline = build_jbpce_pipeline(
                emb_dim_a=dims["emb_a"], emb_dim_b=dims["emb_b"], random_state=42
            )

            # Grid Search
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,  # Internal CV for hyperparam tuning
                scoring="roc_auc",
                n_jobs=4,  # Use available cores
                verbose=0,
            )

            grid.fit(X_train_fold, y_train_fold)

            best_model = grid.best_estimator_
            best_score = grid.best_score_
            print(f"Best Internal CV AUC: {best_score}")
            print(f"Best Params: {grid.best_params_}")

            # Validation on the held-out fold
            val_probs = best_model.predict_proba(X_val_fold)[:, 1]
            fold_auc = roc_auc_score(y_val_fold, val_probs)
            print(f"Fold {fold + 1} Validation AUC: {fold_auc}")

            oof_preds[val_idx] = val_probs
            fold_scores.append(fold_auc)

            # Inference on Test Set (CV-Bagging)
            test_probs = best_model.predict_proba(X_test)[:, 1]
            test_preds_sum += test_probs

        # Calculate average test predictions
        avg_test_preds = test_preds_sum / n_splits

        # Calculate overall OOF AUC
        overall_auc = roc_auc_score(y, oof_preds)
        print("\n=== Training Complete ===")
        print(f"Average Fold AUC: {np.mean(fold_scores)}")
        print(f"Overall OOF AUC: {overall_auc}")

        return avg_test_preds

    def train_and_predict(self, load_cached_data=True, debug_limit=None):
        """
        Main execution method.
        """
        # 1. Prepare Data
        X_full, y_full, X_test, test_ids, dims = self.prepare_data(
            load_cached_data=load_cached_data, debug_limit=debug_limit
        )

        # 2. Run CV and Inference
        final_probs = self.run_stratified_cv(X_full, y_full, X_test, dims)

        # 3. Save Submission
        print("Saving submission...")
        save_submission(
            test_ids, final_probs, output_path="./submission/submission.csv"
        )
