import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import SUBMISSION_PATH, N_FOLDS, RANDOM_SEED, TEST_PATH
from library.model_definitions import (
    get_lexical_bagger,
    get_community_bagger,
    get_semantic_booster,
    get_semantic_bagger,
    get_manifold_neighbor,
    get_metadata_anchor,
    get_meta_learner,
)


class StackingEnsemble:
    def __init__(self):
        self.n_folds = N_FOLDS
        self.seed = RANDOM_SEED

        # Initialize base models container
        self.base_models = {
            "lexical_rf": get_lexical_bagger(),
            "community_rf": get_community_bagger(),
            "semantic_xgb": get_semantic_booster(),
            "semantic_rf": get_semantic_bagger(),
            "manifold_knn": get_manifold_neighbor(),
            "contextual_lr": get_metadata_anchor(),
        }

        # Initialize meta learner
        self.meta_learner = get_meta_learner()

        # Mapping models to their required feature view
        self.model_view_map = {
            "lexical_rf": "lexical",
            "community_rf": "behavioral",
            "semantic_xgb": "dense",
            "semantic_rf": "dense",
            "manifold_knn": "dense",
            "contextual_lr": "contextual",
        }

    def _construct_feature_view(self, data, split, view_type):
        """
        Constructs the specific feature matrix for a given view type and split.

        Args:
            data (dict): The dictionary returned by FeatureFactory.process_data
            split (str): 'train', 'val', or 'test'
            view_type (str): 'lexical', 'behavioral', 'dense', or 'contextual'

        Returns:
            np.array or scipy.sparse.csr_matrix: The feature matrix
        """
        # Retrieve components
        metadata = data[f"X_{split}_metadata"]  # Dense

        if view_type == "lexical":
            # Sparse Text TFIDF + Dense Metadata
            lexical_sparse = data[f"X_{split}_lexical"]
            return sp.hstack([lexical_sparse, metadata], format="csr")

        elif view_type == "behavioral":
            # Sparse Subreddit TFIDF + Dense Metadata
            behavioral_sparse = data[f"X_{split}_behavioral"]
            return sp.hstack([behavioral_sparse, metadata], format="csr")

        elif view_type == "dense":
            # Dense Text Emb + Dense Sub Emb + Dense Metadata
            text_emb = data[f"X_{split}_text_emb"]
            sub_emb = data[f"X_{split}_sub_emb"]
            return np.hstack([text_emb, sub_emb, metadata])

        elif view_type == "contextual":
            # Just Metadata
            return metadata

        else:
            raise ValueError(f"Unknown view_type: {view_type}")

    def generate_oof_predictions(self, data):
        """
        Performs 5-Fold Stratified CV on the training set to generate OOF predictions.

        Args:
            data (dict): Data dictionary from FeatureFactory

        Returns:
            pd.DataFrame: OOF predictions (N_train x N_models)
        """
        print(f"Generating OOF predictions with {self.n_folds}-Fold CV...")

        y_train = data["y_train"]
        n_samples = len(y_train)
        model_names = list(self.base_models.keys())

        # Initialize OOF matrix
        oof_preds = pd.DataFrame(
            np.zeros((n_samples, len(model_names))), columns=model_names
        )

        # Pre-compute training feature views to avoid re-stacking in loop
        train_views = {}
        unique_views = set(self.model_view_map.values())
        for view in unique_views:
            train_views[view] = self._construct_feature_view(data, "train", view)

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_train)
        ):
            print(f"  Processing Fold {fold + 1}/{self.n_folds}...")

            y_fold_train = y_train[train_idx]
            # y_fold_val = y_train[val_idx] # Not needed for prediction, just indexing

            for name, model in self.base_models.items():
                view_type = self.model_view_map[name]
                X_view = train_views[view_type]

                # Slice data
                X_fold_train = X_view[train_idx]
                X_fold_val = X_view[val_idx]

                # Clone model (re-instantiate to reset)
                # We use the factory functions again to get a fresh instance
                if name == "lexical_rf":
                    clf = get_lexical_bagger()
                elif name == "community_rf":
                    clf = get_community_bagger()
                elif name == "semantic_xgb":
                    clf = get_semantic_booster()
                elif name == "semantic_rf":
                    clf = get_semantic_bagger()
                elif name == "manifold_knn":
                    clf = get_manifold_neighbor()
                elif name == "contextual_lr":
                    clf = get_metadata_anchor()

                # Train
                # Note: No early stopping in OOF generation for simplicity/consistency with standard stacking,
                # or we could use a portion of fold_train. Standard RF/LR/KNN don't use it.
                # For XGB in OOF, we fit on fold_train.
                if name == "semantic_xgb":
                    # For OOF, we can't easily do early stopping without splitting fold_train further.
                    # We will fit normally.
                    clf.set_params(early_stopping_rounds=None)
                    clf.fit(X_fold_train, y_fold_train, verbose=False)
                else:
                    clf.fit(X_fold_train, y_fold_train)

                # Predict
                preds = clf.predict_proba(X_fold_val)[:, 1]
                oof_preds.iloc[val_idx, oof_preds.columns.get_loc(name)] = preds

        # Calculate scores
        print("\nBase Model OOF Scores (AUC):")
        for name in model_names:
            score = roc_auc_score(y_train, oof_preds[name])
            print(f"  {name}: {score:.16f}")

        return oof_preds

    def train_meta_learner(self, oof_preds, y_train):
        """
        Trains the Level 2 Meta-Learner on OOF predictions.
        """
        print("\nTraining Meta-Learner...")
        self.meta_learner.fit(oof_preds, y_train)

        # Check coefficients to see contribution
        coefs = self.meta_learner.coef_[0]
        print("Meta-Learner Coefficients:")
        for name, coef in zip(oof_preds.columns, coefs):
            print(f"  {name}: {coef:.4f}")

        # In-sample score (just for sanity check)
        meta_preds = self.meta_learner.predict_proba(oof_preds)[:, 1]
        score = roc_auc_score(y_train, meta_preds)
        print(f"Meta-Learner OOF CV Score (AUC): {score:.16f}")

    def retrain_base_models(self, data):
        """
        Retrains base models on the full dataset (Train + Val) using the Validation-Guided protocol.
        """
        print("\nRetraining Base Models on Full Data...")

        y_train = data["y_train"]
        y_val = data["y_val"]

        # Combine targets for non-XGB models
        y_full = np.concatenate([y_train, y_val])

        unique_views = set(self.model_view_map.values())

        for name, model in self.base_models.items():
            print(f"  Retraining {name}...")
            view_type = self.model_view_map[name]

            # Construct feature views
            X_train_view = self._construct_feature_view(data, "train", view_type)
            X_val_view = self._construct_feature_view(data, "val", view_type)

            if name == "semantic_xgb":
                # XGBoost Protocol: Train on Train, use Val for Early Stopping
                # This prevents blind overfitting.
                model.fit(
                    X_train_view, y_train, eval_set=[(X_val_view, y_val)], verbose=False
                )
            else:
                # RF / KNN / LR Protocol: Train on Train + Val
                if sp.issparse(X_train_view):
                    X_full = sp.vstack([X_train_view, X_val_view], format="csr")
                else:
                    X_full = np.vstack([X_train_view, X_val_view])

                model.fit(X_full, y_full)

    def predict(self, data):
        """
        Generates predictions for the test set.
        1. Base models predict on test features.
        2. Meta learner aggregates base predictions.
        3. Saves to submission file.
        """
        print("\nGenerating Final Predictions...")

        model_names = list(self.base_models.keys())
        n_test = data["X_test_metadata"].shape[0]

        base_test_preds = pd.DataFrame(
            np.zeros((n_test, len(model_names))), columns=model_names
        )

        # Generate Base Predictions
        for name, model in self.base_models.items():
            view_type = self.model_view_map[name]
            X_test_view = self._construct_feature_view(data, "test", view_type)

            preds = model.predict_proba(X_test_view)[:, 1]
            base_test_preds[name] = preds

        # Generate Meta Predictions
        final_probs = self.meta_learner.predict_proba(base_test_preds)[:, 1]

        # Load Test IDs
        # We need to read the test parquet file to get the IDs
        if not os.path.exists(TEST_PATH):
            raise FileNotFoundError(f"Test metadata not found at {TEST_PATH}")

        test_df = pd.read_parquet(TEST_PATH)
        request_ids = test_df["request_id"].values

        if len(request_ids) != len(final_probs):
            raise ValueError(
                f"ID count ({len(request_ids)}) matches prediction count ({len(final_probs)})"
            )

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"request_id": request_ids, "requester_received_pizza": final_probs}
        )

        # Save
        print(f"Saving submission to {SUBMISSION_PATH}...")
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print("Done.")
