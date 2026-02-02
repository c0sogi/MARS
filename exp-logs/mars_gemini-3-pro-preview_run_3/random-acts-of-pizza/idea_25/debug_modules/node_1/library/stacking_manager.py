import os
import copy
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.features import FeatureEngine
from library.model_factory import ModelFactory


class StackingManager:
    """
    Manages the Hex-View Hybrid-Topology Stacking Ensemble.
    Handles data preparation, cross-validation, meta-learning, and final prediction.
    """

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.base_models = ModelFactory.get_base_models()
        self.meta_model = ModelFactory.get_meta_model()

    def _hstack(self, matrices):
        """
        Concatenates matrices horizontally, handling both sparse and dense inputs.
        """
        if any(sp.issparse(m) for m in matrices):
            return sp.hstack(matrices).tocsr()
        return np.hstack(matrices)

    def _vstack(self, matrices):
        """
        Concatenates matrices vertically, handling both sparse and dense inputs.
        """
        if any(sp.issparse(m) for m in matrices):
            return sp.vstack(matrices).tocsr()
        return np.vstack(matrices)

    def _get_model_input(self, features_dict, model_name):
        """
        Selects and concatenates specific feature views based on the model topology.

        Args:
            features_dict (dict): Dictionary of feature matrices (lexical, behavioral, etc.).
            model_name (str): Name of the model to prepare input for.

        Returns:
            Matrix: The concatenated feature matrix for the model.
        """
        X_meta = features_dict["metadata"]

        if model_name == "LexicalBagger":
            # Sparse Lexical Branch: Text TF-IDF + Metadata
            return self._hstack([features_dict["lexical"], X_meta])

        elif model_name == "BehavioralBagger":
            # Sparse Behavioral Branch: History TF-IDF + Metadata
            return self._hstack([features_dict["behavioral"], X_meta])

        elif model_name in ["SemanticBooster", "SemanticBagger"]:
            # Dense Semantic Branch: Embeddings + Metadata
            return self._hstack([features_dict["semantic"], X_meta])

        elif model_name == "ManifoldNeighbor":
            # Manifold Neighbor Branch: PCA Embeddings + Metadata
            return self._hstack([features_dict["manifold"], X_meta])

        elif model_name == "ContextualAnchor":
            # Contextual Branch: Metadata only
            return X_meta

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def train_and_predict(self, debug_sample_size=None):
        """
        Executes the full pipeline:
        1. Feature Generation
        2. Level 1 Cross-Validation
        3. Level 2 Meta-Training
        4. Final Retraining (Validation-Guided)
        5. Test Prediction & Submission
        """

        # =========================================================================
        # 1. Feature Generation
        # =========================================================================
        print("Generating features for Train split...")
        feats_train, y_train = self.feature_engine.fit_transform(
            split="train", load_cached_data=True, debug_sample_size=debug_sample_size
        )

        print("Generating features for Validation split...")
        feats_val, y_val = self.feature_engine.transform(
            split="val", load_cached_data=True, debug_sample_size=debug_sample_size
        )

        print("Generating features for Test split...")
        feats_test, _ = self.feature_engine.transform(
            split="test", load_cached_data=True, debug_sample_size=debug_sample_size
        )

        # =========================================================================
        # 2. Level 1 Cross-Validation (OOF Generation)
        # =========================================================================
        print("Starting Level 1 Cross-Validation...")
        n_train = len(y_train)
        oof_preds = pd.DataFrame(index=range(n_train))

        # Initialize Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE
        )

        for name, model in self.base_models.items():
            print(f"Cross-validating {name}...")
            X_model = self._get_model_input(feats_train, name)
            model_oof = np.zeros(n_train)

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_model, y_train)):
                X_tr, y_tr = X_model[train_idx], y_train[train_idx]
                X_va, y_va = X_model[val_idx], y_train[val_idx]

                # Clone model to ensure fresh start for each fold
                clf = copy.deepcopy(model)

                # Fit with specific logic for XGBoost
                if name == "SemanticBooster":
                    # Use fold validation set for early stopping
                    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                else:
                    clf.fit(X_tr, y_tr)

                # Predict probabilities
                preds = clf.predict_proba(X_va)[:, 1]
                model_oof[val_idx] = preds

            # Store OOF predictions
            oof_preds[name] = model_oof

            # Calculate and print metric (Full Precision)
            auc = roc_auc_score(y_train, model_oof)
            print(f"{name} CV AUC: {auc}")

        # =========================================================================
        # 3. Level 2 Meta-Learner Training
        # =========================================================================
        print("Training Meta-Learner on OOF predictions...")
        self.meta_model.fit(oof_preds, y_train)

        # Check Meta-Learner performance on OOF (sanity check)
        meta_oof_auc = roc_auc_score(
            y_train, self.meta_model.predict_proba(oof_preds)[:, 1]
        )
        print(f"Meta-Learner OOF AUC: {meta_oof_auc}")

        # =========================================================================
        # 4. Final Retraining (Validation-Guided)
        # =========================================================================
        print("Retraining base models on Full Training Set (Train + Val)...")

        # Concatenate Train and Val features
        feats_full = {}
        for key in feats_train:
            feats_full[key] = self._vstack([feats_train[key], feats_val[key]])

        y_full = np.concatenate([y_train, y_val])

        for name, model in self.base_models.items():
            print(f"Retraining {name}...")
            X_full_input = self._get_model_input(feats_full, name)

            if name == "SemanticBooster":
                # Validation-Guided Retraining for XGBoost:
                # Train on Full (Train+Val) but use Val for early stopping to control complexity.
                # We need the Val-only input for eval_set.
                X_val_input = self._get_model_input(feats_val, name)

                model.fit(
                    X_full_input, y_full, eval_set=[(X_val_input, y_val)], verbose=False
                )
            else:
                model.fit(X_full_input, y_full)

        # =========================================================================
        # 5. Prediction on Test Set
        # =========================================================================
        print("Generating Level 1 Test Predictions...")
        test_l1_preds = pd.DataFrame()

        for name, model in self.base_models.items():
            X_test_input = self._get_model_input(feats_test, name)
            test_l1_preds[name] = model.predict_proba(X_test_input)[:, 1]

        print("Generating Final Level 2 Predictions...")
        final_preds = self.meta_model.predict_proba(test_l1_preds)[:, 1]

        # =========================================================================
        # 6. Save Submission
        # =========================================================================
        # Load test metadata to get request_ids
        test_df = pd.read_parquet(Config.TEST_METADATA_PATH)
        if debug_sample_size:
            test_df = test_df.iloc[:debug_sample_size]

        submission = pd.DataFrame(
            {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: final_preds}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
