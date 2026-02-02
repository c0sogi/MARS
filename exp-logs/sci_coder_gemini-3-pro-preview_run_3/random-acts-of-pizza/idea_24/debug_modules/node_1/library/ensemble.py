import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import Timer, set_seed, print_header
from library.features import FeaturePipeline
from library.model_definitions import (
    get_lexical_bagger,
    get_community_bagger,
    get_semantic_booster,
    get_semantic_bagger,
    get_manifold_neighbor,
    get_metadata_anchor,
    get_meta_learner,
)


class HexStackingEngine:
    """
    Orchestrates the Hex-View Hybrid-Topology Stacking Ensemble.
    Manages OOF generation, Meta-Learner training, Base Model retraining, and Inference.
    """

    def __init__(self):
        self.feature_pipeline = FeaturePipeline()
        self.base_models = {}
        self.meta_learner = None
        self.model_names = [
            "lexical_rf",
            "community_rf",
            "semantic_xgb",
            "semantic_rf",
            "manifold_knn",
            "metadata_lr",
        ]

        # Ensure working directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def _prepare_model_input(self, features, model_name):
        """
        Constructs the specific feature set for a given model branch.

        Args:
            features (dict): Dictionary of feature arrays (lexical, behavioral, etc.)
            model_name (str): Name of the model to prepare features for.

        Returns:
            array-like: The concatenated feature matrix.
        """
        meta = features["metadata"]

        if model_name == "lexical_rf":
            # Sparse Lexical + Dense Metadata
            return sp.hstack([features["lexical"], meta], format="csr")

        elif model_name == "community_rf":
            # Sparse Behavioral + Dense Metadata
            return sp.hstack([features["behavioral"], meta], format="csr")

        elif model_name == "semantic_xgb":
            # Dense Semantic + Dense Metadata
            return np.hstack([features["semantic"], meta])

        elif model_name == "semantic_rf":
            # Dense Semantic + Dense Metadata
            return np.hstack([features["semantic"], meta])

        elif model_name == "manifold_knn":
            # Dense Manifold (PCA) + Dense Metadata
            return np.hstack([features["manifold"], meta])

        elif model_name == "metadata_lr":
            # Dense Metadata only
            return meta

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _get_model_instance(self, model_name):
        """Factory method to get fresh model instances."""
        if model_name == "lexical_rf":
            return get_lexical_bagger()
        elif model_name == "community_rf":
            return get_community_bagger()
        elif model_name == "semantic_xgb":
            return get_semantic_booster()
        elif model_name == "semantic_rf":
            return get_semantic_bagger()
        elif model_name == "manifold_knn":
            return get_manifold_neighbor()
        elif model_name == "metadata_lr":
            return get_metadata_anchor()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def generate_oof(self, load_cached_data=True):
        """
        Performs 5-Fold Stratified CV to generate Out-of-Fold predictions.

        Returns:
            tuple: (oof_matrix, y_target)
        """
        print_header("Generating Out-of-Fold Predictions")

        # Load Train Data
        df_train = pd.read_parquet(Config.TRAIN_PATH)
        y = df_train[Config.TARGET_COL].values

        # Generate Features
        features = self.feature_pipeline.fit_transform(
            df_train, split_name="train", load_cached_data=load_cached_data
        )

        # Initialize OOF Matrix (N_samples x N_models)
        oof_matrix = np.zeros((len(df_train), len(self.model_names)))

        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            print(f"Processing Fold {fold + 1}/{Config.N_FOLDS}...")

            y_train_fold = y[train_idx]
            # y_val_fold = y[val_idx] # Not strictly needed for training, but useful for debug metrics

            for i, name in enumerate(self.model_names):
                # Prepare data for this specific model
                X_full = self._prepare_model_input(features, name)
                X_train_fold = X_full[train_idx]
                X_val_fold = X_full[val_idx]

                # Instantiate and Train
                model = self._get_model_instance(name)

                # Special handling for XGBoost inside CV (optional early stopping or just fit)
                # To keep CV consistent and simple, we usually just fit.
                # Early stopping is critical for the final model, but in CV we use fixed params
                # or internal validation if strictly necessary.
                # Given the prompt instructions, we apply standard fit here
                # and reserve strict early stopping for the final retraining phase.
                if name == "semantic_xgb":
                    model.fit(X_train_fold, y_train_fold, verbose=False)
                else:
                    model.fit(X_train_fold, y_train_fold)

                # Predict
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_val_fold)[:, 1]
                else:
                    # Fallback if needed, though all classifiers here support predict_proba
                    preds = model.predict(X_val_fold)

                oof_matrix[val_idx, i] = preds

        # Calculate individual OOF Scores
        print("\n--- OOF Scores per Base Model ---")
        for i, name in enumerate(self.model_names):
            auc = roc_auc_score(y, oof_matrix[:, i])
            print(f"{name}: {auc:.16f}")

        return oof_matrix, y

    def train_meta_learner(self, oof_matrix, y):
        """
        Trains the Level 2 Logistic Regression Meta-Learner.
        """
        print_header("Training Meta-Learner")

        self.meta_learner = get_meta_learner()
        self.meta_learner.fit(oof_matrix, y)

        # Check coefficients
        print("Meta-Learner Coefficients:")
        for name, coef in zip(self.model_names, self.meta_learner.coef_[0]):
            print(f"  {name}: {coef:.6f}")

        # OOF Ensemble Score
        oof_preds = self.meta_learner.predict_proba(oof_matrix)[:, 1]
        auc = roc_auc_score(y, oof_preds)
        print(f"Ensemble OOF AUC: {auc:.16f}")

    def retrain_base_models(self, load_cached_data=True):
        """
        Retrains all Level 1 models on the full dataset (Train + Val).
        Implements specific logic for XGBoost (Train + Val Early Stopping).
        """
        print_header("Retraining Base Models")

        # Load Data
        df_train = pd.read_parquet(Config.TRAIN_PATH)
        df_val = pd.read_parquet(Config.VAL_PATH)

        y_train = df_train[Config.TARGET_COL].values
        y_val = df_val[Config.TARGET_COL].values

        # Generate Features
        # Note: fit_transform on train was already done in generate_oof, so it should load from cache
        # We need to transform val using the pipeline fitted on train.
        feats_train = self.feature_pipeline.fit_transform(
            df_train, split_name="train", load_cached_data=load_cached_data
        )
        feats_val = self.feature_pipeline.transform(
            df_val, split_name="val", load_cached_data=load_cached_data
        )

        # Combine for non-early-stopping models
        # We need to concatenate features carefully
        feats_combined = {}
        for key in feats_train.keys():
            if sp.issparse(feats_train[key]):
                feats_combined[key] = sp.vstack(
                    [feats_train[key], feats_val[key]], format="csr"
                )
            else:
                feats_combined[key] = np.vstack([feats_train[key], feats_val[key]])

        y_combined = np.concatenate([y_train, y_val])

        self.base_models = {}

        for name in self.model_names:
            print(f"Retraining {name}...")
            model = self._get_model_instance(name)

            if name == "semantic_xgb":
                # Special Logic: Train on Train, Early Stop on Val
                X_tr = self._prepare_model_input(feats_train, name)
                X_va = self._prepare_model_input(feats_val, name)

                model.fit(
                    X_tr,
                    y_train,
                    eval_set=[(X_va, y_val)],
                    early_stopping_rounds=50,
                    verbose=False,
                )
            else:
                # Standard Logic: Train on Combined Train + Val
                X_comb = self._prepare_model_input(feats_combined, name)
                model.fit(X_comb, y_combined)

            self.base_models[name] = model

    def predict(self, load_cached_data=True):
        """
        Generates predictions for the Test set and saves submission.
        """
        print_header("Generating Final Predictions")

        # Load Test Data
        df_test = pd.read_parquet(Config.TEST_PATH)
        ids = df_test[Config.ID_COL].values

        # Generate Features
        feats_test = self.feature_pipeline.transform(
            df_test, split_name="test", load_cached_data=load_cached_data
        )

        # Level 1 Predictions
        L1_preds = np.zeros((len(df_test), len(self.model_names)))

        for i, name in enumerate(self.model_names):
            model = self.base_models[name]
            X_test = self._prepare_model_input(feats_test, name)

            if hasattr(model, "predict_proba"):
                L1_preds[:, i] = model.predict_proba(X_test)[:, 1]
            else:
                L1_preds[:, i] = model.predict(X_test)

        # Level 2 Prediction
        final_probs = self.meta_learner.predict_proba(L1_preds)[:, 1]

        # Save Submission
        submission = pd.DataFrame({Config.ID_COL: ids, Config.TARGET_COL: final_probs})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission.shape}")

        return submission

    def run(self):
        """
        Executes the full pipeline.
        """
        set_seed(Config.SEED)

        # 1. Generate OOF
        oof_matrix, y_train = self.generate_oof(load_cached_data=True)

        # 2. Train Meta-Learner
        self.train_meta_learner(oof_matrix, y_train)

        # 3. Retrain Base Models
        self.retrain_base_models(load_cached_data=True)

        # 4. Predict
        self.predict(load_cached_data=True)
