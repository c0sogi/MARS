import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import Timer, set_seed
from library.model_zoo import (
    LexicalBagger,
    CommunityBagger,
    SemanticBooster,
    SemanticBagger,
    MetadataAnchor,
    StackingMetaLearner,
)


def slice_features(X_dict, indices):
    """
    Slices a dictionary of features (sparse or dense) by row indices.
    """
    sliced = {}
    for key, data in X_dict.items():
        # Works for both numpy arrays and scipy sparse matrices
        sliced[key] = data[indices]
    return sliced


def concat_features(X_train_dict, X_val_dict):
    """
    Concatenates training and validation feature dictionaries vertically.
    Handles both sparse and dense matrices.
    """
    concatenated = {}
    for key in X_train_dict.keys():
        train_data = X_train_dict[key]
        val_data = X_val_dict[key]

        if sp.issparse(train_data):
            concatenated[key] = sp.vstack([train_data, val_data], format="csr")
        else:
            concatenated[key] = np.concatenate([train_data, val_data], axis=0)
    return concatenated


class CrossValidationEngine:
    """
    Manages the 5-Fold Stratified CV to generate OOF predictions for the Meta-Learner.
    """

    def __init__(self, n_folds=Config.N_FOLDS, seed=Config.SEED):
        self.n_folds = n_folds
        self.seed = seed
        self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    def _init_base_models(self):
        """Returns a fresh list of the 5 base learners."""
        return [
            LexicalBagger(),
            CommunityBagger(),
            SemanticBooster(),
            SemanticBagger(),
            MetadataAnchor(),
        ]

    def _get_model_input(self, model_name, X_dict):
        """Maps model names to their specific feature view."""
        if model_name == "LexicalBagger":
            return X_dict["lexical"]
        elif model_name == "CommunityBagger":
            return X_dict["behavioral"]
        elif model_name == "SemanticBooster":
            return X_dict["semantic"]
        elif model_name == "SemanticBagger":
            return X_dict["semantic"]
        elif model_name == "MetadataAnchor":
            return X_dict["metadata"]
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def run_cv(self, X_train_full, y_train_full):
        """
        Performs CV.
        Returns:
            oof_preds: (N_samples, 5) matrix of probabilities.
            y_train_full: The target array (aligned).
        """
        set_seed(self.seed)

        n_samples = len(y_train_full)
        n_models = 5  # We have 5 base learners
        oof_preds = np.zeros((n_samples, n_models))

        # Track metrics
        fold_aucs = []

        print(f"Starting {self.n_folds}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(
            self.skf.split(np.zeros(n_samples), y_train_full)
        ):
            with Timer(f"Fold {fold + 1}"):
                # Slice data
                X_tr_fold = slice_features(X_train_full, train_idx)
                y_tr_fold = y_train_full[train_idx]

                X_val_fold = slice_features(X_train_full, val_idx)
                y_val_fold = y_train_full[val_idx]

                # Instantiate fresh models
                models = self._init_base_models()

                # Train and Predict for each model
                for i, model in enumerate(models):
                    X_tr_input = self._get_model_input(model.name, X_tr_fold)
                    X_val_input = self._get_model_input(model.name, X_val_fold)

                    # Fit
                    if model.name == "SemanticBooster":
                        # XGBoost within CV uses internal validation split if desired,
                        # or just fits. Here we fit on fold train.
                        # We don't use early stopping inside CV to keep it simple and consistent
                        # with standard stacking, or we could split X_tr_fold again.
                        # Given the design, we fit on the fold training data.
                        model.fit(X_tr_input, y_tr_fold)
                    else:
                        model.fit(X_tr_input, y_tr_fold)

                    # Predict OOF
                    preds = model.predict_proba(X_val_input)
                    oof_preds[val_idx, i] = preds

                # Evaluate Fold Ensemble (Simple Average for monitoring)
                fold_avg_pred = np.mean(oof_preds[val_idx], axis=1)
                fold_auc = roc_auc_score(y_val_fold, fold_avg_pred)
                fold_aucs.append(fold_auc)
                print(f"  Fold {fold + 1} Average Ensemble AUC: {fold_auc}")

        print(f"CV Complete. Mean Fold AUC (Simple Avg): {np.mean(fold_aucs)}")
        return oof_preds, y_train_full


class ValidationGuidedRetrainer:
    """
    Retrains models for final inference using the specific protocols:
    - RF/Linear: Train on combined Train + Val.
    - XGB: Train on Train, Early Stop on Val.
    - Meta: Train on OOFs.
    """

    def __init__(self):
        pass

    def _get_model_input(self, model_name, X_dict):
        if model_name == "LexicalBagger":
            return X_dict["lexical"]
        elif model_name == "CommunityBagger":
            return X_dict["behavioral"]
        elif model_name == "SemanticBooster":
            return X_dict["semantic"]
        elif model_name == "SemanticBagger":
            return X_dict["semantic"]
        elif model_name == "MetadataAnchor":
            return X_dict["metadata"]
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def train_final_models(self, X_train, y_train, X_val, y_val, oof_X, oof_y):
        """
        Returns a dictionary of trained models (base + meta).
        """
        set_seed(Config.SEED)
        trained_models = {}

        # 1. Train Meta-Learner
        print("\nTraining Level 2 Meta-Learner...")
        meta_learner = StackingMetaLearner()
        meta_learner.fit(oof_X, oof_y)
        trained_models["meta"] = meta_learner

        # Check Meta-Learner Performance on OOF
        oof_meta_pred = meta_learner.predict_proba(oof_X)
        oof_score = roc_auc_score(oof_y, oof_meta_pred)
        print(f"Meta-Learner OOF AUC: {oof_score}")

        # 2. Train Base Learners
        print("\nRetraining Level 1 Base Learners...")

        # Prepare Combined Data for RF/Linear
        X_combined = concat_features(X_train, X_val)
        y_combined = np.concatenate([y_train, y_val])

        base_models = [
            LexicalBagger(),
            CommunityBagger(),
            SemanticBooster(),
            SemanticBagger(),
            MetadataAnchor(),
        ]

        for model in base_models:
            print(f"Retraining {model.name}...")

            if model.name == "SemanticBooster":
                # Protocol: Train on Train, Early Stop on Val
                X_tr_in = self._get_model_input(model.name, X_train)
                X_val_in = self._get_model_input(model.name, X_val)
                model.fit(X_tr_in, y_train, X_val=X_val_in, y_val=y_val)
            else:
                # Protocol: Train on Full History (Train + Val)
                X_comb_in = self._get_model_input(model.name, X_combined)
                model.fit(X_comb_in, y_combined)

            trained_models[model.name] = model

        return trained_models


def generate_submission(trained_models, X_test, test_df):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("\nGenerating Test Predictions...")

    # 1. Generate Level 1 Predictions
    n_test = len(test_df)
    base_preds = np.zeros((n_test, 5))

    # Order must match the OOF generation order in CrossValidationEngine
    model_order = [
        "LexicalBagger",
        "CommunityBagger",
        "SemanticBooster",
        "SemanticBagger",
        "MetadataAnchor",
    ]

    # Helper to map input
    def get_input(name, X):
        if name == "LexicalBagger":
            return X["lexical"]
        if name == "CommunityBagger":
            return X["behavioral"]
        if name == "SemanticBooster":
            return X["semantic"]
        if name == "SemanticBagger":
            return X["semantic"]
        if name == "MetadataAnchor":
            return X["metadata"]
        return None

    for i, name in enumerate(model_order):
        model = trained_models[name]
        X_in = get_input(name, X_test)
        preds = model.predict_proba(X_in)
        base_preds[:, i] = preds

    # 2. Generate Level 2 Predictions
    meta_learner = trained_models["meta"]
    final_probs = meta_learner.predict_proba(base_preds)

    # 3. Create Submission DataFrame
    submission = pd.DataFrame(
        {"request_id": test_df[Config.ID_COL], Config.TARGET_COL: final_probs}
    )

    # 4. Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")
    print(f"Head:\n{submission.head()}")
