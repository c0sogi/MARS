import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import Config
from library.utils import log, set_seed
from library.model_definitions import get_base_models, get_meta_model


class StackingEnsemble:
    """
    Implements the Regularized Pent-View Stacking Ensemble.
    Manages Level 1 Base Learners and Level 2 Meta-Learner.
    """

    def __init__(self):
        set_seed()
        self.base_models = get_base_models()
        self.meta_model = get_meta_model()
        self.n_folds = Config.N_FOLDS

    def _get_feature_view(self, model_name, feature_dict):
        """
        Maps a model name to its corresponding feature view from the dictionary.
        """
        if model_name == "LexicalBagger":
            return feature_dict["lexical"]
        elif model_name == "CommunityBagger":
            return feature_dict["behavioral"]
        elif model_name in ["SemanticBooster", "SemanticBagger"]:
            return feature_dict["semantic"]
        elif model_name == "MetadataAnchor":
            return feature_dict["contextual"]
        else:
            raise ValueError(f"Mapping not defined for model: {model_name}")

    def _concat_features(self, X1, X2):
        """
        Concatenates two feature matrices (sparse or dense).
        """
        if sp.issparse(X1) or sp.issparse(X2):
            return sp.vstack([X1, X2])
        else:
            return np.vstack([X1, X2])

    def fit_oof(self, train_features, y_train):
        """
        Performs Stratified K-Fold Cross-Validation to generate Out-of-Fold (OOF) predictions.
        Trains the Meta-Learner on these OOF predictions.

        Args:
            train_features (dict): Dictionary of training feature views.
            y_train (pd.Series): Training targets.

        Returns:
            pd.DataFrame: OOF predictions for each base model.
        """
        log("Starting OOF generation (Level 1)...")

        # Initialize OOF container
        model_names = list(self.base_models.keys())
        n_samples = len(y_train)
        oof_preds = pd.DataFrame(index=np.arange(n_samples), columns=model_names)
        oof_preds = oof_preds.fillna(0.0)

        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=Config.RANDOM_SEED
        )

        # Iterate Folds
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_train)
        ):
            log(f"  Processing Fold {fold + 1}/{self.n_folds}...")

            y_tr_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]

            for name, model in self.base_models.items():
                # Get specific view
                X_view = self._get_feature_view(name, train_features)

                # Slice data
                if sp.issparse(X_view):
                    X_tr_fold = X_view[train_idx]
                    X_val_fold = X_view[val_idx]
                else:
                    X_tr_fold = X_view[train_idx]
                    X_val_fold = X_view[val_idx]

                # Clone model to ensure fresh start
                clf = clone(model)

                # Fit
                if name == "SemanticBooster":
                    # Use fold validation for early stopping in XGBoost
                    fit_params = Config.SEMANTIC_XGB_FIT_PARAMS.copy()
                    fit_params["eval_set"] = [(X_val_fold, y_val_fold)]
                    clf.fit(X_tr_fold, y_tr_fold, **fit_params)
                else:
                    clf.fit(X_tr_fold, y_tr_fold)

                # Predict
                probs = clf.predict_proba(X_val_fold)[:, 1]
                oof_preds.loc[val_idx, name] = probs

        # Evaluate Base Models
        log("Level 1 OOF Performance (AUC):")
        for name in model_names:
            auc = roc_auc_score(y_train, oof_preds[name])
            print(f"  {name}: {auc}")

        # Train Meta-Learner
        log("Training Meta-Learner (Level 2)...")
        self.meta_model.fit(oof_preds, y_train)

        # Meta Performance
        meta_oof_probs = self.meta_model.predict_proba(oof_preds)[:, 1]
        meta_auc = roc_auc_score(y_train, meta_oof_probs)
        print(f"  Meta-Learner OOF AUC: {meta_auc}")

        return oof_preds

    def retrain_and_predict(
        self, train_features, y_train, val_features, y_val, test_features, test_ids
    ):
        """
        Retrains base models on full data (using validation-guided protocol) and generates test predictions.

        Args:
            train_features (dict): Train views.
            y_train (pd.Series): Train targets.
            val_features (dict): Validation views.
            y_val (pd.Series): Validation targets.
            test_features (dict): Test views.
            test_ids (pd.Series): Test IDs.
        """
        log("Starting Final Retraining and Prediction...")

        model_names = list(self.base_models.keys())
        n_test = test_ids.shape[0]

        # Matrix to hold Level 1 Test Predictions
        test_meta_features = pd.DataFrame(index=np.arange(n_test), columns=model_names)

        for name, model in self.base_models.items():
            log(f"  Retraining {name}...")

            # Get Views
            X_train = self._get_feature_view(name, train_features)
            X_val = self._get_feature_view(name, val_features)
            X_test = self._get_feature_view(name, test_features)

            # Clone model
            clf = clone(model)

            # Logic Branch
            if name == "SemanticBooster":  # XGBoost
                # Train on Train, Eval on Val (Validation-Guided)
                fit_params = Config.SEMANTIC_XGB_FIT_PARAMS.copy()
                fit_params["eval_set"] = [(X_val, y_val)]

                clf.fit(X_train, y_train, **fit_params)

            else:  # RF or Linear
                # Train on Train + Val (Full Training Set)
                X_full = self._concat_features(X_train, X_val)
                y_full = pd.concat([y_train, y_val], axis=0)

                clf.fit(X_full, y_full)

            # Predict on Test
            test_probs = clf.predict_proba(X_test)[:, 1]
            test_meta_features[name] = test_probs

        # Meta-Learner Prediction
        log("Generating Final Predictions via Meta-Learner...")
        final_probs = self.meta_model.predict_proba(test_meta_features)[:, 1]

        # Create Submission
        submission = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": final_probs}
        )

        # Save
        save_path = Config.SUBMISSION_FILE_PATH
        log(f"Saving submission to {save_path}...")
        submission.to_csv(save_path, index=False)

        return submission
