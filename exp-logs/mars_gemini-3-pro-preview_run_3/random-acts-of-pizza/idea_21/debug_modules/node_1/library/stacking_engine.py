import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone, is_classifier
from xgboost import XGBClassifier

from library import config, utils
from library.model_definitions import ModelFactory


class StackingTrainer:
    """
    Manages the training, validation, and prediction of the Hex-View Stacking Ensemble.
    Implements the Validation-Guided Retraining Protocol.
    """

    def __init__(self, feature_dict):
        """
        Args:
            feature_dict (dict): Dictionary containing 'train', 'val', 'test' feature views.
        """
        self.logger = utils.get_logger("StackingTrainer")
        self.features = feature_dict
        self.models = ModelFactory.get_level1_models()
        self.meta_learner = ModelFactory.get_meta_learner()
        self.oof_preds = None
        self.test_preds = None

    def run(self):
        """
        Executes the full stacking pipeline:
        1. Generate OOF predictions via 5-Fold CV on Training set.
        2. Train Meta-Learner on OOF predictions.
        3. Retrain Base Learners (using Val set strategy).
        4. Generate Final Test Predictions.
        5. Save Submission.
        """
        with utils.Timer("Full Stacking Pipeline"):
            # 1. Level 1: OOF Generation
            self.logger.info("Starting Level 1: OOF Generation (5-Fold CV)...")
            X_meta_train, y_train = self._generate_oof()

            # 2. Level 2: Meta-Learner Training
            self.logger.info("Starting Level 2: Meta-Learner Training...")
            self._train_level2(X_meta_train, y_train)

            # 3. Final Retraining & Prediction
            self.logger.info("Starting Final Retraining & Test Prediction...")
            final_probs = self._retrain_and_predict_test()

            # 4. Save Submission
            # We need request_ids for the submission.
            # We load the test dataframe just to get IDs.
            df_test = pd.read_parquet(config.TEST_PATH)
            test_ids = df_test[config.ID_COL].values

            utils.save_submission(test_ids, final_probs)

    def _generate_oof(self):
        """
        Performs 5-Fold Stratified CV on the Train split to generate OOF predictions.
        Returns:
            X_meta_train (np.ndarray): (N_train, N_models) OOF probability matrix.
            y_train (np.ndarray): Training targets.
        """
        y_train = self.features["train"]["y"]
        n_samples = len(y_train)
        n_models = len(self.models)

        # Initialize OOF matrix
        oof_matrix = np.zeros((n_samples, n_models))

        # Setup CV
        skf = StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
        )

        # Pre-prepare full training features for each model to avoid re-assembling in loop
        # But we need to slice them, so we keep them as full matrices first.
        model_features_train = {}
        for name in self.models.keys():
            model_features_train[name] = ModelFactory.prepare_features(
                name, self.features, "train"
            )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_train)
        ):
            self.logger.info(f"Processing Fold {fold + 1}/{config.N_FOLDS}...")

            for i, (name, model) in enumerate(self.models.items()):
                # Clone model to ensure fresh training
                clf = clone(model)

                # Get specific feature view
                X_full = model_features_train[name]

                # Slice data
                # Handle sparse vs dense slicing
                if sp.issparse(X_full):
                    X_tr_fold = X_full[train_idx]
                    X_val_fold = X_full[val_idx]
                else:
                    X_tr_fold = X_full[train_idx]
                    X_val_fold = X_full[val_idx]

                y_tr_fold = y_train[train_idx]
                # y_val_fold = y_train[val_idx] # Not needed for fit, only for metric if we wanted

                # Fit
                # Note: For OOF generation, we don't strictly use early stopping with the fold-val set
                # to keep it simple and consistent with standard stacking, unless specified.
                # The prompt specifies early stopping for the *Final Retraining*.
                # For CV, we stick to standard fit to avoid leakage within the fold structure if not careful.
                # However, XGBoost usually benefits from it. Given the strict instructions are for "Retraining",
                # we will use standard fit here for all models.
                if isinstance(clf, XGBClassifier):
                    # For CV, we can use the fold validation set for early stopping
                    clf.fit(
                        X_tr_fold,
                        y_tr_fold,
                        eval_set=[(X_val_fold, y_train[val_idx])],
                        verbose=False,
                    )
                else:
                    clf.fit(X_tr_fold, y_tr_fold)

                # Predict (Probabilities of class 1)
                preds = clf.predict_proba(X_val_fold)[:, 1]
                oof_matrix[val_idx, i] = preds

        # Calculate and print OOF AUC for each base model
        self.logger.info("--- Level 1 OOF Performance ---")
        for i, name in enumerate(self.models.keys()):
            auc = utils.print_metrics(
                y_train, oof_matrix[:, i], split_name=f"OOF_{name}"
            )

        return oof_matrix, y_train

    def _train_level2(self, X_meta_train, y_train):
        """
        Trains the Meta-Learner on the OOF predictions.
        """
        self.meta_learner.fit(X_meta_train, y_train)

        # Check coefficients to see which models are trusted
        coefs = self.meta_learner.coef_[0]
        self.logger.info("--- Level 2 Meta-Learner Coefficients ---")
        for i, name in enumerate(self.models.keys()):
            self.logger.info(f"{name}: {coefs[i]:.4f}")

    def _retrain_and_predict_test(self):
        """
        Retrains base models on full data (Train + Val) or (Train w/ Val ES),
        then generates Test predictions.

        Returns:
            final_probs (np.ndarray): Final predicted probabilities for Test set.
        """
        # Prepare Test Features for all models
        model_features_test = {}
        for name in self.models.keys():
            model_features_test[name] = ModelFactory.prepare_features(
                name, self.features, "test"
            )

        n_test_samples = self.features["test"]["metadata"].shape[0]
        n_models = len(self.models)
        test_meta_features = np.zeros((n_test_samples, n_models))

        # Load Targets
        y_train = self.features["train"]["y"]
        y_val = self.features["val"]["y"]

        for i, (name, model) in enumerate(self.models.items()):
            self.logger.info(f"Retraining {name}...")
            clf = clone(model)

            # Prepare Features
            X_train = ModelFactory.prepare_features(name, self.features, "train")
            X_val = ModelFactory.prepare_features(name, self.features, "val")
            X_test = model_features_test[name]

            # Retraining Logic based on Model Type
            if isinstance(clf, XGBClassifier):
                # XGBoost: Train on Train, Eval on Val (Early Stopping)
                # As per instruction: "Retrain on the Full Training Set but explicitly pass the Validation Set... as eval_set"
                # We interpret this as using the 'train' split for learning and 'val' split for stopping.
                clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                # Log validation score
                val_preds = clf.predict_proba(X_val)[:, 1]
                utils.print_metrics(y_val, val_preds, split_name=f"Final_Val_{name}")

            else:
                # RF / Linear: Train on Train + Val
                if sp.issparse(X_train):
                    X_full = sp.vstack([X_train, X_val], format="csr")
                else:
                    X_full = np.vstack([X_train, X_val])

                y_full = np.concatenate([y_train, y_val])

                clf.fit(X_full, y_full)

            # Predict on Test
            test_preds = clf.predict_proba(X_test)[:, 1]
            test_meta_features[:, i] = test_preds

        # Meta-Learner Prediction
        self.logger.info("Generating Final Predictions via Meta-Learner...")
        final_probs = self.meta_learner.predict_proba(test_meta_features)[:, 1]

        return final_probs
