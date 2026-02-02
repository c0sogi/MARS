import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library import config
from library import utils
from library.models import ModelFactory


class StackingEnsemble:
    """
    Manages the training, validation, and prediction of the Stacking Ensemble.
    Includes 6 Base Learners (Level 1) and 1 Meta-Learner (Level 2).
    """

    def __init__(self, train_feats, val_feats, test_feats):
        """
        Args:
            train_feats (dict): Dictionary of training features (numpy arrays).
            val_feats (dict): Dictionary of validation features (numpy arrays).
            test_feats (dict): Dictionary of test features (numpy arrays).
        """
        self.train_feats = train_feats
        self.val_feats = val_feats
        self.test_feats = test_feats

        # Define Base Learners and their corresponding feature keys
        # Format: (Model Name, Factory Method, Feature Key)
        self.learners_config = [
            ("interaction_bagger", ModelFactory.get_interaction_bagger, "holistic"),
            ("lexical_bagger", ModelFactory.get_lexical_bagger, "lexical"),
            ("community_bagger", ModelFactory.get_community_bagger, "community"),
            ("semantic_booster", ModelFactory.get_semantic_booster, "semantic"),
            ("semantic_bagger", ModelFactory.get_semantic_bagger, "semantic"),
            ("metadata_anchor", ModelFactory.get_metadata_anchor, "metadata"),
        ]

        self.base_models = {}
        self.final_base_models = {}
        self.meta_learner = ModelFactory.get_meta_learner()

        # Placeholders for OOF predictions
        self.oof_preds = None
        self.y_train_oof = None

    def _get_X_y(self, feats_dict, key):
        """Helper to extract X and y from the feature dictionary."""
        return feats_dict[key], feats_dict["y"]

    def train_cv(self, n_folds=config.N_FOLDS):
        """
        Performs Stratified K-Fold CV to generate OOF predictions and train the Meta-Learner.
        """
        utils.print_header("Starting Level 1 Cross-Validation")

        # We use the target from the training set to define folds
        y_train_full = self.train_feats["y"]
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE
        )

        # Initialize OOF matrix: (n_samples, n_models)
        n_samples = len(y_train_full)
        n_models = len(self.learners_config)
        self.oof_preds = np.zeros((n_samples, n_models))
        self.y_train_oof = y_train_full  # Target aligns with OOF indices

        # Iterate over folds
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_train_full)
        ):
            print(f"Processing Fold {fold_idx + 1}/{n_folds}...")

            for model_idx, (name, factory_func, key) in enumerate(self.learners_config):
                # Retrieve full data for this view
                X_full = self.train_feats[key]
                y_full = y_train_full

                # Split for this fold
                X_fold_train, X_fold_val = X_full[train_idx], X_full[val_idx]
                y_fold_train, y_fold_val = y_full[train_idx], y_full[val_idx]

                # Instantiate model
                model = factory_func()

                # Train
                # Special handling for XGBoost (Semantic Booster) to use early stopping within CV
                if name == "semantic_booster":
                    model.fit(
                        X_fold_train,
                        y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
                        verbose=False,
                    )
                else:
                    model.fit(X_fold_train, y_fold_train)

                # Predict (Probabilities for class 1)
                y_pred_val = model.predict_proba(X_fold_val)[:, 1]

                # Store in OOF matrix
                self.oof_preds[val_idx, model_idx] = y_pred_val

        # Calculate CV Score
        overall_auc = roc_auc_score(self.y_train_oof, self.oof_preds.mean(axis=1))
        utils.print_metric("Level 1 CV Average Ensemble AUC", overall_auc)

        # Train Meta-Learner on OOF predictions
        utils.print_header("Training Level 2 Meta-Learner")
        self.meta_learner.fit(self.oof_preds, self.y_train_oof)

        # Check Meta-Learner Performance on OOF (In-sample for meta-learner, but OOF for base)
        meta_oof_preds = self.meta_learner.predict_proba(self.oof_preds)[:, 1]
        meta_auc = roc_auc_score(self.y_train_oof, meta_oof_preds)
        utils.print_metric("Level 2 Meta-Learner OOF AUC", meta_auc)

    def train_final_models(self):
        """
        Retrains all base learners for the final submission.
        - RF/Linear: Retrain on Train + Val.
        - XGBoost: Retrain on Train, use Val for early stopping.
        """
        utils.print_header("Retraining Final Base Learners")

        for name, factory_func, key in self.learners_config:
            print(f"Retraining {name}...")
            model = factory_func()

            if name == "semantic_booster":
                # XGBoost: Train on Train, Validate on Val (Early Stopping)
                X_train = self.train_feats[key]
                y_train = self.train_feats["y"]
                X_val = self.val_feats[key]
                y_val = self.val_feats["y"]

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
            else:
                # RF / Linear: Train on Train + Val combined
                X_train = self.train_feats[key]
                X_val = self.val_feats[key]
                y_train = self.train_feats["y"]
                y_val = self.val_feats["y"]

                X_full = np.vstack([X_train, X_val])
                y_full = np.concatenate([y_train, y_val])

                model.fit(X_full, y_full)

            self.final_base_models[name] = model

    def predict(self):
        """
        Generates predictions for the test set using the retrained models and meta-learner.
        Saves the submission file.
        """
        utils.print_header("Generating Final Predictions")

        n_test = self.test_feats["metadata"].shape[0]
        n_models = len(self.learners_config)

        # Matrix to hold Level 1 Test Predictions
        level1_test_preds = np.zeros((n_test, n_models))

        for i, (name, _, key) in enumerate(self.learners_config):
            print(f"Predicting with {name}...")
            model = self.final_base_models[name]
            X_test = self.test_feats[key]

            # Predict probabilities
            preds = model.predict_proba(X_test)[:, 1]
            level1_test_preds[:, i] = preds

        # Level 2 Prediction
        print("Predicting with Meta-Learner...")
        final_probs = self.meta_learner.predict_proba(level1_test_preds)[:, 1]

        # Save Submission
        self._save_submission(final_probs)

    def _save_submission(self, probabilities):
        """
        Saves the probabilities to a CSV file in the required format.
        """
        # Load test metadata to get request_ids
        # We can assume the order matches because we processed test.parquet sequentially
        test_df = utils.load_parquet(config.TEST_PATH)

        submission = pd.DataFrame(
            {config.ID_COL: test_df[config.ID_COL], config.TARGET_COL: probabilities}
        )

        utils.ensure_dir(config.SUBMISSION_PATH)
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission.shape}")
        print(f"First 5 predictions:\n{submission.head()}")
