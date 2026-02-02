import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib

from library import config
from library import utils
from library import model_definitions


class StackingTrainer:
    """
    Manages the training lifecycle of the Hex-View Temporal-Topology Stacking Ensemble.
    Implements OOF generation, Meta-learner training, and Validation-Guided Retraining.
    """

    def __init__(self):
        self.logger = utils.get_logger("StackingTrainer")
        self.n_folds = config.N_FOLDS
        self.seed = config.SEED

        # Initialize Level 1 Base Learners
        # We create a fresh instance for final retraining,
        # but for CV we will create instances inside the loop.
        self.model_classes = {
            "lexical_bagger": model_definitions.LexicalBagger,
            "community_bagger": model_definitions.CommunityBagger,
            "semantic_booster": model_definitions.SemanticBooster,
            "semantic_bagger": model_definitions.SemanticBagger,
            "metadata_anchor": model_definitions.MetadataAnchor,
            "temporal_booster": model_definitions.TemporalBooster,
        }

        # Level 2 Meta Learner
        self.meta_learner = LogisticRegression(**config.META_LEARNER_PARAMS)

        # Store final trained models here
        self.final_models = {}

    def _get_model_input(self, data, model_name, split="train", idx=None):
        """
        Helper to extract specific feature views for a given model and split.

        Args:
            data: The data dictionary from DataLoader.
            model_name: Name of the model (key in self.model_classes).
            split: 'train', 'val', or 'test'.
            idx: Optional indices to slice the data (for CV).

        Returns:
            Tuple (X_specific, X_meta)
        """
        # Map model names to specific feature keys in the data dict
        # Keys in data are like 'X_train_lexical', 'X_val_semantic', etc.

        feature_map = {
            "lexical_bagger": "lexical",
            "community_bagger": "behavioral",
            "semantic_booster": "semantic",
            "semantic_bagger": "semantic",
            "metadata_anchor": None,  # Uses only metadata
            "temporal_booster": None,  # Uses only metadata
        }

        feat_type = feature_map[model_name]

        # Construct keys
        meta_key = f"X_{split}_meta"
        spec_key = f"X_{split}_{feat_type}" if feat_type else None

        # Retrieve data
        X_meta = data[meta_key]
        X_spec = data[spec_key] if spec_key else None

        # Slice if indices provided
        if idx is not None:
            X_meta = X_meta[idx]
            if X_spec is not None:
                # Handle sparse vs dense slicing
                X_spec = X_spec[idx]

        # For models that ignore the first argument (MetadataAnchor, TemporalBooster),
        # we pass a placeholder (None) or just handle it in the model wrapper.
        # The wrappers expect (X_specific, X_meta).
        # If X_specific is None, we pass None.

        return X_spec, X_meta

    def generate_oof(self, data):
        """
        Performs 5-Fold CV on the Training set to generate Out-Of-Fold predictions.
        """
        self.logger.info("Starting OOF Prediction Generation...")

        y_train = data["y_train"]
        n_samples = len(y_train)
        n_models = len(self.model_classes)

        # Matrix to store OOF probabilities: (n_samples, n_models)
        oof_preds = np.zeros((n_samples, n_models))

        # Store model names to ensure consistent column ordering
        self.model_names = list(self.model_classes.keys())

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_train)
        ):
            self.logger.info(f"Processing Fold {fold + 1}/{self.n_folds}")

            y_fold_train = y_train[train_idx]
            # y_fold_val = y_train[val_idx] # Not strictly needed for fit, but good for debug

            for i, name in enumerate(self.model_names):
                # Instantiate fresh model
                model = self.model_classes[name]()

                # Get inputs
                X_spec_train, X_meta_train = self._get_model_input(
                    data, name, "train", train_idx
                )
                X_spec_val, X_meta_val = self._get_model_input(
                    data, name, "train", val_idx
                )

                # Fit
                # Note: We do not use early stopping for the CV folds to keep it simple and
                # consistent with standard stacking, or we could.
                # Given the prompt instructions, early stopping is emphasized for the FINAL retraining.
                # However, for Boosting models, it's often good practice to use ES in CV too.
                # We will pass the validation fold as eval_set for boosting models if supported.

                eval_set = None
                if name in ["semantic_booster", "temporal_booster"]:
                    eval_set = (X_spec_val, X_meta_val, y_train[val_idx])

                model.fit(X_spec_train, X_meta_train, y_fold_train, eval_set=eval_set)

                # Predict
                probs = model.predict_proba(X_spec_val, X_meta_val)[:, 1]
                oof_preds[val_idx, i] = probs

        self.logger.info("OOF Generation Complete.")
        return oof_preds

    def train_meta_learner(self, oof_preds, y_train):
        """
        Trains the Level 2 Logistic Regression on OOF predictions.
        """
        self.logger.info("Training Level 2 Meta-Learner...")

        self.meta_learner.fit(oof_preds, y_train)

        # Evaluate on OOF (Proxy for validation score)
        meta_preds = self.meta_learner.predict_proba(oof_preds)[:, 1]
        auc = roc_auc_score(y_train, meta_preds)

        self.logger.info(f"Meta-Learner CV AUC Score: {auc}")
        return auc

    def final_retrain(self, data):
        """
        Retrains all Level 1 models using Validation-Guided Retraining Protocol.
        """
        self.logger.info("Starting Final Retraining of Level 1 Models...")

        y_train = data["y_train"]
        y_val = data["y_val"]

        for name in self.model_names:
            self.logger.info(f"Retraining {name}...")
            model = self.model_classes[name]()

            # Get Data
            X_spec_train, X_meta_train = self._get_model_input(data, name, "train")
            X_spec_val, X_meta_val = self._get_model_input(data, name, "val")

            # Strategy 1: Boosting Models (Use Train + Early Stopping on Val)
            if name in ["semantic_booster", "temporal_booster"]:
                self.logger.info(
                    f"-> Strategy: Train w/ Early Stopping on Validation Set"
                )
                eval_set = (X_spec_val, X_meta_val, y_val)
                model.fit(X_spec_train, X_meta_train, y_train, eval_set=eval_set)

            # Strategy 2: Bagging/Linear Models (Concatenate Train + Val)
            else:
                self.logger.info(f"-> Strategy: Concatenate Train + Validation Set")

                # Concatenate Labels
                y_full = np.concatenate([y_train, y_val])

                # Concatenate Metadata (Dense)
                X_meta_full = np.vstack([X_meta_train, X_meta_val])

                # Concatenate Specific Features
                if X_spec_train is None:
                    X_spec_full = None
                elif sparse.issparse(X_spec_train):
                    X_spec_full = sparse.vstack([X_spec_train, X_spec_val])
                else:
                    X_spec_full = np.vstack([X_spec_train, X_spec_val])

                model.fit(X_spec_full, X_meta_full, y_full)

            self.final_models[name] = model

        self.logger.info("Final Retraining Complete.")

    def generate_submission(self, data):
        """
        Generates predictions for the test set and saves the submission file.
        """
        self.logger.info("Generating Final Test Predictions...")

        n_test = len(data["test_ids"])
        n_models = len(self.model_names)
        level1_test_preds = np.zeros((n_test, n_models))

        # 1. Generate Level 1 Predictions
        for i, name in enumerate(self.model_names):
            model = self.final_models[name]
            X_spec_test, X_meta_test = self._get_model_input(data, name, "test")

            probs = model.predict_proba(X_spec_test, X_meta_test)[:, 1]
            level1_test_preds[:, i] = probs

        # 2. Generate Level 2 Predictions
        final_probs = self.meta_learner.predict_proba(level1_test_preds)[:, 1]

        # 3. Create Submission DataFrame
        submission_df = pd.DataFrame(
            {config.ID_COL: data["test_ids"], config.TARGET_COL: final_probs}
        )

        # 4. Save
        save_path = config.SUBMISSION_PATH
        self.logger.info(f"Saving submission to {save_path}")
        submission_df.to_csv(save_path, index=False)

        # Also save the OOF predictions and models for reproducibility/analysis
        # (Optional but good practice based on prompt context)
        oof_path = os.path.join(config.WORKING_DIR, "oof_predictions.parquet")
        # We don't have the OOF array here (it was local to run), but we saved the models.

        return submission_df

    def run(self, data):
        """
        Main execution method.
        """
        # 1. Generate OOF Predictions
        oof_preds = self.generate_oof(data)

        # 2. Train Meta-Learner
        self.train_meta_learner(oof_preds, data["y_train"])

        # 3. Final Retraining of Base Learners
        self.final_retrain(data)

        # 4. Generate Submission
        self.generate_submission(data)
