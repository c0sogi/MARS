import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.feature_engineering import get_all_features
from library.model_zoo import (
    LexicalBagger,
    BehavioralBagger,
    SemanticBooster,
    SemanticBagger,
    MetadataAnchor,
)
from library.data_loader import load_data


class StackingManager:
    """
    Manages the Robust Pent-View Stacking Ensemble.
    Handles CV for OOF generation, Meta-learner training, and Validation-Guided Retraining.
    """

    def __init__(self):
        # Initialize Level 1 Base Learners
        self.l1_models = {
            "lexical_bagger": LexicalBagger(),
            "behavioral_bagger": BehavioralBagger(),
            "semantic_booster": SemanticBooster(),
            "semantic_bagger": SemanticBagger(),
            "metadata_anchor": MetadataAnchor(),
        }
        # Initialize Level 2 Meta Learner
        self.meta_learner = LogisticRegression(**Config.LR_PARAMS)

    def get_model_input(self, model_name, data_dict, split="train"):
        """
        Retrieves the appropriate feature matrices (Main + Meta) for a given model and split.
        """
        # Always get metadata
        meta_key = f"X_{split}_meta"
        X_meta = data_dict[meta_key]

        # Get main view based on model type
        if model_name == "lexical_bagger":
            main_key = f"X_{split}_lexical"
            X_main = data_dict[main_key]
        elif model_name == "behavioral_bagger":
            main_key = f"X_{split}_behavioral"
            X_main = data_dict[main_key]
        elif model_name in ["semantic_booster", "semantic_bagger"]:
            main_key = f"X_{split}_semantic"
            X_main = data_dict[main_key]
        elif model_name == "metadata_anchor":
            X_main = None  # Model ignores main input
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return X_main, X_meta

    def train_cv(self, data):
        """
        Performs 5-Fold CV on the Training set to generate OOF predictions.
        Trains the Meta-Learner on these OOF predictions.
        """
        print("Starting Level 1 Cross-Validation (OOF Generation)...")

        X_train_meta = data["X_train_meta"]
        y_train = data["y_train"]

        n_samples = y_train.shape[0]
        n_models = len(self.l1_models)

        # Matrix to store OOF predictions
        oof_preds = np.zeros((n_samples, n_models))

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_meta, y_train)):
            # Get fold targets
            y_tr_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            for i, (name, model) in enumerate(self.l1_models.items()):
                # Retrieve full data for this model
                X_main_full, X_meta_full = self.get_model_input(name, data, "train")

                # Split data into fold train/val
                X_meta_tr = X_meta_full[train_idx]
                X_meta_val = X_meta_full[val_idx]

                if X_main_full is not None:
                    # Handle sparse vs dense slicing
                    if sparse.issparse(X_main_full):
                        X_main_tr = X_main_full[train_idx]
                        X_main_val = X_main_full[val_idx]
                    else:
                        X_main_tr = X_main_full[train_idx]
                        X_main_val = X_main_full[val_idx]
                else:
                    X_main_tr = None
                    X_main_val = None

                # Determine eval_set for XGBoost (using the fold validation set)
                eval_set = None
                if name == "semantic_booster":
                    eval_set = (X_main_val, X_meta_val, y_val_fold)

                # Train model on fold
                # Note: We re-fit the existing instance. Sklearn/XGBoost reset on fit.
                model.fit(X_main_tr, X_meta_tr, y_tr_fold, eval_set=eval_set)

                # Predict on fold validation
                preds = model.predict_proba(X_main_val, X_meta_val)
                oof_preds[val_idx, i] = preds

        # Calculate and print OOF AUC
        # We use the mean of L1 predictions just for a sanity check baseline
        avg_oof_auc = roc_auc_score(y_train, oof_preds.mean(axis=1))
        print(f"Level 1 Average OOF AUC: {avg_oof_auc}")

        # Train Meta-Learner
        print("Training Level 2 Meta-Learner on OOF Matrix...")
        self.meta_learner.fit(oof_preds, y_train)

        # Evaluate Meta-Learner on OOF (Consistency Check)
        meta_preds = self.meta_learner.predict_proba(oof_preds)[:, 1]
        meta_auc = roc_auc_score(y_train, meta_preds)
        print(f"Level 2 Stacked OOF AUC: {meta_auc}")

        return meta_auc

    def retrain_and_predict(self, data):
        """
        Retrains Level 1 models on full data and generates Test predictions.
        Implements specific logic:
        - RF/Linear: Train on concatenated Train + Val.
        - XGBoost: Train on Train, use Val for Early Stopping.
        """
        print("Retraining Level 1 Models and Generating Test Predictions...")

        y_train = data["y_train"]
        y_val = data["y_val"]

        # Concatenate targets for models that use full data
        y_full = np.concatenate([y_train, y_val])

        n_test = data["X_test_meta"].shape[0]
        test_preds_l1 = np.zeros((n_test, len(self.l1_models)))

        for i, (name, model) in enumerate(self.l1_models.items()):
            # Get data for all splits
            X_train_main, X_train_meta = self.get_model_input(name, data, "train")
            X_val_main, X_val_meta = self.get_model_input(name, data, "val")
            X_test_main, X_test_meta = self.get_model_input(name, data, "test")

            if name == "semantic_booster":
                # Validation-Guided Retraining for XGBoost
                # Train on Train, Stop on Val
                eval_set = (X_val_main, X_val_meta, y_val)
                model.fit(X_train_main, X_train_meta, y_train, eval_set=eval_set)
            else:
                # Full Retraining for RF/Linear
                # Concatenate Train and Val
                X_full_meta = np.vstack([X_train_meta, X_val_meta])

                if X_train_main is not None:
                    if sparse.issparse(X_train_main):
                        X_full_main = sparse.vstack([X_train_main, X_val_main])
                    else:
                        X_full_main = np.vstack([X_train_main, X_val_main])
                else:
                    X_full_main = None

                model.fit(X_full_main, X_full_meta, y_full)

            # Generate Test Predictions
            test_preds_l1[:, i] = model.predict_proba(X_test_main, X_test_meta)

        # Generate Final Predictions using Meta-Learner
        # Meta-learner is already fitted from train_cv
        print("Generating Final Level 2 Predictions...")
        final_preds = self.meta_learner.predict_proba(test_preds_l1)[:, 1]

        return final_preds

    def save_submission(self, predictions):
        """
        Saves predictions to submission.csv.
        """
        # Load test dataframe to get IDs
        _, _, test_df = load_data()

        submission = pd.DataFrame(
            {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: predictions}
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    def run(self):
        """
        Main execution pipeline.
        """
        # 1. Load Features (Cached or Computed)
        data = get_all_features(load_cached_data=True)

        # 2. Run CV and Train Meta-Learner
        self.train_cv(data)

        # 3. Retrain L1 Models and Predict on Test
        final_preds = self.retrain_and_predict(data)

        # 4. Save Submission
        self.save_submission(final_preds)


def run_stacking():
    """
    Entry point function.
    """
    manager = StackingManager()
    manager.run()
