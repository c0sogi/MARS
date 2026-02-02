import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from library import config
from library import utils
from library import model_factory
from library import data_processing


class EnsembleTrainer:
    def __init__(self):
        self.base_models = model_factory.get_base_models()
        self.meta_learner = model_factory.get_meta_learner()
        self.model_names = list(self.base_models.keys())

    def _prepare_input(self, data_dict, model_name):
        """
        Prepares the specific feature set for a given model based on the architecture.
        Handles concatenation of sparse and dense features.
        """
        # Ensure metadata is 2D
        meta = data_dict["metadata"]
        if len(meta.shape) == 1:
            meta = meta.reshape(-1, 1)

        if model_name == "lexical_bagger":
            # Enhanced Lexical Bagger: Sparse Text + Metadata
            # Returns Sparse CSR
            return sparse.hstack([data_dict["lexical"], meta]).tocsr()

        elif model_name == "community_bagger":
            # Constrained Community Bagger: Sparse Community + Metadata
            # Returns Sparse CSR
            return sparse.hstack([data_dict["community"], meta]).tocsr()

        elif model_name in ["semantic_booster", "semantic_bagger"]:
            # Semantic Booster/Bagger: Dense Semantic + Metadata
            # Returns Dense Numpy Array
            return np.hstack([data_dict["semantic"], meta])

        elif model_name == "contextual_anchor":
            # Contextual Anchor: Metadata only
            return meta

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def generate_oof(self, X_train_dict, y_train):
        """
        Generates Out-Of-Fold predictions for the meta-learner using Stratified K-Fold.
        Trains the meta-learner on the resulting OOF matrix.
        """
        print("[EnsembleTrainer] Generating OOF predictions...")

        n_samples = len(y_train)
        n_models = len(self.model_names)
        oof_preds = np.zeros((n_samples, n_models))

        skf = StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
        )

        # Iterate through folds
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_train)
        ):
            print(f"  Fold {fold + 1}/{config.N_FOLDS}")

            y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # Calculate dynamic scale_pos_weight for XGBoost based on fold balance
            neg_count = np.sum(y_tr_fold == 0)
            pos_count = np.sum(y_tr_fold == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

            for i, name in enumerate(self.model_names):
                # Get a fresh model instance (or reset existing one via fit)
                # Since we don't need to save fold models, we reuse the instance in self.base_models
                model = self.base_models[name]

                # Prepare data for this model
                X_full = self._prepare_input(X_train_dict, name)
                X_tr_fold = X_full[train_idx]
                X_val_fold = X_full[val_idx]

                # Train
                if name == "semantic_booster":
                    model.set_params(scale_pos_weight=scale_pos_weight)
                    model.fit(
                        X_tr_fold,
                        y_tr_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        verbose=False,
                    )
                else:
                    model.fit(X_tr_fold, y_tr_fold)

                # Predict
                probs = model.predict_proba(X_val_fold)[:, 1]
                oof_preds[val_idx, i] = probs

        # Report OOF Scores
        print("[EnsembleTrainer] OOF Scores (AUC):")
        for i, name in enumerate(self.model_names):
            auc = roc_auc_score(y_train, oof_preds[:, i])
            print(f"  {name}: {auc}")

        # Train Meta-Learner on OOF matrix
        print("[EnsembleTrainer] Training Meta-Learner on OOF matrix...")
        self.meta_learner.fit(oof_preds, y_train)

        # Check Meta-Learner fit
        meta_probs = self.meta_learner.predict_proba(oof_preds)[:, 1]
        meta_auc = roc_auc_score(y_train, meta_probs)
        print(f"  Meta-Learner OOF AUC: {meta_auc}")

        return oof_preds

    def train_final_models(self, X_train_dict, y_train, X_val_dict, y_val):
        """
        Retrains base models using the Validation-Guided Retraining Protocol.
        - RF/Linear: Retrain on concatenated Train + Val.
        - XGBoost: Retrain on Train, use Val for Early Stopping.
        """
        print("[EnsembleTrainer] Retraining final base models...")

        # Prepare combined data for non-XGB models
        X_combined_dict = {}
        X_combined_dict["metadata"] = np.vstack(
            [X_train_dict["metadata"], X_val_dict["metadata"]]
        )
        X_combined_dict["semantic"] = np.vstack(
            [X_train_dict["semantic"], X_val_dict["semantic"]]
        )
        X_combined_dict["lexical"] = sparse.vstack(
            [X_train_dict["lexical"], X_val_dict["lexical"]]
        )
        X_combined_dict["community"] = sparse.vstack(
            [X_train_dict["community"], X_val_dict["community"]]
        )

        y_combined = np.hstack([y_train, y_val])

        for name in self.model_names:
            model = self.base_models[name]
            print(f"  Retraining {name}...")

            if name == "semantic_booster":
                # XGBoost: Train on Train, Early Stop on Val
                # Recalculate scale_pos_weight for the training set
                neg_count = np.sum(y_train == 0)
                pos_count = np.sum(y_train == 1)
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                model.set_params(scale_pos_weight=scale_pos_weight)

                X_tr = self._prepare_input(X_train_dict, name)
                X_v = self._prepare_input(X_val_dict, name)

                model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)
            else:
                # Other models: Train on Full (Train + Val)
                X_full = self._prepare_input(X_combined_dict, name)
                model.fit(X_full, y_combined)

    def predict(self, X_test_dict, test_ids):
        """
        Generates final predictions for the test set using the stacked ensemble.
        """
        print("[EnsembleTrainer] Generating test predictions...")
        n_samples = len(test_ids)
        n_models = len(self.model_names)
        L1_preds = np.zeros((n_samples, n_models))

        # Generate Level 1 Predictions
        for i, name in enumerate(self.model_names):
            model = self.base_models[name]
            X_test = self._prepare_input(X_test_dict, name)
            L1_preds[:, i] = model.predict_proba(X_test)[:, 1]

        # Generate Level 2 Prediction (Meta-Learner)
        final_probs = self.meta_learner.predict_proba(L1_preds)[:, 1]

        # Create submission DataFrame
        submission = pd.DataFrame(
            {config.ID_COL: test_ids, config.TARGET_COL: final_probs}
        )

        return submission

    def run(self, load_cached_data=True):
        """
        Main execution method.
        """
        utils.set_seed()

        with utils.timer("Full Pipeline"):
            # 1. Load Data
            X_train_dict, X_val_dict, X_test_dict = data_processing.get_processed_data(
                load_cached_data=load_cached_data
            )

            # Extract targets and IDs
            y_train = X_train_dict["y"]
            y_val = X_val_dict["y"]
            test_ids = X_test_dict["ids"]

            # 2. Generate OOF & Train Meta-Learner (using Train split)
            self.generate_oof(X_train_dict, y_train)

            # 3. Train Final Models (using Train + Val splits appropriately)
            self.train_final_models(X_train_dict, y_train, X_val_dict, y_val)

            # 4. Predict Test
            submission_df = self.predict(X_test_dict, test_ids)

            # 5. Save Submission
            print(f"[EnsembleTrainer] Saving submission to {config.SUBMISSION_PATH}")
            submission_df.to_csv(config.SUBMISSION_PATH, index=False)
            print("Done.")
