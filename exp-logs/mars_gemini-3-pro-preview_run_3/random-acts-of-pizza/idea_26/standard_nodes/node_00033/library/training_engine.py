import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from library import config
from library import utils
from library.model_zoo import ModelFactory


class StackingTrainer:
    """
    Orchestrates the training of the Pent-View Hybrid-Topology Stacking Ensemble.
    Implements 5-Fold CV for OOF generation and the Validation-Guided Retraining Protocol.
    """

    def __init__(self):
        # Define the 5 base learners and their corresponding view keys
        self.base_learners = {
            "lexical_bagger": {
                "model": ModelFactory.get_lexical_bagger(),
                "view": "lexical",
                "type": "sparse_rf",
            },
            "community_bagger": {
                "model": ModelFactory.get_community_bagger(),
                "view": "behavioral",
                "type": "sparse_rf",
            },
            "semantic_booster": {
                "model": ModelFactory.get_semantic_booster(),
                "view": "semantic",
                "type": "dense_xgb",
            },
            "semantic_bagger": {
                "model": ModelFactory.get_semantic_bagger(),
                "view": "semantic",
                "type": "dense_rf",
            },
            "metadata_anchor": {
                "model": ModelFactory.get_metadata_anchor(),
                "view": "metadata",
                "type": "dense_lr",
            },
        }

        self.meta_learner = ModelFactory.get_meta_learner()
        self.final_models = {}  # Store retrained models here
        self.n_folds = 5

    def _slice_data(self, data, indices):
        """
        Slices data (numpy array or sparse matrix) based on indices.
        """
        if sparse.issparse(data):
            return data[indices]
        else:
            return data[indices]

    def _concat_data(self, data1, data2):
        """
        Concatenates two datasets (numpy array or sparse matrix).
        """
        if sparse.issparse(data1) or sparse.issparse(data2):
            # Ensure both are sparse for efficient stacking
            s1 = sparse.csr_matrix(data1)
            s2 = sparse.csr_matrix(data2)
            return sparse.vstack([s1, s2])
        else:
            return np.vstack([data1, data2])

    def _get_learner_data(self, views, learner_name):
        """
        Retrieves the specific view required by a learner.
        """
        view_key = self.base_learners[learner_name]["view"]
        return views[view_key]

    def fit(self, X_train_views, y_train, X_val_views, y_val):
        """
        Executes the full training pipeline:
        1. 5-Fold CV on X_train to generate OOF predictions.
        2. Train Meta-Learner on OOF predictions.
        3. Retrain Base Learners using the Validation-Guided Protocol.
        """
        utils.print_header("Starting Stacking Ensemble Training")

        # --- Step 1: Generate OOF Predictions ---
        utils.print_info(f"Generating OOF predictions using {self.n_folds}-Fold CV...")

        # Initialize OOF matrix: (n_samples, n_base_learners)
        oof_preds = np.zeros((len(y_train), len(self.base_learners)))
        learner_names = list(self.base_learners.keys())

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=config.SEED
        )

        # Iterate through folds
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(y_train)), y_train)
        ):
            utils.print_info(f"Processing Fold {fold + 1}/{self.n_folds}...")

            y_fold_train = y_train.iloc[train_idx]
            # y_fold_val = y_train.iloc[val_idx] # Not strictly needed for prediction, just X

            for i, name in enumerate(learner_names):
                learner_conf = self.base_learners[name]

                # Get data for this specific learner
                full_view = self._get_learner_data(X_train_views, name)
                X_fold_train = self._slice_data(full_view, train_idx)
                X_fold_val = self._slice_data(full_view, val_idx)

                # clone/re-instantiate model to avoid leakage between folds
                if learner_conf["type"] == "dense_xgb":
                    model = ModelFactory.get_semantic_booster()
                    # For CV, we can use the fold validation set for ES, or just train fixed.
                    # To be consistent with standard OOF, we'll use the fold val for ES.
                    fit_params = config.SEMANTIC_BOOSTER_FIT_PARAMS.copy()
                    fit_params["eval_set"] = [(X_fold_val, y_train.iloc[val_idx])]
                    model.fit(X_fold_train, y_fold_train, **fit_params)
                elif learner_conf["type"] == "sparse_rf":
                    if name == "lexical_bagger":
                        model = ModelFactory.get_lexical_bagger()
                    else:
                        model = ModelFactory.get_community_bagger()
                    model.fit(X_fold_train, y_fold_train)
                elif learner_conf["type"] == "dense_rf":
                    model = ModelFactory.get_semantic_bagger()
                    model.fit(X_fold_train, y_fold_train)
                elif learner_conf["type"] == "dense_lr":
                    model = ModelFactory.get_metadata_anchor()
                    model.fit(X_fold_train, y_fold_train)

                # Predict on fold validation set
                preds = model.predict_proba(X_fold_val)[:, 1]
                oof_preds[val_idx, i] = preds

        # Evaluate OOF Performance
        overall_auc = roc_auc_score(
            y_train, oof_preds.mean(axis=1)
        )  # Simple average for quick check
        utils.print_info(
            f"OOF Generation Complete. Average Ensemble AUC (Simple Mean): {overall_auc}"
        )

        # --- Step 2: Train Meta-Learner ---
        utils.print_info("Training Level 2 Meta-Learner on OOF predictions...")
        self.meta_learner.fit(oof_preds, y_train)

        # Calculate Meta-Learner OOF AUC (Unbiased Metric)
        meta_oof_preds = self.meta_learner.predict_proba(oof_preds)[:, 1]
        meta_oof_auc = roc_auc_score(y_train, meta_oof_preds)
        utils.print_info(f"Meta-Learner OOF AUC: {meta_oof_auc}")

        # Check Meta-Learner coefficients
        utils.print_info(
            f"Meta-Learner Coefficients: {dict(zip(learner_names, self.meta_learner.coef_[0]))}"
        )

        # --- Step 3: Final Retraining (Validation-Guided Protocol) ---
        utils.print_info("Performing Final Retraining of Base Learners...")

        for name in learner_names:
            learner_conf = self.base_learners[name]
            utils.print_info(f"Retraining {name} ({learner_conf['type']})...")

            # Get data views
            X_train_view = self._get_learner_data(X_train_views, name)
            X_val_view = self._get_learner_data(X_val_views, name)

            if learner_conf["type"] == "dense_xgb":
                # XGBoost: Train on Train, use Val for Early Stopping
                # This prevents blind overfitting by using the held-out validation set
                model = ModelFactory.get_semantic_booster()
                fit_params = config.SEMANTIC_BOOSTER_FIT_PARAMS.copy()
                fit_params["eval_set"] = [(X_val_view, y_val)]

                model.fit(X_train_view, y_train, **fit_params)

                # Log validation score
                val_score = model.best_score
                utils.print_info(f"  {name} Best Validation AUC: {val_score}")
                self.final_models[name] = model

            else:
                # RF / LR: Train on Full Data (Train + Val)
                # Combine datasets
                X_full = self._concat_data(X_train_view, X_val_view)
                y_full = pd.concat([y_train, y_val], axis=0)

                if learner_conf["type"] == "sparse_rf":
                    if name == "lexical_bagger":
                        model = ModelFactory.get_lexical_bagger()
                    else:  # community_bagger
                        model = ModelFactory.get_community_bagger()
                elif learner_conf["type"] == "dense_rf":
                    model = ModelFactory.get_semantic_bagger()
                elif learner_conf["type"] == "dense_lr":
                    model = ModelFactory.get_metadata_anchor()

                model.fit(X_full, y_full)
                self.final_models[name] = model

        utils.print_info("Training Complete.")
        return meta_oof_auc

    def predict(self, X_test_views):
        """
        Generates predictions for the test set using the stacked ensemble.
        """
        utils.print_info("Generating predictions on Test Set...")
        learner_names = list(self.base_learners.keys())
        n_samples = self._get_learner_data(X_test_views, learner_names[0]).shape[0]

        # Level 1 Predictions
        l1_preds = np.zeros((n_samples, len(learner_names)))

        for i, name in enumerate(learner_names):
            model = self.final_models[name]
            X_test_view = self._get_learner_data(X_test_views, name)

            # Predict
            preds = model.predict_proba(X_test_view)[:, 1]
            l1_preds[:, i] = preds

        # Level 2 Prediction (Meta-Learner)
        final_preds = self.meta_learner.predict_proba(l1_preds)[:, 1]

        return final_preds

    def save_predictions(self, request_ids, probabilities, output_path):
        """
        Saves predictions to CSV in the required submission format.
        """
        utils.print_info(f"Saving predictions to {output_path}...")

        submission_df = pd.DataFrame(
            {"request_id": request_ids, "requester_received_pizza": probabilities}
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        submission_df.to_csv(output_path, index=False)
        utils.print_info("Submission saved successfully.")
