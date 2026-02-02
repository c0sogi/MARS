import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import library.config as config
from library.feature_engineering import FeatureProcessor
from library.model_definitions import ModelFactory


class HexViewEnsemble:
    """
    Implements the Clean-Signal Hex-View Stacking Ensemble pipeline.
    Manages data preparation, Level 1 OOF generation, Meta-Learner training,
    Final Retraining, and Test Prediction.
    """

    def __init__(self):
        self.models_dir = os.path.join(config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.processor = FeatureProcessor()

        # Define the 6 base learners and their required feature views
        # Format: (Model Name, Factory Method, [Feature Views to Concatenate])
        self.base_learners_config = [
            (
                "lexical_bagger",
                ModelFactory.get_lexical_bagger,
                ["X_lexical", "X_meta"],
            ),
            (
                "community_bagger",
                ModelFactory.get_community_bagger,
                ["X_behavioral", "X_meta"],
            ),
            (
                "semantic_booster",
                ModelFactory.get_semantic_booster,
                ["X_semantic", "X_meta"],
            ),
            (
                "semantic_bagger",
                ModelFactory.get_semantic_bagger,
                ["X_semantic", "X_meta"],
            ),
            ("metadata_anchor", ModelFactory.get_metadata_anchor, ["X_meta"]),
            ("temporal_booster", ModelFactory.get_temporal_booster, ["X_meta"]),
        ]

    def _get_model_input(self, data_split, view_keys):
        """
        Constructs the input matrix for a specific model by concatenating required views.
        Handles Sparse (CSR) and Dense (Numpy) combinations.

        Args:
            data_split (dict): Dictionary containing X matrices for a specific split (train/val/test).
            view_keys (list): List of keys corresponding to views in data_split (e.g., ['X_lexical', 'X_meta']).
        """
        matrices = [data_split[k] for k in view_keys]

        # Check if any matrix is sparse
        is_sparse = any(sp.issparse(m) for m in matrices)

        if is_sparse:
            # Convert all to sparse if one is sparse to allow hstack
            # Note: Dense matrices are cast to sparse for concatenation
            return sp.hstack(matrices, format="csr")
        else:
            return np.hstack(matrices)

    def train_and_predict_oof(self, load_cached_data=True):
        """
        Executes 5-Fold Stratified CV to generate OOF predictions and trains the Meta-Learner.
        """
        print("Fetching data...")
        data = self.processor.run(load_cached_data=load_cached_data)

        X_train_raw = data["train"]
        y_train = data["train"]["y"]

        # Initialize OOF matrix: (n_samples, n_models)
        n_samples = len(y_train)
        n_models = len(self.base_learners_config)
        oof_preds = np.zeros((n_samples, n_models))

        skf = StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
        )

        print(f"Starting Level 1 OOF Generation ({config.N_FOLDS} folds)...")

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_train)
        ):
            print(f"  Processing Fold {fold + 1}/{config.N_FOLDS}...")

            # Slice targets
            y_tr_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            for model_idx, (name, factory_func, views) in enumerate(
                self.base_learners_config
            ):
                # Prepare inputs for this specific model
                # We need to slice the raw matrices provided by processor
                # Since processor returns full train matrices, we slice by index

                # Construct full matrix first (efficient enough for this dataset size)
                full_X = self._get_model_input(X_train_raw, views)

                # Slice for fold
                X_tr_fold = full_X[train_idx]
                X_val_fold = full_X[val_idx]

                # Instantiate and Train
                model = factory_func()

                # Handle fit params for Boosting models if needed (though OOF usually just fits on train)
                # For OOF, we typically fit on train_idx and predict on val_idx.
                # Early stopping inside CV is tricky without a nested split,
                # but standard practice for stacking is often just fit.
                # However, XGB/LGBM might benefit from early stopping.
                # We will use a simple fit here as per standard stacking implementations
                # unless we want to split train_idx further.
                # Given the prompt instructions for "Final Retraining" are specific about early stopping,
                # but OOF instructions are generic ("Train on 4 folds, predict on 5th"),
                # we will proceed with standard fit.

                model.fit(X_tr_fold, y_tr_fold)

                # Predict
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_val_fold)[:, 1]
                else:
                    # Fallback if needed, though all classifiers here support proba
                    preds = model.predict(X_val_fold)

                oof_preds[val_idx, model_idx] = preds

        # Calculate and Print OOF Scores
        print("\nLevel 1 OOF Performance:")
        for i, (name, _, _) in enumerate(self.base_learners_config):
            auc = roc_auc_score(y_train, oof_preds[:, i])
            print(f"  {name}: AUC = {auc}")

        # Train Meta-Learner
        print("\nTraining Level 2 Meta-Learner...")
        meta_learner = ModelFactory.get_meta_learner()
        meta_learner.fit(oof_preds, y_train)

        meta_auc = roc_auc_score(y_train, meta_learner.predict_proba(oof_preds)[:, 1])
        print(f"  Meta-Learner CV AUC = {meta_auc}")

        # Save Meta-Learner
        joblib.dump(meta_learner, os.path.join(self.models_dir, "meta_learner.joblib"))

        return oof_preds, y_train

    def train_final_models(self, load_cached_data=True):
        """
        Retrains base learners according to the Validation-Guided Retraining Protocol.
        """
        print("\nStarting Final Retraining of Base Learners...")
        data = self.processor.run(load_cached_data=load_cached_data)

        # Raw Data
        train_data = data["train"]
        val_data = data["val"]
        y_train = train_data["y"]
        y_val = val_data["y"]

        for name, factory_func, views in self.base_learners_config:
            print(f"  Retraining {name}...")
            model = factory_func()

            # Prepare Inputs
            X_train = self._get_model_input(train_data, views)
            X_val = self._get_model_input(val_data, views)

            # Logic Branch: Boosting vs Bagging/Linear
            if "booster" in name:
                # XGBoost / LightGBM: Train on Train, Early Stop on Val
                eval_set = [(X_val, y_val)]
                model.fit(X_train, y_train, eval_set=eval_set)
            else:
                # RF / Linear: Train on Train + Val
                # Combine inputs
                if sp.issparse(X_train):
                    X_full = sp.vstack([X_train, X_val], format="csr")
                else:
                    X_full = np.vstack([X_train, X_val])

                y_full = np.concatenate([y_train, y_val])

                model.fit(X_full, y_full)

            # Save Model
            joblib.dump(model, os.path.join(self.models_dir, f"{name}.joblib"))

        print("Final models saved.")

    def predict_test(self, load_cached_data=True):
        """
        Generates predictions for the test set using the ensemble.
        """
        print("\nGenerating Test Predictions...")
        data = self.processor.run(load_cached_data=load_cached_data)
        test_data = data["test"]

        # Load Meta-Learner
        meta_learner_path = os.path.join(self.models_dir, "meta_learner.joblib")
        if not os.path.exists(meta_learner_path):
            raise FileNotFoundError(
                "Meta-learner not found. Run train_and_predict_oof first."
            )
        meta_learner = joblib.load(meta_learner_path)

        # Generate Level 1 Test Predictions
        n_test = 0
        # Determine test size from one of the views
        for k in test_data:
            if hasattr(test_data[k], "shape"):
                n_test = test_data[k].shape[0]
                break

        L1_test_preds = np.zeros((n_test, len(self.base_learners_config)))

        for i, (name, _, views) in enumerate(self.base_learners_config):
            model_path = os.path.join(self.models_dir, f"{name}.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model {name} not found. Run train_final_models first."
                )

            model = joblib.load(model_path)
            X_test = self._get_model_input(test_data, views)

            if hasattr(model, "predict_proba"):
                L1_test_preds[:, i] = model.predict_proba(X_test)[:, 1]
            else:
                L1_test_preds[:, i] = model.predict(X_test)

        # Generate Level 2 Predictions
        final_probs = meta_learner.predict_proba(L1_test_preds)[:, 1]

        # Load Test IDs for Submission
        # We need to read the test metadata parquet to get IDs
        test_meta_df = pd.read_parquet(config.TEST_METADATA_PATH)
        request_ids = test_meta_df[config.ID_COL].values

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {config.ID_COL: request_ids, config.TARGET_COL: final_probs}
        )

        # Save
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
