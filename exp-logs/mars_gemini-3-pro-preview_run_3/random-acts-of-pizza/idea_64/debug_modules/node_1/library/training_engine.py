import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import Config
from library.utils import Timer, set_seed
from library.data_factory import DataFactory
from library.feature_engine import FeatureGenerator
from library.model_zoo import ModelZoo


class CrossValidationTrainer:
    """
    Implements the Consistent Hybrid Inference Protocol for the High-Fidelity
    Hept-View Stacking Ensemble.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.submission_dir = Config.SUBMISSION_DIR

        # Ensure directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # Set Global Seed
        set_seed(Config.SEED)

    def _prepare_feature_set(self, feature_key, X_feature, X_meta):
        """
        Concatenates specific modality features with the global metadata vector.
        Handles Sparse/Dense compatibility.

        Args:
            feature_key (str): 'lexical', 'behavioral', 'semantic', or 'metadata'.
            X_feature: The primary feature matrix (sparse or dense).
            X_meta: The dense metadata matrix.

        Returns:
            Combined feature matrix (CSR sparse or Numpy dense).
        """
        if feature_key in ["lexical", "behavioral"]:
            # Sparse Primary + Dense Metadata -> Sparse CSR
            if not scipy.sparse.issparse(X_meta):
                X_meta_sparse = scipy.sparse.csr_matrix(X_meta)
            else:
                X_meta_sparse = X_meta

            # Use hstack for efficient concatenation
            return scipy.sparse.hstack([X_feature, X_meta_sparse], format="csr")

        elif feature_key == "semantic":
            # Dense Primary + Dense Metadata -> Dense Numpy
            return np.hstack([X_feature, X_meta])

        elif feature_key == "metadata":
            # Metadata Only -> Dense Numpy
            return X_meta

        else:
            raise ValueError(f"Unknown feature key: {feature_key}")

    def run(self):
        """
        Executes the full training pipeline:
        1. Load Data & Features
        2. Level 1 CV Training (Hybrid Volatile/Stable logic)
        3. Level 2 Meta-Learning
        4. Submission Generation
        """
        with Timer("Full Training Pipeline"):

            # =========================================================
            # 1. Data Loading & Feature Generation
            # =========================================================
            train_df, test_df = DataFactory.load_union_dataset(load_cached_data=True)

            # Extract Target and IDs
            y = train_df[Config.TARGET_COL].values.astype(int)
            request_ids = test_df[Config.ID_COL].values

            # Generate Features
            fg = FeatureGenerator(train_df, test_df)

            # Load all feature modalities
            X_train_lex, X_test_lex = fg.get_lexical_features(load_cached_data=True)
            X_train_beh, X_test_beh = fg.get_behavioral_features(load_cached_data=True)
            X_train_sem, X_test_sem = fg.get_semantic_features(load_cached_data=True)
            X_train_meta, X_test_meta = fg.get_metadata_features(load_cached_data=True)

            # =========================================================
            # 2. Feature Assembly (Concatenation)
            # =========================================================
            print("Assembling concatenated feature sets for ensemble branches...")

            # Map raw features to keys
            raw_features = {
                "lexical": (X_train_lex, X_test_lex),
                "behavioral": (X_train_beh, X_test_beh),
                "semantic": (X_train_sem, X_test_sem),
                "metadata": (X_train_meta, X_test_meta),
            }

            # Pre-assemble inputs for each branch type
            inputs_train = {}
            inputs_test = {}

            for f_key in raw_features.keys():
                X_tr_raw, X_te_raw = raw_features[f_key]

                # Metadata branch uses ONLY metadata. Others use Feature + Metadata.
                if f_key == "metadata":
                    inputs_train[f_key] = X_tr_raw
                    inputs_test[f_key] = X_te_raw
                else:
                    inputs_train[f_key] = self._prepare_feature_set(
                        f_key, X_tr_raw, X_train_meta
                    )
                    inputs_test[f_key] = self._prepare_feature_set(
                        f_key, X_te_raw, X_test_meta
                    )

            # =========================================================
            # 3. Level 1 Ensemble Training
            # =========================================================
            models_conf = ModelZoo.get_models_dict()

            # Containers for Level 2 Data
            oof_preds_df = pd.DataFrame()
            test_preds_df = pd.DataFrame()

            skf = StratifiedKFold(
                n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
            )

            for model_name, conf in models_conf.items():
                print(
                    f"\n--- Training Level 1 Model: {model_name} [{conf['type']}] ---"
                )

                # Select appropriate feature set
                feature_key = conf["feature_set"]
                X = inputs_train[feature_key]
                X_test_full = inputs_test[feature_key]

                # Initialize arrays
                oof_preds = np.zeros(len(y))
                test_preds_accum = np.zeros(X_test_full.shape[0])

                # Cross-Validation Loop
                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    # Clone fresh model instance
                    model = clone(conf["model"])

                    if conf["type"] == "volatile":
                        # Volatile: Train with Early Stopping, Predict Test (CV-Bagging)
                        try:
                            model.fit(
                                X_tr,
                                y_tr,
                                eval_set=[(X_val, y_val)],
                                early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                                verbose=False,
                            )
                        except TypeError:
                            # Fallback if specific wrapper version has different API
                            model.fit(
                                X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False
                            )

                        # Predict Val (OOF)
                        val_pred = model.predict_proba(X_val)[:, 1]

                        # Predict Test (Accumulate)
                        test_pred = model.predict_proba(X_test_full)[:, 1]
                        test_preds_accum += test_pred

                    else:
                        # Stable: Train on Fold, Predict Val Only
                        model.fit(X_tr, y_tr)
                        val_pred = model.predict_proba(X_val)[:, 1]
                        # Test prediction happens after full retrain

                    # Store OOF
                    oof_preds[val_idx] = val_pred

                    # Persistence: Save fold model
                    model_path = os.path.join(
                        self.working_dir, f"{model_name}_fold_{fold}.joblib"
                    )
                    joblib.dump(model, model_path)

                # Evaluate OOF
                auc_score = roc_auc_score(y, oof_preds)
                print(f"  {model_name} OOF AUC: {auc_score}")

                # Store OOF predictions
                oof_preds_df[model_name] = oof_preds

                # Final Test Predictions Logic
                if conf["type"] == "volatile":
                    # Average the accumulated predictions
                    final_test_pred = test_preds_accum / Config.N_FOLDS
                else:
                    # Stable: Retrain on Full Dataset
                    print(f"  Retraining {model_name} on full Union Dataset...")
                    full_model = clone(conf["model"])
                    full_model.fit(X, y)
                    final_test_pred = full_model.predict_proba(X_test_full)[:, 1]

                    # Save full model
                    joblib.dump(
                        full_model,
                        os.path.join(self.working_dir, f"{model_name}_full.joblib"),
                    )

                test_preds_df[model_name] = final_test_pred

            # =========================================================
            # 4. Level 2 Meta-Learner Training
            # =========================================================
            print("\n--- Training Level 2 Meta-Learner ---")

            X_meta_train = oof_preds_df.values
            X_meta_test = test_preds_df.values

            meta_learner = ModelZoo.get_meta_learner()
            meta_learner.fit(X_meta_train, y)

            # Evaluate Meta-Learner on OOF (Proxy for performance)
            meta_oof_probs = meta_learner.predict_proba(X_meta_train)[:, 1]
            meta_auc = roc_auc_score(y, meta_oof_probs)
            print(f"Meta-Learner OOF AUC: {meta_auc}")

            # Save Meta-Learner
            joblib.dump(
                meta_learner, os.path.join(self.working_dir, "meta_learner.joblib")
            )

            # =========================================================
            # 5. Submission Generation
            # =========================================================
            print("Generating submission...")

            final_test_probs = meta_learner.predict_proba(X_meta_test)[:, 1]

            submission = pd.DataFrame(
                {
                    "request_id": request_ids,
                    "requester_received_pizza": final_test_probs,
                }
            )

            submission.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
