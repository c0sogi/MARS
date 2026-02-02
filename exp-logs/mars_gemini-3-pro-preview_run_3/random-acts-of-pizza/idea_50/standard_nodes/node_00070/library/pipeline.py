import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import N_FOLDS, SEED, WORKING_DIR, SUBMISSION_FILE
from library.utils import Timer, set_seed, save_submission
from library.models import ModelFactory


class HybridStackingRunner:
    """
    Implements the Hex-View Stacking Ensemble with Hybrid Inference Protocol.
    Orchestrates Level 1 training (CV), Hybrid Inference (Bagging vs Retraining),
    and Level 2 Meta-Learning.
    """

    def __init__(self):
        set_seed(SEED)
        self.models_config = ModelFactory.get_level1_models()
        self.meta_learner = ModelFactory.get_meta_learner()

        # Define which models are 'Volatile' (Boosters) and which are 'Stable' (Baggers/Linear)
        # Volatile models use CV-Bagging (Average of fold models) to utilize Early Stopping safely.
        # Stable models use Full-Retraining (One model on all data) to maximize data utilization.
        self.volatile_models = ["semantic_booster", "temporal_booster"]

        # Directory for saving intermediate models
        self.model_dir = os.path.join(WORKING_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def _merge_features(self, feats1, feats2):
        """
        Merges two feature dictionaries (e.g., train and val) into one.
        Handles both sparse (scipy) and dense (numpy) arrays.
        """
        merged = {}
        for key in feats1.keys():
            d1 = feats1[key]
            d2 = feats2[key]

            if scipy.sparse.issparse(d1):
                merged[key] = scipy.sparse.vstack([d1, d2])
            else:
                merged[key] = np.vstack([d1, d2])
        return merged

    def _slice_features(self, feats, indices):
        """
        Slices a feature dictionary based on indices.
        """
        sliced = {}
        for key, data in feats.items():
            sliced[key] = data[indices]
        return sliced

    def run_stacking(
        self, train_feats, val_feats, test_feats, train_y, val_y, test_ids
    ):
        """
        Executes the full stacking pipeline:
        1. Merge Train + Val for CV.
        2. Level 1: 5-Fold CV to generate OOF preds and Fold Test preds.
        3. Hybrid Inference:
           - Average Test preds for Volatile models.
           - Retrain Stable models on full data and predict Test.
        4. Level 2: Train Meta-Learner on OOF, predict on stacked Test.
        5. Save Submission.
        """

        # 1. Merge Data
        with Timer("Merging Train and Val for Cross-Validation"):
            X_full = self._merge_features(train_feats, val_feats)
            y_full = np.concatenate([train_y, val_y])

            print(f"Full Training Set Size: {len(y_full)}")
            print(f"Test Set Size: {test_feats['metadata'].shape[0]}")

        # Initialize containers
        n_samples = len(y_full)
        n_test = test_feats["metadata"].shape[0]
        model_names = list(self.models_config.keys())

        # OOF Predictions: [n_samples, n_models]
        oof_preds = pd.DataFrame(index=range(n_samples), columns=model_names)

        # Test Predictions Accumulator for Volatile Models: {model_name: list_of_fold_preds}
        volatile_test_preds = {m: [] for m in self.volatile_models}

        # Final Level 1 Test Predictions: [n_test, n_models]
        l1_test_preds = pd.DataFrame(index=range(n_test), columns=model_names)

        # 2. Level 1 Cross-Validation Loop
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        print(
            f"\n{'='*10} Starting Level 1: {N_FOLDS} Folds for OOF + Hybrid Inference {'='*10}"
        )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y_full)
        ):
            print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

            # Slice data
            X_fold_train = self._slice_features(X_full, train_idx)
            y_fold_train = y_full[train_idx]
            X_fold_val = self._slice_features(X_full, val_idx)
            y_fold_val = y_full[val_idx]

            for name, base_model_wrapper in self.models_config.items():
                # Clone the internal sklearn/xgb model
                model_instance = clone(base_model_wrapper.model)
                # Re-wrap in the HexModel wrapper
                wrapper = base_model_wrapper.__class__(model_instance)

                # Fit
                # If volatile (booster), use early stopping with val set
                if name in self.volatile_models:
                    wrapper.fit(
                        X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)]
                    )
                else:
                    wrapper.fit(X_fold_train, y_fold_train)

                # Predict OOF (Val)
                val_probs = wrapper.predict_proba(X_fold_val)[:, 1]
                oof_preds.loc[val_idx, name] = val_probs

                # Calculate Fold Score
                fold_auc = roc_auc_score(y_fold_val, val_probs)
                print(f"  {name}: AUC = {fold_auc:.10f}")

                # Handle Inference Strategy
                if name in self.volatile_models:
                    # CV-Bagging: Predict on Test and store
                    test_probs = wrapper.predict_proba(test_feats)[:, 1]
                    volatile_test_preds[name].append(test_probs)

                    # Save fold model
                    joblib.dump(
                        wrapper,
                        os.path.join(self.model_dir, f"{name}_fold_{fold}.joblib"),
                    )

                # Stable models are NOT predicted on test here (waste of time),
                # they are retrained later on full data.

        # 3. Hybrid Inference: Finalize Level 1 Test Predictions
        print(f"\n{'='*10} Finalizing Level 1 Predictions (Hybrid Protocol) {'='*10}")

        # A. Process Volatile Models (Average of Folds)
        for name in self.volatile_models:
            print(f"Processing Volatile Model: {name} (CV-Bagging)")
            # Stack folds [n_folds, n_test] -> Mean -> [n_test]
            avg_preds = np.mean(np.vstack(volatile_test_preds[name]), axis=0)
            l1_test_preds[name] = avg_preds

        # B. Process Stable Models (Full Retraining)
        stable_models = [m for m in model_names if m not in self.volatile_models]

        for name in stable_models:
            print(f"Processing Stable Model: {name} (Full Retraining)")
            base_model_wrapper = self.models_config[name]

            # Clone and Fit on FULL data
            model_instance = clone(base_model_wrapper.model)
            wrapper = base_model_wrapper.__class__(model_instance)

            wrapper.fit(X_full, y_full)

            # Predict on Test
            test_probs = wrapper.predict_proba(test_feats)[:, 1]
            l1_test_preds[name] = test_probs

            # Save final model
            joblib.dump(wrapper, os.path.join(self.model_dir, f"{name}.joblib"))

        # 4. Level 2: Meta-Learner
        print(f"\n{'='*10} Level 2: Meta-Learner Stacking {'='*10}")

        # Prepare Inputs
        X_level2_train = oof_preds.values.astype(float)
        X_level2_test = l1_test_preds.values.astype(float)

        # Evaluate Level 1 Performance on OOF
        print("Level 1 OOF Performance:")
        for i, name in enumerate(model_names):
            auc = roc_auc_score(y_full, X_level2_train[:, i])
            print(f"  {name}: OOF AUC = {auc:.10f}")

        # Train Meta-Learner
        self.meta_learner.fit(X_level2_train, y_full)

        # Meta-Learner Coefficients
        print("\nMeta-Learner Coefficients:")
        for name, coef in zip(model_names, self.meta_learner.coef_[0]):
            print(f"  {name}: {coef:.4f}")

        # Save Meta-Learner
        joblib.dump(
            self.meta_learner, os.path.join(self.model_dir, "meta_learner.joblib")
        )

        # 5. Final Prediction
        final_preds = self.meta_learner.predict_proba(X_level2_test)[:, 1]

        # Save Submission
        save_submission(final_preds, test_ids, SUBMISSION_FILE)

        return final_preds
