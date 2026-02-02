import os
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import early_stopping, log_evaluation

from library.config import Config
from library.feature_engineering import FeaturePipeline
from library.model_factory import ModelFactory


class TrainingEngine:
    """
    Orchestrates the CV-Bagging training protocol for the Hex-View Stacking Ensemble.

    Responsibilities:
    1. Data Preparation: Merges Train/Val splits for 5-Fold CV and constructs modality-specific feature sets.
    2. Level 1 Training: Trains 6 base learners per fold (30 models total) with Early Stopping.
    3. Inference Bagging: Generates Test predictions by averaging outputs from all fold models.
    4. Level 2 Stacking: Trains a Meta-Learner on OOF predictions.
    5. Submission: Generates the final submission file.
    """

    def __init__(self, load_cached_data=True):
        self.feature_pipeline = FeaturePipeline(load_cached_data=load_cached_data)
        self.models_dir = Config.MODEL_DIR
        self.submission_path = Config.SUBMISSION_PATH

        # Define the list of base learners and their required feature modalities
        # Format: (learner_name, feature_type)
        # feature_type maps to how we construct the input matrix
        self.learners = [
            ("lexical_bagger", "lexical_meta"),
            ("community_bagger", "community_meta"),
            ("semantic_booster", "semantic_meta"),
            ("semantic_bagger", "semantic_meta"),
            ("temporal_booster", "meta_only"),
            ("metadata_anchor", "meta_only"),
        ]

    def _prepare_data(self):
        """
        Loads features, merges Train+Val for CV, and constructs specific input matrices.
        """
        print("Loading and preparing feature sets...")
        data = self.feature_pipeline.get_all_features()

        # 1. Merge Train and Val for full Cross-Validation
        # We do this to maximize data utility and generate unbiased OOFs for the meta-learner
        y_full = np.concatenate([data["y_train"], data["y_val"]])

        # Helper to merge and stack
        def merge_and_stack(
            train_key,
            val_key,
            test_key,
            meta_train_key,
            meta_val_key,
            meta_test_key,
            is_sparse=False,
        ):
            # Get primary modality
            feat_train = data[train_key]
            feat_val = data[val_key]
            feat_test = data[test_key]

            # Get metadata
            meta_train = data[meta_train_key]
            meta_val = data[meta_val_key]
            meta_test = data[meta_test_key]

            # Concatenate Train + Val
            if is_sparse:
                feat_full = sparse.vstack([feat_train, feat_val])
                meta_full = (
                    meta_train
                    if meta_train.shape[0] == feat_full.shape[0]
                    else np.concatenate([meta_train, meta_val])
                )

                # Stack Primary + Meta
                # Note: meta is dense, need to treat as sparse for hstack or convert
                X_full = sparse.hstack([feat_full, sparse.csr_matrix(meta_full)])
                X_test = sparse.hstack([feat_test, sparse.csr_matrix(meta_test)])
            else:
                feat_full = np.concatenate([feat_train, feat_val])
                meta_full = np.concatenate([meta_train, meta_val])

                # Stack Primary + Meta
                X_full = np.hstack([feat_full, meta_full])
                X_test = np.hstack([feat_test, meta_test])

            return X_full, X_test

        # Construct specific inputs
        inputs = {}

        # A. Lexical + Meta (Sparse)
        inputs["lexical_meta"] = merge_and_stack(
            "X_train_lexical",
            "X_val_lexical",
            "X_test_lexical",
            "X_train_meta",
            "X_val_meta",
            "X_test_meta",
            is_sparse=True,
        )

        # B. Community + Meta (Sparse)
        inputs["community_meta"] = merge_and_stack(
            "X_train_community",
            "X_val_community",
            "X_test_community",
            "X_train_meta",
            "X_val_meta",
            "X_test_meta",
            is_sparse=True,
        )

        # C. Semantic + Meta (Dense)
        inputs["semantic_meta"] = merge_and_stack(
            "X_train_semantic",
            "X_val_semantic",
            "X_test_semantic",
            "X_train_meta",
            "X_val_meta",
            "X_test_meta",
            is_sparse=False,
        )

        # D. Meta Only (Dense)
        # Just merge train/val
        meta_full = np.concatenate([data["X_train_meta"], data["X_val_meta"]])
        meta_test = data["X_test_meta"]
        inputs["meta_only"] = (meta_full, meta_test)

        return inputs, y_full, data["test_ids"]

    def run(self, n_folds=5):
        """
        Executes the full training pipeline.
        """
        # 1. Prepare Data
        inputs_map, y, test_ids = self._prepare_data()

        n_samples = y.shape[0]
        n_test = len(test_ids)
        n_learners = len(self.learners)

        # Storage for OOF predictions (Train) and Bagged predictions (Test)
        # Shape: [n_samples, n_learners]
        oof_preds = np.zeros((n_samples, n_learners))
        # Shape: [n_test, n_learners]
        test_preds_accum = np.zeros((n_test, n_learners))

        # 2. Stratified K-Fold CV
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

        print(f"\nStarting Level 1 Training ({n_folds}-Fold CV)...")

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n_samples), y)
        ):
            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

            y_tr, y_va = y[train_idx], y[val_idx]

            for learner_idx, (learner_name, input_type) in enumerate(self.learners):
                print(f"Training {learner_name}...", end=" ")

                # Get data for this learner
                X_full, X_test_base = inputs_map[input_type]

                # Split for this fold
                # Note: X_full can be sparse or dense
                if sparse.issparse(X_full):
                    X_tr = X_full[train_idx]
                    X_va = X_full[val_idx]
                else:
                    X_tr = X_full[train_idx]
                    X_va = X_full[val_idx]

                # Instantiate model
                model = ModelFactory.get_base_learner(learner_name)

                # Fit with Early Stopping where applicable
                if learner_name == "semantic_booster":  # XGBoost
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                elif learner_name == "temporal_booster":  # LightGBM
                    callbacks = [
                        early_stopping(stopping_rounds=50, verbose=False),
                        log_evaluation(period=0),  # Silence
                    ]
                    model.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_va, y_va)],
                        eval_metric="auc",
                        callbacks=callbacks,
                    )
                else:
                    # RF / LR / Others
                    model.fit(X_tr, y_tr)

                # Save Fold Model
                model_filename = f"{learner_name}_fold_{fold_idx}.joblib"
                joblib.dump(model, os.path.join(self.models_dir, model_filename))

                # Predict OOF (Validation)
                # Use predict_proba for classification
                val_probs = model.predict_proba(X_va)[:, 1]
                oof_preds[val_idx, learner_idx] = val_probs

                # Predict Test (Accumulate for Bagging)
                test_probs = model.predict_proba(X_test_base)[:, 1]
                test_preds_accum[:, learner_idx] += test_probs

                # Fold Metric
                auc = roc_auc_score(y_va, val_probs)
                print(f"AUC: {auc:.16f}")

        # Average Test Predictions (Bagging)
        test_preds_avg = test_preds_accum / n_folds

        # 3. Level 2 Meta-Learner
        print("\nStarting Level 2 Training (Meta-Learner)...")

        # Calculate OOF Score
        for i, (name, _) in enumerate(self.learners):
            auc = roc_auc_score(y, oof_preds[:, i])
            print(f"L1 OOF AUC - {name}: {auc:.16f}")

        # Train Meta-Learner
        meta_learner = ModelFactory.get_meta_learner()
        meta_learner.fit(oof_preds, y)

        # Save Meta-Learner
        joblib.dump(meta_learner, os.path.join(self.models_dir, "meta_learner.joblib"))

        # Meta-Learner Coefficients
        print("Meta-Learner Coefficients:")
        for name, coef in zip([l[0] for l in self.learners], meta_learner.coef_[0]):
            print(f"  {name}: {coef:.6f}")

        # 4. Final Prediction
        print("\nGenerating Final Submission...")
        final_probs = meta_learner.predict_proba(test_preds_avg)[:, 1]

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_probs}
        )

        # Save
        submission.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")

        # Print sample
        print("Sample predictions:")
        print(submission.head())
