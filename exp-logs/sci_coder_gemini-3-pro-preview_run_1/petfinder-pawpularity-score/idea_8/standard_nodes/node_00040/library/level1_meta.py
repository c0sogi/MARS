import os
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import (
    save_array,
    load_array,
    print_metric,
    seed_everything,
    ensure_dir,
)
from library.level0_experts import Level0Trainer


class MetaLearner:
    """
    Level-1 Meta-Learner for the Stacking Ensemble.
    Aggregates predictions from Level-0 experts and generates final submission.
    """

    def __init__(self):
        self.backbones = [Config.MODEL_CLIP, Config.MODEL_DINO, Config.MODEL_CONVNEXT]
        self.experts = ["ridge", "svr", "extratrees"]
        self.n_folds = Config.N_FOLDS
        self.seed = Config.SEED

        # Define cache paths for meta-features
        self.cache_dir = Config.WORKING_DIR
        self.path_X_meta_train = os.path.join(self.cache_dir, "meta_X_train.npy")
        self.path_y_meta_train = os.path.join(self.cache_dir, "meta_y_train.npy")
        self.path_X_meta_test = os.path.join(self.cache_dir, "meta_X_test.npy")
        self.path_ids_test = os.path.join(self.cache_dir, "meta_ids_test.npy")

    def _collect_level0_data(self, load_cached_experts=True):
        """
        Iterates through all backbone-expert combinations to build the meta-feature matrices.
        """
        print("Collecting Level-0 Expert Predictions...")

        meta_train_list = []
        meta_test_list = []

        y_train_ref = None
        ids_test_ref = None

        # Iterate over all combinations (3 backbones * 4 experts = 12 features)
        for backbone in self.backbones:
            for expert in self.experts:
                # Initialize trainer for specific expert
                trainer = Level0Trainer(backbone, expert)

                # Run or retrieve results
                # We typically want to load cached L0 results to save time
                results = trainer.run(load_cached_data=load_cached_experts)

                # Extract predictions
                oof_preds = results["oof"]
                test_preds = results["test_pred"]
                train_targets = results["train_targets"]
                test_ids = results["test_ids"]

                # Sanity Check: Targets should be identical across all experts
                if y_train_ref is None:
                    y_train_ref = train_targets
                else:
                    if not np.allclose(y_train_ref, train_targets):
                        raise ValueError(
                            f"Target mismatch detected for {backbone}-{expert}. "
                            "Ensure data ordering is deterministic."
                        )

                # Sanity Check: Test IDs should be identical
                if ids_test_ref is None:
                    ids_test_ref = test_ids
                else:
                    if not np.array_equal(ids_test_ref, test_ids):
                        raise ValueError(
                            f"Test ID mismatch detected for {backbone}-{expert}."
                        )

                # Reshape for concatenation (N, 1)
                meta_train_list.append(oof_preds.reshape(-1, 1))
                meta_test_list.append(test_preds.reshape(-1, 1))

        # Concatenate to form (N_samples, N_experts) matrices
        X_meta_train = np.hstack(meta_train_list)
        X_meta_test = np.hstack(meta_test_list)

        return X_meta_train, y_train_ref, X_meta_test, ids_test_ref

    def get_meta_features(self, load_cached_data=True):
        """
        Loads meta-features from cache or computes them from Level-0 experts.
        """
        # 1. Try Loading Cache
        if load_cached_data:
            files_exist = (
                os.path.exists(self.path_X_meta_train)
                and os.path.exists(self.path_y_meta_train)
                and os.path.exists(self.path_X_meta_test)
                and os.path.exists(self.path_ids_test)
            )
            if files_exist:
                # Cite debug_lesson_5: Validate Cached Artifacts
                X_train_cached = load_array(self.path_X_meta_train)

                if Config.DEBUG:
                    expected_len = Config.DEBUG_SAMPLE_SIZE * 2
                else:
                    expected_len = len(pd.read_csv(Config.TRAIN_METADATA_PATH)) + len(
                        pd.read_csv(Config.VAL_METADATA_PATH)
                    )

                if len(X_train_cached) == expected_len:
                    print("Loading cached Meta-Features...")
                    return (
                        X_train_cached,
                        load_array(self.path_y_meta_train),
                        load_array(self.path_X_meta_test),
                        load_array(self.path_ids_test),
                    )
                else:
                    print(
                        f"Meta-Feature Cache mismatch! Found {len(X_train_cached)} samples, expected {expected_len}. Recomputing..."
                    )
                    load_cached_data = False

        # 2. Compute from Scratch (collecting L0 outputs)
        # Pass updated load_cached_data to force re-computation of L0 experts if needed
        X_train, y_train, X_test, ids_test = self._collect_level0_data(
            load_cached_experts=load_cached_data
        )

        # 3. Save to Cache
        save_array(self.path_X_meta_train, X_train)
        save_array(self.path_y_meta_train, y_train)
        save_array(self.path_X_meta_test, X_test)
        save_array(self.path_ids_test, ids_test)

        return X_train, y_train, X_test, ids_test

    def run(self, load_cached_data=True):
        """
        Main execution method:
        1. Get Meta Features.
        2. Perform Nested CV to evaluate Meta-Learner.
        3. Train on full data and predict on Test.
        4. Generate Submission file.
        """
        seed_everything(self.seed)

        # 1. Prepare Data
        X_train, y_train, X_test, ids_test = self.get_meta_features(
            load_cached_data=load_cached_data
        )

        print(f"Meta-Feature Shape: {X_train.shape}")

        # 2. Nested Cross-Validation for Evaluation
        # We perform CV on the meta-features (which are themselves OOFs)
        # to get a robust estimate of the ensemble performance.
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        meta_oof_preds = np.zeros(len(y_train))

        print("Training Level-1 Meta-Learner (Bayesian Ridge)...")

        for fold, (t_idx, v_idx) in enumerate(kf.split(X_train, y_train)):
            X_t, y_t = X_train[t_idx], y_train[t_idx]
            X_v, y_v = X_train[v_idx], y_train[v_idx]

            # Bayesian Ridge is robust to collinearity (common in stacking)
            model = BayesianRidge(max_iter=Config.META_N_ITER, tol=Config.META_TOL)
            model.fit(X_t, y_t)

            val_pred = model.predict(X_v)
            meta_oof_preds[v_idx] = val_pred

            fold_rmse = np.sqrt(mean_squared_error(y_v, val_pred))
            print(f"  [Meta-Learner] Fold {fold+1} RMSE: {fold_rmse}")

        # Calculate Final CV Score
        final_rmse = np.sqrt(mean_squared_error(y_train, meta_oof_preds))
        print_metric("Final Ensemble CV RMSE", final_rmse)

        # 3. Final Training & Prediction
        # Retrain on all available training data
        final_model = BayesianRidge(max_iter=Config.META_N_ITER, tol=Config.META_TOL)
        final_model.fit(X_train, y_train)

        # Predict on Test Set
        final_test_preds = final_model.predict(X_test)

        # 4. Generate Submission
        submission_df = pd.DataFrame({"Id": ids_test, "Pawpularity": final_test_preds})

        # Ensure output directory exists
        ensure_dir(Config.SUBMISSION_PATH)

        # Save submission
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())

        return final_rmse
