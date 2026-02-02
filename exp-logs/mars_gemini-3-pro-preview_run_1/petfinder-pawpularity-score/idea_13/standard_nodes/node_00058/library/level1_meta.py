import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from library.config import Config
from library.utils import (
    load_array,
    save_array,
    check_cache_exists,
    rmse_score,
    create_submission,
    set_seed,
)


class InteractionDesignMatrix:
    """
    Handles the construction of the Interaction-Aware feature matrix
    for the Level-1 Meta-Learner.
    """

    @staticmethod
    def construct(predictions: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        """
        Constructs the feature matrix containing:
        1. Raw Expert Predictions (P)
        2. Metadata Features (M)
        3. Interaction Terms (P * M)

        Args:
            predictions (np.ndarray): Shape (N, n_experts)
            metadata (np.ndarray): Shape (N, n_meta_features)

        Returns:
            np.ndarray: The constructed feature matrix of shape (N, n_features).
        """
        n_samples, n_experts = predictions.shape
        _, n_meta = metadata.shape

        # List to hold all feature components
        features = []

        # 1. Raw Predictions
        features.append(predictions)

        # 2. Metadata
        features.append(metadata)

        # 3. Interactions
        # We compute the element-wise product of every expert column with every metadata column.
        # This allows the model to learn conditional weights (e.g., "If Blur=1, reduce weight of ConvNeXt").
        interactions = []
        for i in range(n_experts):
            expert_col = predictions[:, i : i + 1]  # Keep dim (N, 1)
            # Broadcast multiply with all metadata columns
            # expert_col is (N, 1), metadata is (N, n_meta) -> result is (N, n_meta)
            inter = expert_col * metadata
            interactions.append(inter)

        features.append(np.hstack(interactions))

        # Concatenate all features horizontally
        X = np.hstack(features)
        return X


class Level1MetaLearner:
    """
    Implements the Level-1 Stacking logic with Interaction-Aware Meta-Learning.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.backbones = list(Config.BACKBONES.keys())
        # The order of models must match how they were generated in Level 0
        self.models = ["ridge", "svr", "et", "lgbm"]
        self.meta_features = Config.META_FEATURES

    def _load_level0_data(self):
        """
        Loads OOF predictions, Test predictions, Targets, and Metadata from the cache.
        Aggregates them into matrices P (Predictions) and M (Metadata).
        """
        print("Loading Level-0 Expert outputs...")

        # We need to load the reference targets and metadata.
        # These were saved by Level0Experts. Since the merge order is deterministic,
        # we can load from any backbone (e.g., the first one).
        ref_backbone = self.backbones[0]

        # Load Targets (y) and Metadata (M) for OOF (Train+Val)
        y_oof = load_array(f"{ref_backbone}_merged_targets.npy")
        meta_oof = load_array(f"{ref_backbone}_merged_meta.npy")

        # Load Metadata (M) and IDs for Test
        meta_test = load_array(f"{ref_backbone}_test_meta_ref.npy")
        ids_test = load_array(f"{ref_backbone}_test_ids_ref.npy")

        # Containers for expert predictions
        oof_preds_list = []
        test_preds_list = []
        feature_names = []

        # Iterate through all combinations of Backbone x Model
        for backbone in self.backbones:
            for model in self.models:
                # Construct filenames
                oof_file = f"{backbone}_{model}_oof.npy"
                test_file = f"{backbone}_{model}_test_pred.npy"

                # Load arrays
                p_oof = load_array(oof_file)
                p_test = load_array(test_file)

                oof_preds_list.append(p_oof.reshape(-1, 1))
                test_preds_list.append(p_test.reshape(-1, 1))
                feature_names.append(f"{backbone}_{model}")

        # Stack to form Prediction Matrices P
        P_oof = np.hstack(oof_preds_list)
        P_test = np.hstack(test_preds_list)

        if self.debug:
            print(f"[DEBUG] Slicing data for fast check.")
            limit = 200
            P_oof = P_oof[:limit]
            meta_oof = meta_oof[:limit]
            y_oof = y_oof[:limit]
            # Keep test as is or slice small
            P_test = P_test[:20]
            meta_test = meta_test[:20]
            ids_test = ids_test[:20]

        return {
            "P_oof": P_oof,
            "M_oof": meta_oof,
            "y_oof": y_oof,
            "P_test": P_test,
            "M_test": meta_test,
            "ids_test": ids_test,
        }

    def run(self, load_cached_data: bool = True):
        """
        Main execution pipeline for Level-1.

        Args:
            load_cached_data (bool): If True, attempts to load constructed design matrices from cache.
                                     Note: Level-0 outputs are always loaded from cache; this flag
                                     controls the caching of the Level-1 design matrix construction.
        """
        set_seed(Config.SEED)

        # Define cache filenames for Level-1 Design Matrices
        cache_files = {
            "X_oof": "level1_X_oof.npy",
            "y_oof": "level1_y_oof.npy",
            "X_test": "level1_X_test.npy",
            "ids_test": "level1_ids_test.npy",
        }

        # Check if we can load pre-constructed matrices
        all_cached = all(check_cache_exists(f) for f in cache_files.values())

        if load_cached_data and all_cached:
            print("Loading cached Level-1 Design Matrices...")
            X_oof = load_array(cache_files["X_oof"])
            y_oof = load_array(cache_files["y_oof"])
            X_test = load_array(cache_files["X_test"])
            ids_test = load_array(cache_files["ids_test"])
        else:
            # 1. Load Level-0 Data
            data = self._load_level0_data()
            P_oof, M_oof, y_oof = data["P_oof"], data["M_oof"], data["y_oof"]
            P_test, M_test, ids_test = data["P_test"], data["M_test"], data["ids_test"]

            print(f"Constructing Interaction-Aware Design Matrices...")
            print(f"Experts: {P_oof.shape[1]}, Meta Features: {M_oof.shape[1]}")

            # 2. Construct Interaction Matrices
            X_oof = InteractionDesignMatrix.construct(P_oof, M_oof)
            X_test = InteractionDesignMatrix.construct(P_test, M_test)

            print(f"Design Matrix Shape: {X_oof.shape}")

            # Cache the constructed matrices
            save_array(cache_files["X_oof"], X_oof)
            save_array(cache_files["y_oof"], y_oof)
            save_array(cache_files["X_test"], X_test)
            save_array(cache_files["ids_test"], ids_test)

        # 3. Nested Cross-Validation for Evaluation
        print("\nRunning Nested Cross-Validation on Meta-Learner...")
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

        fold_scores = []
        oof_meta_preds = np.zeros(len(y_oof))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_oof, y_oof)):
            X_train_fold, X_val_fold = X_oof[train_idx], X_oof[val_idx]
            y_train_fold, y_val_fold = y_oof[train_idx], y_oof[val_idx]

            # Bayesian Ridge handles regularization automatically
            model = BayesianRidge(**Config.META_MODEL_PARAMS)
            model.fit(X_train_fold, y_train_fold)

            preds = model.predict(X_val_fold)

            # Clip predictions to valid range [1, 100]
            preds = np.clip(preds, 1.0, 100.0)

            oof_meta_preds[val_idx] = preds
            score = rmse_score(y_val_fold, preds)
            fold_scores.append(score)

            print(f"Fold {fold_idx+1} RMSE: {score}")

        overall_cv_score = rmse_score(y_oof, oof_meta_preds)
        print(f"\nOverall CV RMSE: {overall_cv_score}")
        print(f"Average Fold RMSE: {np.mean(fold_scores)}")

        # 4. Final Training and Inference
        print("\nRetraining Meta-Learner on full OOF set for submission...")
        final_model = BayesianRidge(**Config.META_MODEL_PARAMS)
        final_model.fit(X_oof, y_oof)

        final_predictions = final_model.predict(X_test)

        # Clip predictions to valid range
        final_predictions = np.clip(final_predictions, 1.0, 100.0)

        # 5. Create Submission
        print(f"Generating submission file at {Config.SUBMISSION_PATH}...")
        create_submission(ids_test, final_predictions)
        print("Submission generation complete.")


def train_meta_learner(debug: bool = False, load_cached_data: bool = True):
    """
    Public interface to run the Level-1 Meta-Learner pipeline.

    Args:
        debug (bool): If True, runs on a subset of data for debugging.
        load_cached_data (bool): If True, uses cached design matrices if available.
    """
    learner = Level1MetaLearner(debug=debug)
    learner.run(load_cached_data=load_cached_data)
