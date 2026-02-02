import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from library.config import WORKING_DIR, FLOAT_PRECISION, RANDOM_SEED
from library.utils import clipped_log_loss

# =============================================================================
# Helper Functions
# =============================================================================


def _extract_and_freeze_params(fitted_pipeline):
    """
    Extracts learned hyperparameters (e.g., best C from LogRegCV) and returns
    a dictionary of parameters and a 'frozen' pipeline definition where
    CV estimators are replaced by fixed estimators.
    """
    params = {}
    new_steps = []

    for name, estimator in fitted_pipeline.steps:
        if isinstance(estimator, LogisticRegressionCV):
            # Extract best C
            # C_ is shape (1, n_classes) or (1,) depending on config.
            # With multi_class='multinomial', it's usually (1,).
            # We take the first element.
            if hasattr(estimator, "C_"):
                best_C = estimator.C_[0]
                params["C"] = best_C

                # Create fixed estimator
                fixed_est = LogisticRegression(
                    C=best_C,
                    solver=estimator.solver,
                    multi_class=estimator.multi_class,
                    max_iter=estimator.max_iter,
                    n_jobs=estimator.n_jobs,
                    random_state=RANDOM_SEED,
                )
                new_steps.append((name, fixed_est))
            else:
                # Fallback if fit failed or unexpected state
                new_steps.append((name, clone(estimator)))
        else:
            # For other estimators (Transformers, LDA), we use a fresh clone
            # because their parameters (like covariance matrix) should be
            # re-learned on the full dataset.
            new_steps.append((name, clone(estimator)))

    frozen_pipeline = Pipeline(new_steps)
    return params, frozen_pipeline


def _apply_frozen_params(original_pipeline, params):
    """
    Reconstructs a frozen pipeline from the original configuration and
    a dictionary of saved parameters.
    """
    new_steps = []
    for name, estimator in original_pipeline.steps:
        if isinstance(estimator, LogisticRegressionCV):
            if "C" in params:
                fixed_est = LogisticRegression(
                    C=params["C"],
                    solver=estimator.solver,
                    multi_class=estimator.multi_class,
                    max_iter=estimator.max_iter,
                    n_jobs=estimator.n_jobs,
                    random_state=RANDOM_SEED,
                )
                new_steps.append((name, fixed_est))
            else:
                # Fallback if param missing
                new_steps.append((name, clone(estimator)))
        else:
            new_steps.append((name, clone(estimator)))

    return Pipeline(new_steps)


# =============================================================================
# Main Training Engine
# =============================================================================


def train_and_predict_experts(
    X_train_global,
    X_train_morph,
    y_train,
    X_val_global,
    X_val_morph,
    expert_configs,
    load_cached_preds=True,
):
    """
    Phase 1: Trains all experts in the library on the training set,
    generates validation predictions, and extracts optimal hyperparameters.

    Args:
        X_train_global, X_train_morph: Training features.
        y_train: Training labels.
        X_val_global, X_val_morph: Validation features.
        expert_configs: List of expert dictionaries from topologies.py.
        load_cached_preds: Boolean to enable caching.

    Returns:
        results: Dict mapping expert_id to {
            'val_preds': np.ndarray,
            'frozen_pipeline': sklearn.pipeline.Pipeline (unfitted, fixed params)
        }
    """
    results = {}

    print(f"Starting Phase 1: Training {len(expert_configs)} experts...")

    for config in expert_configs:
        expert_id = config["id"]
        view_name = config["view"]
        original_pipeline = config["pipeline"]

        # Select Data Views
        if view_name == "global_view":
            X_tr = X_train_global
            X_val = X_val_global
        elif view_name == "morph_view":
            X_tr = X_train_morph
            X_val = X_val_morph
        else:
            raise ValueError(f"Unknown view: {view_name}")

        # Define Cache Paths
        cache_preds_path = os.path.join(WORKING_DIR, f"val_preds_{expert_id}.npy")
        cache_params_path = os.path.join(WORKING_DIR, f"params_{expert_id}.npy")

        # 1. Try Loading from Cache
        cache_hit = False
        if (
            load_cached_preds
            and os.path.exists(cache_preds_path)
            and os.path.exists(cache_params_path)
        ):
            try:
                val_preds = np.load(cache_preds_path)
                params = np.load(cache_params_path, allow_pickle=True).item()

                # Reconstruct frozen pipeline
                frozen_pipeline = _apply_frozen_params(original_pipeline, params)

                results[expert_id] = {
                    "val_preds": val_preds,
                    "frozen_pipeline": frozen_pipeline,
                }
                # print(f"[{expert_id}] Loaded from cache.")
                cache_hit = True
            except Exception as e:
                print(f"[{expert_id}] Cache load failed ({e}). Retraining...")
                cache_hit = False

        # 2. Train if Cache Miss
        if not cache_hit:
            # print(f"[{expert_id}] Training...")

            # Clone pipeline to ensure fresh start
            model = clone(original_pipeline)

            # Fit
            model.fit(X_tr, y_train)

            # Predict
            val_preds = model.predict_proba(X_val).astype(FLOAT_PRECISION)

            # Extract Params and Freeze
            params, frozen_pipeline = _extract_and_freeze_params(model)

            # Save to Cache
            try:
                np.save(cache_preds_path, val_preds)
                np.save(cache_params_path, params)
            except Exception as e:
                print(f"Warning: Could not save cache for {expert_id}: {e}")

            results[expert_id] = {
                "val_preds": val_preds,
                "frozen_pipeline": frozen_pipeline,
            }

    return results


def retrain_final_ensemble(
    X_full_global,
    X_full_morph,
    y_full,
    selected_experts_info,
    X_test_global,
    X_test_morph,
):
    """
    Phase 2: Retrains the selected experts on the full dataset (Train + Val)
    using the frozen hyperparameters identified in Phase 1.

    Args:
        X_full_global, X_full_morph: Combined features.
        y_full: Combined labels.
        selected_experts_info: List of dicts containing:
            {'id': str, 'weight': int, 'frozen_pipeline': Pipeline, 'view': str}
        X_test_global, X_test_morph: Test features.

    Returns:
        test_predictions: Dict mapping expert_id to np.ndarray (test probabilities).
    """
    test_predictions = {}

    print(
        f"Starting Phase 2: Retraining {len(selected_experts_info)} selected experts on full data..."
    )

    for expert_info in selected_experts_info:
        expert_id = expert_info["id"]
        view_name = expert_info["view"]
        frozen_pipeline = expert_info["frozen_pipeline"]

        # Select Data Views
        if view_name == "global_view":
            X_full = X_full_global
            X_test = X_test_global
        elif view_name == "morph_view":
            X_full = X_full_morph
            X_test = X_test_morph
        else:
            raise ValueError(f"Unknown view: {view_name}")

        # Clone the frozen pipeline (it has fixed C, but needs to re-fit PCA/LDA/Coefs)
        model = clone(frozen_pipeline)

        # Fit on Full Data
        # print(f"[{expert_id}] Retraining...")
        model.fit(X_full, y_full)

        # Predict on Test
        test_preds = model.predict_proba(X_test).astype(FLOAT_PRECISION)

        test_predictions[expert_id] = test_preds

    return test_predictions
