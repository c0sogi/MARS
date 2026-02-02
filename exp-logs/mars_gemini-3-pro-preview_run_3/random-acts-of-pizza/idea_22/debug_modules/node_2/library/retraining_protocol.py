import numpy as np
import scipy.sparse
from library.config import Config


def _concat_views(v1, v2):
    """
    Helper function to concatenate two data view dictionaries.
    Handles both sparse matrices and numpy arrays.
    """
    combined = {}
    for k in v1.keys():
        if k == "y":
            continue
        if scipy.sparse.issparse(v1[k]):
            combined[k] = scipy.sparse.vstack([v1[k], v2[k]])
        else:
            combined[k] = np.concatenate([v1[k], v2[k]])
    return combined


def retrain_final_models(ensemble, data_train, data_val):
    """
    Retrains the Level-1 base learners of the HexEnsemble on the full dataset (Train + Val).

    This function implements the final phase of the Validation-Guided Retraining Protocol.
    It concatenates the training and validation sets to create a full training set.

    Specific Logic:
    - XGBoost ('semantic_booster'): Trains on the Full Set (Train + Val) but uses the
      explicit Validation Set as 'eval_set' to trigger Early Stopping. This prevents
      blind overfitting while allowing the model to see all data.
    - Other Models (RF, kNN, Linear): Train on the Full Set (Train + Val) directly.

    Args:
        ensemble (HexEnsemble): The ensemble instance with initialized base learners.
        data_train (dict): Training data dictionary from FeaturePipeline.
        data_val (dict): Validation data dictionary from FeaturePipeline.
    """
    print("\nPhase 3: Retraining Base Learners on Full Data (Train + Val)...")

    # Combine Train and Val for final training
    # Note: data_train['y'] is numpy array
    y_full = np.concatenate([data_train["y"], data_val["y"]])
    data_full = _concat_views(data_train, data_val)

    model_names = list(ensemble.base_learners.keys())

    for name in model_names:
        # Construct the specific input feature matrix for this base learner
        # We access the protected method _construct_input from the ensemble instance
        X_full = ensemble._construct_input(name, data_full)

        if name == "semantic_booster":
            # Logic for XGBoost: Use Validation set for Early Stopping
            # We construct the validation-only set for evaluation metrics
            X_val_only = ensemble._construct_input(name, data_val)
            y_val_only = data_val["y"]

            # Recalculate scale_pos_weight for the full dataset to handle class imbalance
            n_pos = np.sum(y_full)
            n_neg = len(y_full) - n_pos
            scale_weight = n_neg / n_pos if n_pos > 0 else 1.0
            ensemble.base_learners[name].set_params(scale_pos_weight=scale_weight)

            print(f"Retraining {name} with Early Stopping...")
            ensemble.base_learners[name].set_params(
                early_stopping_rounds=Config.XGB_EARLY_STOPPING_ROUNDS
            )
            ensemble.base_learners[name].fit(
                X_full,
                y_full,
                eval_set=[(X_val_only, y_val_only)],
                verbose=False,
            )
        else:
            # Logic for other models (Random Forest, kNN, Logistic Regression)
            print(f"Retraining {name} on full dataset...")
            ensemble.base_learners[name].fit(X_full, y_full)

    print("Retraining Complete.")
