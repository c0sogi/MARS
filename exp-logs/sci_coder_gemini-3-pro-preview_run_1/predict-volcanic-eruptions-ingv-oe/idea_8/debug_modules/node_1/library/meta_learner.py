import os
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from library import config, utils

# Ensure reproducibility
utils.seed_everything(config.SEED)


def train_meta_learner(tabular_oof, vision_oof, y_true):
    """
    Trains a Ridge Regression meta-learner on the Out-Of-Fold predictions
    from the Tabular (LightGBM) and Vision (CNN) branches.

    Args:
        tabular_oof (np.array): OOF predictions from the tabular model.
        vision_oof (np.array): OOF predictions from the vision model.
        y_true (np.array): Ground truth target values.

    Returns:
        model (sklearn.linear_model.Ridge): The trained meta-learner.
        score (float): The MAE score of the meta-learner on the OOF set.
    """
    print("Training Meta-Learner (Ridge Regression)...")

    # 1. Prepare Meta-Features
    # Stack the predictions to form the input matrix (N_samples, 2)
    # Column 0: Tabular, Column 1: Vision
    X_meta = np.column_stack((tabular_oof, vision_oof))

    # 2. Initialize Model
    # Use parameters from config (alpha, fit_intercept, random_state)
    model = Ridge(**config.RIDGE_PARAMS)

    # 3. Train
    model.fit(X_meta, y_true)

    # 4. Evaluate on OOF (Training set for the meta-learner)
    # This gives us an estimate of the ensemble performance
    meta_preds = model.predict(X_meta)

    # Enforce physical constraint (time >= 0)
    meta_preds = np.maximum(0, meta_preds)

    score = utils.mae_score(y_true, meta_preds)
    utils.print_metric("Meta-Learner OOF MAE", score)

    # Log coefficients to understand the contribution of each branch
    # This helps verify if the ensemble is effectively utilizing both modalities
    print(
        f"Meta-Learner Coefficients: Tabular={model.coef_[0]:.4f}, "
        f"Vision={model.coef_[1]:.4f}, Intercept={model.intercept_:.4f}"
    )

    # 5. Save Model
    # Ensure directory exists
    os.makedirs(os.path.dirname(config.META_MODEL_PATH), exist_ok=True)
    joblib.dump(model, config.META_MODEL_PATH)
    print(f"Meta-learner saved to {config.META_MODEL_PATH}")

    return model, score


def predict_meta_learner(tabular_preds, vision_preds):
    """
    Generates final predictions using the trained meta-learner.

    Args:
        tabular_preds (np.array): Test predictions from the tabular model.
        vision_preds (np.array): Test predictions from the vision model.

    Returns:
        final_preds (np.array): The weighted ensemble predictions.
    """
    # 1. Load Model
    if not os.path.exists(config.META_MODEL_PATH):
        raise FileNotFoundError(
            f"Meta-learner model not found at {config.META_MODEL_PATH}. "
            "Please run training before prediction."
        )

    model = joblib.load(config.META_MODEL_PATH)

    # 2. Prepare Meta-Features
    # Must match the order used in training: [Tabular, Vision]
    X_test_meta = np.column_stack((tabular_preds, vision_preds))

    # 3. Predict
    final_preds = model.predict(X_test_meta)

    # 4. Post-processing
    # Enforce physical constraint: time_to_eruption cannot be negative
    final_preds = np.maximum(0, final_preds)

    return final_preds
