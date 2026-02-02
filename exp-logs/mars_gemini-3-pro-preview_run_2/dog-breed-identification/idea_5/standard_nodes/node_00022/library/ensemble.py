import os
import joblib
import numpy as np
from sklearn.metrics import log_loss
import library.config as config
import library.feature_engine as feature_engine
import library.classifier as lib_classifier


def load_models():
    """
    Loads the trained Logistic Regression models for Stream A and Stream B from disk.

    Returns:
        tuple: (model_a, model_b)
    """
    path_a = os.path.join(config.WORKING_DIR, f"{config.MODEL_A_NAME}_logreg.joblib")
    path_b = os.path.join(config.WORKING_DIR, f"{config.MODEL_B_NAME}_logreg.joblib")

    if not os.path.exists(path_a):
        raise FileNotFoundError(f"Model A not found at {path_a}")
    if not os.path.exists(path_b):
        raise FileNotFoundError(f"Model B not found at {path_b}")

    print(f"Loading Model A from {path_a}...")
    model_a = joblib.load(path_a)

    print(f"Loading Model B from {path_b}...")
    model_b = joblib.load(path_b)

    return model_a, model_b


def get_validation_predictions(model_a, model_b, debug=False, load_cached_data=True):
    """
    Generates probability predictions for the validation set using both models.
    Used for weight optimization.

    Args:
        model_a: Trained model for Stream A.
        model_b: Trained model for Stream B.
        debug (bool): If True, uses debug dataset size.
        load_cached_data (bool): If True, loads features from cache.

    Returns:
        tuple: (probs_a, probs_b, y_true)
    """
    print("Generating validation predictions for ensemble optimization...")

    # Stream A Features
    X_val_a, y_val_a, _ = feature_engine.extract_features(
        dataset_key="val",
        model_name=config.MODEL_A_NAME,
        weights_name=config.MODEL_A_WEIGHTS,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # Stream B Features
    X_val_b, y_val_b, _ = feature_engine.extract_features(
        dataset_key="val",
        model_name=config.MODEL_B_NAME,
        weights_name=config.MODEL_B_WEIGHTS,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # Verify label consistency
    if not np.array_equal(y_val_a, y_val_b):
        raise ValueError("Validation labels mismatch between Stream A and Stream B.")

    # Predict
    probs_a = model_a.predict_proba(X_val_a)
    probs_b = model_b.predict_proba(X_val_b)

    return probs_a, probs_b, y_val_a


def optimize_weights(probs_a, probs_b, y_true):
    """
    Finds the optimal weight w for the ensemble P = w * P_a + (1-w) * P_b
    by minimizing log loss on the validation set.

    Args:
        probs_a (np.ndarray): Probabilities from Stream A.
        probs_b (np.ndarray): Probabilities from Stream B.
        y_true (np.ndarray): True labels.

    Returns:
        float: Best weight for Stream A.
    """
    print("Optimizing ensemble weights...")

    best_loss = float("inf")
    best_w = 0.5

    # Grid search from 0.0 to 1.0
    weights = np.linspace(0, 1, 101)

    for w in weights:
        probs_ensemble = w * probs_a + (1 - w) * probs_b
        loss = log_loss(y_true, probs_ensemble)

        if loss < best_loss:
            best_loss = loss
            best_w = w

    print(f"  Best Weight (Stream A): {best_w}")
    print(f"  Best Weight (Stream B): {1 - best_w}")
    # Printing full precision as requested
    print(f"  Best Ensemble Validation Log Loss: {best_loss:.16f}")

    return best_w


def generate_submission(model_a, model_b, weight_a, debug=False, load_cached_data=True):
    """
    Generates the submission file using the optimized weights.
    Wraps the library function to ensure consistent pipeline usage.

    Args:
        model_a: Trained model for Stream A.
        model_b: Trained model for Stream B.
        weight_a (float): Optimal weight for Stream A.
        debug (bool): Debug flag.
        load_cached_data (bool): Cache flag.
    """
    # Delegate to library function which handles feature extraction,
    # ID alignment, and CSV formatting robustly.
    lib_classifier.generate_submission(
        model_a=model_a,
        model_b=model_b,
        weight_a=weight_a,
        debug=debug,
        load_cached_data=load_cached_data,
    )
