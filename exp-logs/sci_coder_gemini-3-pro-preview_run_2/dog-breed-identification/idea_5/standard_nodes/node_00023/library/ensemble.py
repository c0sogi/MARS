import os
import joblib
import numpy as np
import pandas as pd
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

    Args:
        model_a: Trained model for Stream A.
        model_b: Trained model for Stream B.
        weight_a (float): Optimal weight for Stream A.
        debug (bool): Debug flag.
        load_cached_data (bool): Cache flag.
    """
    print("Generating ensemble submission...")

    # 1. Extract Test Features (Stream A)
    X_test_a, _, ids = feature_engine.extract_features(
        dataset_key="test",
        model_name=config.MODEL_A_NAME,
        weights_name=config.MODEL_A_WEIGHTS,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # 2. Extract Test Features (Stream B)
    X_test_b, _, _ = feature_engine.extract_features(
        dataset_key="test",
        model_name=config.MODEL_B_NAME,
        weights_name=config.MODEL_B_WEIGHTS,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # 3. Predict
    probs_a_raw = model_a.predict_proba(X_test_a)
    probs_b_raw = model_b.predict_proba(X_test_b)

    # Pad probabilities if necessary (Cite debug_lesson_3)
    def pad_probs(probs, model):
        if probs.shape[1] < config.NUM_CLASSES:
            new_probs = np.zeros(
                (probs.shape[0], config.NUM_CLASSES), dtype=probs.dtype
            )
            new_probs[:, model.classes_.astype(int)] = probs
            return new_probs
        return probs

    probs_a = pad_probs(probs_a_raw, model_a)
    probs_b = pad_probs(probs_b_raw, model_b)

    # 4. Ensemble
    weight_b = 1.0 - weight_a
    probs_ensemble = (weight_a * probs_a) + (weight_b * probs_b)

    # 5. Create Submission DataFrame
    breeds = lib_classifier.get_breed_list()
    df_sub = pd.DataFrame(probs_ensemble, columns=breeds)
    df_sub.insert(0, "id", ids)

    # 6. Save
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
