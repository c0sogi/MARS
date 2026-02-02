import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar

from library.config import Config
from library.utils import seed_everything, save_submission
from library.data import get_class_names


def train_classifier(
    X_train, y_train, X_val, y_val, stream_name, load_cached_data=True
):
    """
    Trains a LogisticRegressionCV classifier for a specific stream.
    Handles caching of the trained model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        stream_name (str): 'stream_a' or 'stream_b'.
        load_cached_data (bool): Whether to load a cached model if available.

    Returns:
        model: The trained (or loaded) sklearn model.
        val_probs: Predicted probabilities for the validation set.
    """
    seed_everything()

    # Determine model path based on stream
    if stream_name == "stream_a":
        model_path = Config.STREAM_A_MODEL
    elif stream_name == "stream_b":
        model_path = Config.STREAM_B_MODEL
    else:
        raise ValueError(f"Unknown stream name: {stream_name}")

    # Check cache
    if load_cached_data and os.path.exists(model_path):
        print(f"Loading cached model for {stream_name} from {model_path}...")
        model = joblib.load(model_path)
    else:
        print(f"Training classifier for {stream_name}...")
        # Initialize LogisticRegressionCV with Config params
        # We pass the validation set implicitly via CV inside the fit if we wanted to,
        # but LogisticRegressionCV does its own internal CV on X_train.
        # We use the external X_val just for final reporting here.
        model = LogisticRegressionCV(
            Cs=Config.LOGREG_PARAMS["Cs"],
            cv=Config.LOGREG_PARAMS["cv"],
            max_iter=Config.LOGREG_PARAMS["max_iter"],
            solver=Config.LOGREG_PARAMS["solver"],
            multi_class=Config.LOGREG_PARAMS["multi_class"],
            n_jobs=Config.LOGREG_PARAMS["n_jobs"],
            random_state=Config.LOGREG_PARAMS["random_state"],
            verbose=0,  # Keep it silent as requested
        )

        model.fit(X_train, y_train)

        # Save model
        print(f"Saving model for {stream_name} to {model_path}...")
        joblib.dump(model, model_path)

    # Evaluate on Validation Set
    print(f"Evaluating {stream_name} on validation set...")
    val_probs = model.predict_proba(X_val)
    loss = log_loss(y_val, val_probs)
    print(f"{stream_name} Validation Log Loss: {loss}")

    return model, val_probs


def optimize_ensemble_weights(probs_a, probs_b, y_true):
    """
    Finds the optimal weight w to combine predictions: P = w * A + (1-w) * B.
    Minimizes Log Loss on the provided data (validation set).

    Args:
        probs_a (np.ndarray): Probabilities from Stream A.
        probs_b (np.ndarray): Probabilities from Stream B.
        y_true (np.ndarray): True labels.

    Returns:
        tuple: (weight_a, weight_b)
    """
    print("Optimizing ensemble weights...")

    def objective(w):
        # Constrain w to [0, 1] effectively
        w = np.clip(w, 0, 1)
        probs_ensemble = w * probs_a + (1 - w) * probs_b
        # Clip probabilities to avoid log(0) issues, though log_loss handles it usually
        probs_ensemble = np.clip(probs_ensemble, 1e-15, 1 - 1e-15)
        # Normalize just in case
        probs_ensemble /= probs_ensemble.sum(axis=1, keepdims=True)
        return log_loss(y_true, probs_ensemble)

    # Minimize scalar function
    result = minimize_scalar(objective, bounds=(0, 1), method="bounded")

    best_w_a = result.x
    best_w_b = 1.0 - best_w_a
    best_loss = result.fun

    print(f"Optimal Ensemble Weights -> Stream A: {best_w_a}, Stream B: {best_w_b}")
    print(f"Optimized Ensemble Validation Log Loss: {best_loss}")

    return best_w_a, best_w_b


def run_training(data_stream_a, data_stream_b, load_cached_models=True):
    """
    Orchestrates the training, optimization, and submission generation process.

    Args:
        data_stream_a (dict): Data dictionary for Stream A (train/val/test).
        data_stream_b (dict): Data dictionary for Stream B (train/val/test).
        load_cached_models (bool): Whether to load cached models.
    """
    seed_everything()

    # Unpack Data Stream A
    X_train_a, y_train_a = data_stream_a["train"]
    X_val_a, y_val_a = data_stream_a["val"]
    X_test_a, test_ids_a = data_stream_a["test"]

    # Unpack Data Stream B
    X_train_b, y_train_b = data_stream_b["train"]
    X_val_b, y_val_b = data_stream_b["val"]
    X_test_b, test_ids_b = data_stream_b["test"]

    # Sanity Check: Ensure labels and IDs match
    assert np.array_equal(
        y_train_a, y_train_b
    ), "Mismatch in training labels between streams"
    assert np.array_equal(
        y_val_a, y_val_b
    ), "Mismatch in validation labels between streams"
    assert np.array_equal(
        test_ids_a, test_ids_b
    ), "Mismatch in test IDs between streams"

    # 1. Train/Load Stream A
    model_a, val_probs_a = train_classifier(
        X_train_a,
        y_train_a,
        X_val_a,
        y_val_a,
        stream_name="stream_a",
        load_cached_data=load_cached_models,
    )

    # 2. Train/Load Stream B
    model_b, val_probs_b = train_classifier(
        X_train_b,
        y_train_b,
        X_val_b,
        y_val_b,
        stream_name="stream_b",
        load_cached_data=load_cached_models,
    )

    # 3. Optimize Ensemble
    w_a, w_b = optimize_ensemble_weights(val_probs_a, val_probs_b, y_val_a)

    # 4. Generate Test Predictions
    print("Generating test predictions...")
    test_probs_a = model_a.predict_proba(X_test_a)
    test_probs_b = model_b.predict_proba(X_test_b)

    # Weighted Late Fusion
    final_test_probs = (w_a * test_probs_a) + (w_b * test_probs_b)

    # Normalize (just to be safe, though linear combination of normalized probs is normalized)
    final_test_probs /= final_test_probs.sum(axis=1, keepdims=True)

    # 5. Save Submission
    class_names = get_class_names()
    save_submission(test_ids_a, final_test_probs, class_names)

    print("Training and submission generation complete.")
