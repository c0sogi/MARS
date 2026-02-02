import os
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegressionCV
from library.config import Config
from library.utils import set_seed, calculate_metric
from library.data_loader import get_class_mapping


def train_logistic_head(
    X: np.ndarray,
    y: np.ndarray,
    stream_name: str,
    load_cached_model: bool = True,
    max_iter: int = 2000,
):
    """
    Trains a LogisticRegressionCV model on the provided features.
    Caches the trained model to disk to avoid re-training.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target labels of shape (n_samples,).
        stream_name (str): Name of the stream (used for filename).
        load_cached_model (bool): Whether to load a cached model if it exists.
        max_iter (int): Maximum number of iterations for the solver.

    Returns:
        model: The trained sklearn LogisticRegressionCV model.
    """
    set_seed()

    # Define model cache path
    model_filename = f"{stream_name}_head.joblib"
    model_path = os.path.join(Config.WORKING_DIR, model_filename)

    # Try to load from cache
    if load_cached_model and os.path.exists(model_path):
        print(f"Loading cached model for {stream_name} from {model_path}...")
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            print(f"Failed to load cached model: {e}. Retraining...")

    print(f"Training Logistic Regression head for {stream_name}...")
    print(f"  Input shape: {X.shape}")

    # Initialize LogisticRegressionCV
    # Cs=10: Try 10 values on a log scale between 1e-4 and 1e4
    # cv=5: 5-fold Stratified Cross-Validation
    model = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty="l2",
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=max_iter,
        random_state=Config.SEED,
        n_jobs=Config.NUM_WORKERS,
        verbose=0,
    )

    # Fit the model
    model.fit(X, y)

    # Save the model
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)

    return model


def predict_stream(model, X: np.ndarray):
    """
    Generates probability predictions for a stream.

    Args:
        model: Trained sklearn model.
        X (np.ndarray): Feature matrix.

    Returns:
        np.ndarray: Probability matrix of shape (n_samples, n_classes).
    """
    # Ensure input is 2D
    if len(X.shape) == 1:
        X = X.reshape(1, -1)

    return model.predict_proba(X)


def optimize_ensemble_weights(
    probs_a: np.ndarray, probs_b: np.ndarray, y_true: np.ndarray
):
    """
    Finds the optimal weights for blending two probability matrices
    to minimize Log Loss on the validation set.

    Args:
        probs_a (np.ndarray): Probabilities from Stream A (N, C).
        probs_b (np.ndarray): Probabilities from Stream B (N, C).
        y_true (np.ndarray): True class indices (N,).

    Returns:
        list: Optimal weights [w_a, w_b].
    """
    print("Optimizing ensemble weights...")

    # Objective function: Log Loss of weighted average
    def objective(weights):
        w_a, w_b = weights
        # Normalize weights to ensure they sum to 1 (soft constraint handling)
        # But we will use constraints in minimize() to enforce this strictly.
        p_final = w_a * probs_a + w_b * probs_b

        # Clip probabilities to avoid numerical instability
        p_final = np.clip(p_final, 1e-15, 1 - 1e-15)

        # Normalize rows to sum to 1
        p_final = p_final / p_final.sum(axis=1, keepdims=True)

        return calculate_metric(y_true, p_final)

    # Constraints: sum(weights) = 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Bounds: 0 <= w <= 1
    bounds = ((0, 1), (0, 1))

    # Initial guess: Equal weights
    init_guess = [0.5, 0.5]

    # Optimization
    result = minimize(
        objective,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": False},
    )

    best_weights = result.x
    final_score = result.fun

    print(
        f"  Optimal Weights: Stream A={best_weights[0]:.4f}, Stream B={best_weights[1]:.4f}"
    )
    print(f"  Best Validation Log Loss: {final_score}")

    return best_weights


def generate_submission(
    test_ids: np.ndarray,
    final_probs: np.ndarray,
    output_path: str = Config.SUBMISSION_PATH,
):
    """
    Generates the submission CSV file.

    Args:
        test_ids (np.ndarray): Array of test image IDs.
        final_probs (np.ndarray): Matrix of predicted probabilities (N, C).
        output_path (str): Path to save the CSV.
    """
    print(f"Generating submission file at {output_path}...")

    # Get class names to use as header columns
    classes, _ = get_class_mapping()

    # Validate shapes
    if final_probs.shape[1] != len(classes):
        raise ValueError(
            f"Probability shape {final_probs.shape} does not match number of classes {len(classes)}"
        )

    # Create DataFrame
    df = pd.DataFrame(final_probs, columns=classes)

    # Insert ID column at the beginning
    df.insert(0, "id", test_ids)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print("Submission saved successfully.")
