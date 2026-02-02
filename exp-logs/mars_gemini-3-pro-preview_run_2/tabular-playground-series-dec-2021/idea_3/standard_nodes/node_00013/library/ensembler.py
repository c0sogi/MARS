import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from scipy.optimize import minimize
from library.config import ID_COL, TARGET_COL


def optimize_weights(oof_preds_dict, y_true, classes):
    """
    Finds the optimal weights for blending OOF predictions to minimize Log Loss.

    Args:
        oof_preds_dict (dict): Dictionary {model_name: oof_probability_array}.
        y_true (pd.Series or np.ndarray): True target labels (original class values).
        classes (np.ndarray): Array of unique class labels corresponding to probability columns.

    Returns:
        dict: Dictionary {model_name: optimal_weight}.
    """
    model_names = list(oof_preds_dict.keys())
    predictions = [oof_preds_dict[name] for name in model_names]
    n_models = len(predictions)

    # Map original class labels to 0-indexed integers for log_loss calculation
    # The probability columns correspond to the sorted unique classes
    class_map = {c: i for i, c in enumerate(classes)}

    # Efficiently map targets to indices
    if isinstance(y_true, pd.Series):
        y_true_indices = y_true.map(class_map).values
    else:
        y_true_indices = np.array([class_map[y] for y in y_true])

    # Objective function: Minimize Multi-class Log Loss
    def loss_func(weights):
        # Normalize weights to sum to 1
        weights = np.array(weights)
        if np.sum(weights) == 0:
            return 1e9
        weights = weights / np.sum(weights)

        # Calculate weighted average of predictions
        final_preds = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            final_preds += weights[i] * pred

        # Clip probabilities to prevent log(0) errors
        final_preds = np.clip(final_preds, 1e-15, 1 - 1e-15)

        return log_loss(y_true_indices, final_preds)

    # Initial guess: Equal weights
    init_weights = [1.0 / n_models] * n_models

    # Constraints: Sum of weights must equal 1
    constraints = {"type": "eq", "fun": lambda w: 1 - np.sum(w)}

    # Bounds: Weights must be between 0 and 1
    bounds = [(0.0, 1.0)] * n_models

    # Run Optimization
    result = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=1e-6,
    )

    # Normalize final weights
    best_weights = result.x / np.sum(result.x)
    weights_dict = dict(zip(model_names, best_weights))

    # --- Calculate and Print Final Ensemble Metrics ---
    final_oof_probs = np.zeros_like(predictions[0])
    for name, w in weights_dict.items():
        final_oof_probs += w * oof_preds_dict[name]

    # Convert probabilities to class labels for accuracy check
    final_pred_indices = np.argmax(final_oof_probs, axis=1)
    final_pred_labels = classes[final_pred_indices]

    acc = accuracy_score(y_true, final_pred_labels)
    ll = log_loss(y_true_indices, final_oof_probs)

    print("-" * 30)
    print("Ensemble Optimization Results")
    print("-" * 30)
    print(f"Optimized Weights: {weights_dict}")
    print(f"Ensemble OOF LogLoss: {ll}")
    print(f"Ensemble OOF Accuracy: {acc}")

    return weights_dict


def blend_predictions(test_preds_dict, weights_dict):
    """
    Blends test predictions using the provided weights.

    Args:
        test_preds_dict (dict): Dictionary {model_name: test_probability_array}.
        weights_dict (dict): Dictionary {model_name: weight}.

    Returns:
        np.ndarray: Blended probability array.
    """
    model_names = list(test_preds_dict.keys())

    # Initialize with zeros based on the shape of the first model's predictions
    first_preds = test_preds_dict[model_names[0]]
    blended_preds = np.zeros_like(first_preds)

    for name, weight in weights_dict.items():
        if name in test_preds_dict:
            blended_preds += weight * test_preds_dict[name]
        else:
            raise ValueError(
                f"Model '{name}' found in weights but not in test predictions."
            )

    return blended_preds


def generate_submission(test_ids, blended_probs, classes, output_path):
    """
    Generates the submission CSV file from blended probabilities.

    Args:
        test_ids (pd.Series or np.ndarray): IDs for the test set.
        blended_probs (np.ndarray): Blended probability predictions.
        classes (np.ndarray): Array of class labels corresponding to probability columns.
        output_path (str): Path to save the CSV.
    """
    # Convert probabilities to class labels (argmax)
    pred_indices = np.argmax(blended_probs, axis=1)
    pred_labels = classes[pred_indices]

    # Create DataFrame
    submission_df = pd.DataFrame({ID_COL: test_ids, TARGET_COL: pred_labels})

    # Ensure ID is integer type
    submission_df[ID_COL] = submission_df[ID_COL].astype(int)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")
    print(f"Submission Head:\n{submission_df.head()}")
