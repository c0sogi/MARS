import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from library.config import PROB_CLIP_EPS, FLOAT_PRECISION


def calculate_log_loss(y_true_indices, y_pred_probs):
    """
    Calculates the Multi-class Log Loss strictly according to the task description.

    Steps:
    1. Rescale: Each row is divided by the row sum.
    2. Clip: Probabilities are clipped to [1e-15, 1-1e-15].
    3. Score: Negative log likelihood of the true class.

    Args:
        y_true_indices (np.ndarray): 1D array of integer class indices (0 to K-1).
        y_pred_probs (np.ndarray): 2D array of probabilities (N, K).

    Returns:
        float: The calculated log loss.
    """
    # Ensure float64 for precision
    y_pred = y_pred_probs.astype(FLOAT_PRECISION)

    # 1. Rescale
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle potential zero sums (though unlikely with proper models)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip
    y_pred_clipped = np.clip(y_pred_norm, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)

    # 3. Score
    n_samples = len(y_true_indices)
    # Extract probabilities corresponding to the true classes
    correct_class_probs = y_pred_clipped[np.arange(n_samples), y_true_indices]

    # Calculate negative log likelihood
    loss = -np.mean(np.log(correct_class_probs))

    return loss


def optimize_ensemble(predictions_dict, y_val, max_iter=100, verbose=True):
    """
    Performs Greedy Forward Selection to find the optimal ensemble weights.

    Args:
        predictions_dict (dict): Dictionary mapping expert names to prediction matrices (N, K).
                                 Assumes columns are aligned with sorted unique classes of y_val.
        y_val (np.ndarray): 1D array of true class labels (strings or ints).
        max_iter (int): Maximum number of iterations (experts to add).
        verbose (bool): Whether to print progress.

    Returns:
        dict: A dictionary mapping expert names to their integer weights (counts).
    """
    # Encode y_val to integers 0..K-1
    # We assume the columns of prediction matrices correspond to sorted(unique(y_val))
    le = LabelEncoder()
    y_val_enc = le.fit_transform(y_val)

    # Validation: Check if number of classes matches prediction columns
    first_expert = next(iter(predictions_dict.values()))
    if first_expert.shape[1] != len(le.classes_):
        # Fallback validation: This might happen if y_val is a subset of total classes.
        # In a robust pipeline, we assume alignment is handled by the model training.
        # We proceed assuming y_val_enc indices map correctly to columns.
        pass

    selected_experts = []
    current_sum_probs = None
    best_score = float("inf")

    # Greedy Loop
    for i in range(max_iter):
        iteration_best_score = float("inf")
        iteration_best_expert = None

        # Try adding each candidate expert to the current ensemble
        for name, probs in predictions_dict.items():
            # Calculate trial ensemble probabilities
            # Ensemble Prob = (Sum of existing + Candidate) / (Count + 1)

            if current_sum_probs is None:
                trial_sum = probs
                n_experts = 1
            else:
                trial_sum = current_sum_probs + probs
                n_experts = len(selected_experts) + 1

            trial_probs = trial_sum / n_experts

            # Evaluate
            score = calculate_log_loss(y_val_enc, trial_probs)

            if score < iteration_best_score:
                iteration_best_score = score
                iteration_best_expert = name

        # Check for improvement
        # We strictly require improvement to continue adding experts
        if iteration_best_score < best_score:
            best_score = iteration_best_score
            selected_experts.append(iteration_best_expert)

            # Update the running sum
            if current_sum_probs is None:
                current_sum_probs = predictions_dict[iteration_best_expert].copy()
            else:
                current_sum_probs += predictions_dict[iteration_best_expert]

            if verbose:
                print(
                    f"Ensemble Selection Iter {i+1}: Added '{iteration_best_expert}' | Val Log Loss: {best_score:.15f}"
                )
        else:
            if verbose:
                print(f"Ensemble Selection: No improvement at iter {i+1}. Stopping.")
            break

    # Convert list of selected experts to weights
    weights = dict(Counter(selected_experts))

    if verbose:
        print(f"Final Ensemble Weights: {weights}")

    return weights
