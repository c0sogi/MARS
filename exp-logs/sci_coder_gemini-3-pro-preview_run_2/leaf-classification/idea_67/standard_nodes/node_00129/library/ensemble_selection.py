import numpy as np
from collections import Counter
from library.utils import log_loss_metric, clip_probabilities


def greedy_forward_selection(
    predictions_dict, y_true, max_iter=100, tol=1e-6, verbose=True
):
    """
    Performs Greedy Forward Selection (with replacement) to find the optimal ensemble of experts.

    This algorithm iteratively adds the expert that maximizes the improvement in the evaluation metric
    (Log Loss) to the ensemble. It allows for weighted ensembling by selecting the same expert multiple times.

    Args:
        predictions_dict (dict): Dictionary where keys are expert names and values are
                                 prediction matrices (numpy arrays of shape (N_samples, N_classes)).
                                 Predictions should be probabilities.
        y_true (np.ndarray): True labels (N_samples,).
        max_iter (int): Maximum number of iterations (ensemble size). Defaults to 100.
        tol (float): Tolerance for improvement. If improvement is less than this, the algorithm stops.
        verbose (bool): Whether to print progress and metrics.

    Returns:
        tuple: (weights, best_score)
            - weights (dict): A dictionary mapping expert names to their calculated weights (summing to 1.0).
            - best_score (float): The best log loss score achieved by the ensemble.
    """
    # 1. Validation
    if not predictions_dict:
        raise ValueError("predictions_dict cannot be empty.")

    expert_names = list(predictions_dict.keys())
    n_samples = len(y_true)

    # Verify shapes
    for name, preds in predictions_dict.items():
        if preds.shape[0] != n_samples:
            raise ValueError(
                f"Prediction shape mismatch for {name}: expected {n_samples} rows, got {preds.shape[0]}."
            )

    # 2. Initialization
    selected_experts = []
    # We maintain the sum of predictions to avoid re-summing at every step
    # Shape: (N_samples, N_classes)
    # Initialize with zeros. We infer N_classes from the first expert.
    first_expert_preds = predictions_dict[expert_names[0]]
    current_sum_preds = np.zeros_like(first_expert_preds, dtype=np.float64)

    best_score = float("inf")

    if verbose:
        print(
            f"Starting Greedy Forward Selection with {len(expert_names)} candidates, max_iter={max_iter}..."
        )

    # 3. Iterative Selection Loop
    for k in range(1, max_iter + 1):
        iteration_best_score = float("inf")
        iteration_best_expert = None

        # Try adding each expert to the current ensemble
        for name in expert_names:
            preds = predictions_dict[name]

            # Calculate trial ensemble average
            # Ensemble size will be k
            trial_ensemble_preds = (current_sum_preds + preds) / k

            # Clip probabilities to avoid log loss extremes (strictly following competition metric)
            trial_ensemble_preds = clip_probabilities(trial_ensemble_preds)

            # Calculate score
            score = log_loss_metric(y_true, trial_ensemble_preds)

            if score < iteration_best_score:
                iteration_best_score = score
                iteration_best_expert = name

        # 4. Check for Improvement
        # We check if the best score of this iteration is better than the global best score
        # minus the tolerance.
        improvement = best_score - iteration_best_score

        if improvement > tol:
            best_score = iteration_best_score
            selected_experts.append(iteration_best_expert)

            # Update the running sum with the selected expert's predictions
            current_sum_preds += predictions_dict[iteration_best_expert]

            if verbose:
                print(
                    f"Step {k}: Added '{iteration_best_expert}'. Best Score: {best_score:.15f} (Improved by {improvement:.15f})"
                )
        else:
            if verbose:
                print(
                    f"Step {k}: No significant improvement (Best Iter Score: {iteration_best_score:.15f}). Stopping."
                )
            break

    # 5. Calculate Weights
    if not selected_experts:
        # Fallback if nothing improved (unlikely unless max_iter=0 or data is broken)
        # Select the single best model
        if verbose:
            print("Warning: No experts selected. Reverting to single best model.")

        best_single_score = float("inf")
        best_single_name = None
        for name, preds in predictions_dict.items():
            s = log_loss_metric(y_true, clip_probabilities(preds))
            if s < best_single_score:
                best_single_score = s
                best_single_name = name

        return {best_single_name: 1.0}, best_single_score

    # Count frequencies
    expert_counts = Counter(selected_experts)
    total_selected = len(selected_experts)

    weights = {name: count / total_selected for name, count in expert_counts.items()}

    if verbose:
        print("\nFinal Selection:")
        for name, weight in weights.items():
            print(f"  - {name}: {weight:.4f} ({expert_counts[name]}/{total_selected})")
        print(f"Final Ensemble Score: {best_score:.15f}")

    return weights, best_score


def compute_ensemble_prediction(predictions_dict, weights):
    """
    Computes the weighted average prediction for an ensemble.

    Args:
        predictions_dict (dict): Dictionary of expert predictions.
        weights (dict): Dictionary of weights for each expert.

    Returns:
        np.ndarray: Weighted average probability matrix.
    """
    if not weights:
        raise ValueError("Weights dictionary cannot be empty.")

    # Initialize sum
    # Get shape from first expert
    first_key = next(iter(weights))
    shape = predictions_dict[first_key].shape
    ensemble_preds = np.zeros(shape, dtype=np.float64)

    total_weight = 0.0

    for name, weight in weights.items():
        if name not in predictions_dict:
            raise KeyError(
                f"Expert '{name}' found in weights but not in predictions dictionary."
            )

        ensemble_preds += predictions_dict[name] * weight
        total_weight += weight

    # Normalize if weights don't sum to exactly 1.0 (though they should)
    if total_weight > 0:
        ensemble_preds /= total_weight

    return ensemble_preds
