import numpy as np
import library.config as conf
import library.utils as utils


def greedy_forward_selection(
    expert_preds,
    y_true,
    n_iterations=conf.SELECTION_ITERATIONS,
    with_replacement=conf.SELECTION_WITH_REPLACEMENT,
):
    """
    Performs Greedy Forward Selection to optimize ensemble weights.

    This function iteratively adds the expert model that maximizes the improvement
    in the validation metric (Log Loss) to the ensemble.

    Args:
        expert_preds (dict): A dictionary where keys are expert names (str) and
                             values are prediction matrices (np.ndarray of shape [n_samples, n_classes]).
                             Predictions should be probabilities (0-1).
        y_true (np.ndarray): Ground truth class indices (shape [n_samples]).
        n_iterations (int): The number of iterations to perform (i.e., final ensemble size).
        with_replacement (bool): If True, the same expert can be selected multiple times
                                 (equivalent to integer weighting).

    Returns:
        tuple: (weights, best_score)
            - weights (dict): Dictionary mapping expert names to their count/weight in the ensemble.
            - best_score (float): The final validation log loss achieved.
    """
    # Ensure y_true is a numpy array of integers
    y_true = np.array(y_true, dtype=int)

    # Get list of expert names
    expert_names = list(expert_preds.keys())

    if not expert_names:
        raise ValueError("expert_preds dictionary is empty.")

    # Initialize variables
    selected_experts = []
    # Accumulator for predictions of selected models (sum of probabilities)
    # Using float64 for precision as defined in config
    current_ensemble_sum = None
    best_loss_history = []

    print(
        f"Starting Greedy Forward Selection (Iterations={n_iterations}, Replacement={with_replacement})..."
    )

    for k in range(n_iterations):
        best_iter_loss = float("inf")
        best_iter_expert = None

        # Define candidates for this iteration
        if with_replacement:
            candidates = expert_names
        else:
            candidates = [name for name in expert_names if name not in selected_experts]

        if not candidates:
            print("No remaining candidates to select from.")
            break

        # Evaluate each candidate
        for name in candidates:
            pred = expert_preds[name]

            # Calculate the new ensemble prediction if we add this candidate
            if k == 0:
                # First iteration: Ensemble is just the candidate
                temp_ensemble_pred = pred
            else:
                # Update mean: (Sum_prev + Candidate) / (k + 1)
                # Note: We compute the average for scoring.
                # The division by (k+1) is crucial for the probabilities to be in valid range [0,1]
                # before clipping/scoring, although clipped_log_loss handles rescaling.
                temp_ensemble_pred = (current_ensemble_sum + pred) / (k + 1)

            # Calculate Log Loss
            # We use the clipped_log_loss from utils which handles row-normalization and clipping
            loss = utils.clipped_log_loss(y_true, temp_ensemble_pred)

            if loss < best_iter_loss:
                best_iter_loss = loss
                best_iter_expert = name

        # Check if we found a valid expert (should always be true unless candidates empty)
        if best_iter_expert is not None:
            # Add to selected list
            selected_experts.append(best_iter_expert)
            best_loss_history.append(best_iter_loss)

            # Update the running sum of predictions
            if k == 0:
                current_ensemble_sum = expert_preds[best_iter_expert].copy()
            else:
                current_ensemble_sum += expert_preds[best_iter_expert]

            # Print full precision metric
            print(
                f"Iteration {k+1}: Selected '{best_iter_expert}' with Validation Loss: {best_iter_loss}"
            )
        else:
            print(f"Iteration {k+1}: No improvement possible.")
            break

    # Compute final weights (counts of each expert)
    weights = {}
    for name in selected_experts:
        weights[name] = weights.get(name, 0) + 1

    final_score = best_loss_history[-1] if best_loss_history else float("inf")

    print("-" * 30)
    print("Selection Complete")
    print(f"Total Experts Selected: {len(selected_experts)}")
    print(f"Final Validation Loss: {final_score}")
    print(f"Weights: {weights}")
    print("-" * 30)

    return weights, final_score
