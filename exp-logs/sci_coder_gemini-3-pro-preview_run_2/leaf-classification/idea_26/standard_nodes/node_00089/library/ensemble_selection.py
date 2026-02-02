import numpy as np
from collections import Counter
from library.utils import clipped_log_loss


def select_best_ensemble(predictions_dict, y_true, max_iter=100, tolerance=1e-6):
    """
    Performs Greedy Forward Selection (with replacement) to find the optimal ensemble combination.

    The algorithm starts with the single best model and iteratively adds the model that
    maximizes the improvement in validation log loss.

    Args:
        predictions_dict (dict): A dictionary where keys are model names and values are
                                 numpy arrays of shape (n_samples, n_classes) representing
                                 predicted probabilities.
        y_true (np.ndarray): Ground truth labels of shape (n_samples,).
        max_iter (int): Maximum number of models to include in the ensemble.
        tolerance (float): Minimum improvement in log loss required to add a model.

    Returns:
        dict: A dictionary mapping selected model names to their integer weights (counts).
              Example: {'LDA_Global_auto': 3, 'QDA_Morph_reg0.1': 1}
    """
    model_names = list(predictions_dict.keys())

    # 1. Initialization: Find the single best model to start
    best_initial_score = float("inf")
    best_initial_model = None

    print("Starting Greedy Forward Selection...")
    print(f"Candidates: {len(model_names)} models")

    for name in model_names:
        preds = predictions_dict[name]
        score = clipped_log_loss(y_true, preds)
        if score < best_initial_score:
            best_initial_score = score
            best_initial_model = name

    # Initialize ensemble with the best single model
    ensemble_models = [best_initial_model]
    current_sum_preds = predictions_dict[best_initial_model].copy()
    best_score = best_initial_score

    print(f"Initial Best Single Model: {best_initial_model}")
    print(f"Initial Score: {best_score}")

    # 2. Iterative Selection
    for i in range(max_iter - 1):  # -1 because we already added the first one
        best_step_score = float("inf")
        best_step_model = None

        current_size = len(ensemble_models)

        # Try adding each candidate to the current ensemble
        for name in model_names:
            candidate_preds = predictions_dict[name]

            # Calculate new average: (Sum + Candidate) / (N + 1)
            # We do this temporarily for evaluation
            temp_sum_preds = current_sum_preds + candidate_preds
            temp_avg_preds = temp_sum_preds / (current_size + 1)

            score = clipped_log_loss(y_true, temp_avg_preds)

            if score < best_step_score:
                best_step_score = score
                best_step_model = name

        # Check for improvement
        improvement = best_score - best_step_score

        if improvement > tolerance:
            # Update state
            ensemble_models.append(best_step_model)
            current_sum_preds += predictions_dict[best_step_model]
            best_score = best_step_score
            print(
                f"Iteration {i+1}: Added {best_step_model}, New Score: {best_score}, Improvement: {improvement}"
            )
        else:
            print(f"Iteration {i+1}: No improvement > {tolerance}. Stopping.")
            break

    # 3. Format Output
    weights = dict(Counter(ensemble_models))

    print("-" * 30)
    print("Ensemble Selection Complete")
    print(f"Final Ensemble Size: {len(ensemble_models)}")
    print(f"Final Validation Log Loss: {best_score}")
    print(f"Selected Models and Weights: {weights}")
    print("-" * 30)

    return weights
