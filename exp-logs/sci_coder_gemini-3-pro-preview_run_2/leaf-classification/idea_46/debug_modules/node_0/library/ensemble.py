import numpy as np
from collections import Counter
from library import config, utils


class GreedySelector:
    """
    Implements Greedy Forward Selection for Dynamic Ensemble Selection.

    This class iteratively selects experts from a library of candidates to
    minimize the validation log loss. It supports selection with replacement,
    effectively assigning integer weights to experts.
    """

    def __init__(
        self, iterations=config.SELECTION_ITERATIONS, random_state=config.RANDOM_STATE
    ):
        """
        Initialize the selector.

        Args:
            iterations (int): Maximum number of selection iterations.
            random_state (int): Seed for reproducibility.
        """
        self.iterations = iterations
        self.random_state = random_state
        self.selected_experts = []
        self.best_score = float("inf")
        self.weights = {}

    def fit(self, expert_predictions_dict, y_true):
        """
        Runs the Greedy Forward Selection process to determine the optimal ensemble.

        Args:
            expert_predictions_dict (dict): Dictionary where keys are expert names and
                                            values are prediction matrices (np.ndarray)
                                            of shape (n_samples, n_classes).
            y_true (np.ndarray): Ground truth labels for the validation set.
        """
        # Ensure consistent random state
        utils.set_seed(self.random_state)

        # Sort keys for deterministic tie-breaking
        available_experts = sorted(list(expert_predictions_dict.keys()))
        n_experts = len(available_experts)

        if n_experts == 0:
            raise ValueError("No expert predictions provided for selection.")

        print(
            f"Starting Greedy Forward Selection with {n_experts} candidates for {self.iterations} iterations..."
        )

        # Initialize running sum of predictions for the current ensemble
        # We use a running sum to avoid re-summing the entire history in the inner loop
        current_ensemble_sum = None

        # Reset state
        self.selected_experts = []
        self.best_score = float("inf")

        for i in range(self.iterations):
            iteration_best_score = float("inf")
            iteration_best_expert = None

            # Iterate through all available experts to find the best addition
            for expert_name in available_experts:
                # Ensure float64 precision
                preds = expert_predictions_dict[expert_name].astype(
                    config.FLOAT_PRECISION
                )

                # Calculate candidate ensemble prediction
                if i == 0:
                    # If ensemble is empty, the candidate is the ensemble
                    candidate_ensemble_preds = preds
                else:
                    # New Average = (Current Sum + Candidate) / (Current Count + 1)
                    candidate_ensemble_preds = (current_ensemble_sum + preds) / (i + 1)

                # Evaluate using the competition metric
                score = utils.clipped_log_loss(y_true, candidate_ensemble_preds)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_expert = expert_name

            # Decision Step: Strictly require improvement to continue
            if iteration_best_score < self.best_score:
                self.best_score = iteration_best_score
                self.selected_experts.append(iteration_best_expert)

                # Update the running sum for the next iteration
                best_preds = expert_predictions_dict[iteration_best_expert].astype(
                    config.FLOAT_PRECISION
                )
                if current_ensemble_sum is None:
                    current_ensemble_sum = best_preds
                else:
                    current_ensemble_sum += best_preds

                print(
                    f"Iteration {i+1}: Added '{iteration_best_expert}' | Val Score: {self.best_score}"
                )
            else:
                print(
                    f"Iteration {i+1}: No improvement (Best Candidate: {iteration_best_score} >= Current: {self.best_score}). Stopping."
                )
                break

        # Compute final weights based on selection frequency
        self.weights = dict(Counter(self.selected_experts))

        print("-" * 30)
        print("Selection Complete.")
        print(f"Final Ensemble Size: {len(self.selected_experts)}")
        print(f"Final Validation Score: {self.best_score}")
        print(f"Selected Experts & Weights: {self.weights}")
        print("-" * 30)

    def predict(self, expert_predictions_dict):
        """
        Aggregates predictions from the provided dictionary using the learned weights.

        Args:
            expert_predictions_dict (dict): Dictionary of expert predictions for the test set.

        Returns:
            np.ndarray: Weighted average prediction matrix.
        """
        if not self.selected_experts:
            print(
                "Warning: No experts were selected during fit. Returning mean of all inputs."
            )
            # Fallback: simple average of all provided inputs
            all_preds = list(expert_predictions_dict.values())
            return np.mean(all_preds, axis=0)

        # Initialize result array using the shape of the first selected expert
        first_expert = self.selected_experts[0]
        if first_expert not in expert_predictions_dict:
            raise KeyError(
                f"Selected expert '{first_expert}' not found in input predictions."
            )

        sample_shape = expert_predictions_dict[first_expert].shape
        weighted_sum = np.zeros(sample_shape, dtype=config.FLOAT_PRECISION)
        total_weight = 0

        # Aggregate weighted predictions
        for expert_name, weight in self.weights.items():
            if expert_name not in expert_predictions_dict:
                raise KeyError(
                    f"Selected expert '{expert_name}' not found in input predictions."
                )

            preds = expert_predictions_dict[expert_name].astype(config.FLOAT_PRECISION)
            weighted_sum += preds * weight
            total_weight += weight

        # Compute final weighted average
        if total_weight == 0:
            return np.zeros(sample_shape, dtype=config.FLOAT_PRECISION)

        final_predictions = weighted_sum / total_weight

        return final_predictions
