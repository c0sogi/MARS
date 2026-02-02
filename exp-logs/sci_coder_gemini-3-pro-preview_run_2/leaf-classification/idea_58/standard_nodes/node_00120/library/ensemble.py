import numpy as np
from collections import Counter
from library import utils
from library import config


class GreedyForwardSelector:
    """
    Implements Greedy Forward Selection for ensemble optimization.

    This selector iteratively adds models to an ensemble to minimize the
    validation log loss. It allows for selection with replacement (weighted ensemble).
    """

    def __init__(self, max_iterations=100, tolerance=1e-6, verbose=True):
        """
        Args:
            max_iterations (int): Maximum number of experts to select (ensemble size).
            tolerance (float): Minimum improvement in log loss required to continue adding experts.
            verbose (bool): Whether to print progress logs.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.selected_experts = []
        self.best_score = float("inf")
        self.weights = {}

    def fit(self, predictions_dict, y_true):
        """
        Runs the greedy forward selection process.

        Args:
            predictions_dict (dict): Dictionary where keys are expert identifiers (str)
                                     and values are numpy arrays of shape (n_samples, n_classes)
                                     containing predicted probabilities.
            y_true (np.array): Ground truth labels of shape (n_samples,).

        Returns:
            dict: A dictionary mapping expert identifiers to their integer weights (counts).
        """
        utils.set_seed()

        if not predictions_dict:
            raise ValueError("predictions_dict cannot be empty.")

        expert_names = list(predictions_dict.keys())
        n_experts = len(expert_names)

        if self.verbose:
            print(f"Starting Greedy Forward Selection with {n_experts} candidates.")
            print(f"Max iterations: {self.max_iterations}, Tolerance: {self.tolerance}")

        # ---------------------------------------------------------------------
        # Step 1: Initialization - Find the single best model
        # ---------------------------------------------------------------------
        best_initial_expert = None
        best_initial_score = float("inf")

        # We store the running sum of predictions to avoid re-summing the whole history every time
        # This makes the complexity O(N_iter * N_experts) instead of O(N_iter^2 * N_experts)
        current_sum_preds = None

        for name in expert_names:
            preds = predictions_dict[name]
            score = utils.clipped_log_loss(y_true, preds)

            if score < best_initial_score:
                best_initial_score = score
                best_initial_expert = name

        # Initialize state with the best single model
        self.selected_experts = [best_initial_expert]
        self.best_score = best_initial_score
        current_sum_preds = predictions_dict[best_initial_expert].copy()

        if self.verbose:
            print(
                f"Iteration 1/{self.max_iterations}: Selected '{best_initial_expert}'"
            )
            print(f"Current Best Log Loss: {self.best_score}")

        # ---------------------------------------------------------------------
        # Step 2: Iterative Selection
        # ---------------------------------------------------------------------
        # We start loop from 2 because we already selected the first one
        for i in range(2, self.max_iterations + 1):
            iteration_best_score = float("inf")
            iteration_best_expert = None

            # Current ensemble size before adding candidate is i-1.
            # New size will be i.
            new_size = float(i)

            # Try adding each candidate to the existing ensemble
            for name in expert_names:
                candidate_preds = predictions_dict[name]

                # Calculate temporary ensemble average
                # Ensemble Prob = (Sum of previous selected preds + candidate preds) / count
                temp_avg_preds = (current_sum_preds + candidate_preds) / new_size

                score = utils.clipped_log_loss(y_true, temp_avg_preds)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_expert = name

            # Check improvement
            improvement = self.best_score - iteration_best_score

            if improvement > self.tolerance:
                self.best_score = iteration_best_score
                self.selected_experts.append(iteration_best_expert)

                # Update the running sum
                current_sum_preds += predictions_dict[iteration_best_expert]

                if self.verbose:
                    print(
                        f"Iteration {i}/{self.max_iterations}: Added '{iteration_best_expert}'"
                    )
                    print(f"  Improvement: {improvement}")
                    print(f"  Current Best Log Loss: {self.best_score}")
            else:
                if self.verbose:
                    print(f"Stopping early at iteration {i}.")
                    print(f"  Improvement {improvement} < Tolerance {self.tolerance}")
                break

        # ---------------------------------------------------------------------
        # Step 3: Finalize Weights
        # ---------------------------------------------------------------------
        self.weights = dict(Counter(self.selected_experts))

        if self.verbose:
            print("-" * 30)
            print("Selection Complete.")
            print(f"Final Ensemble Size: {len(self.selected_experts)}")
            print(f"Final Log Loss: {self.best_score}")
            print(f"Selected Experts & Weights: {self.weights}")
            print("-" * 30)

        return self.weights

    def predict(self, predictions_dict, weights=None):
        """
        Computes the weighted average prediction for a set of experts.

        Args:
            predictions_dict (dict): Dictionary of expert predictions (e.g., on test set).
            weights (dict, optional): Dictionary of weights. If None, uses fitted weights.

        Returns:
            np.array: The weighted average probability matrix.
        """
        if weights is None:
            weights = self.weights

        if not weights:
            raise ValueError(
                "No weights available. Call fit() first or provide weights."
            )

        # Validate that all weighted experts are in the predictions dictionary
        missing_experts = [exp for exp in weights.keys() if exp not in predictions_dict]
        if missing_experts:
            raise ValueError(
                f"Predictions dictionary missing keys for selected experts: {missing_experts}"
            )

        # Compute weighted sum
        # We need the shape from one of the arrays
        first_expert = next(iter(weights))
        sample_shape = predictions_dict[first_expert].shape

        weighted_sum = np.zeros(sample_shape, dtype=np.float64)
        total_weight = 0.0

        for expert_name, weight in weights.items():
            weighted_sum += predictions_dict[expert_name] * weight
            total_weight += weight

        if total_weight == 0:
            raise ValueError("Total weight is zero.")

        return weighted_sum / total_weight
