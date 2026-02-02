import numpy as np
from collections import Counter
from library.utils import score_predictions


class GreedySelector:
    """
    Implements a Greedy Forward Selection (Hill Climbing) strategy to optimize
    ensemble weights by iteratively adding experts that minimize the validation Log Loss.
    """

    def __init__(self, max_iterations=100, tolerance=1e-6, verbose=True):
        """
        Initialize the GreedySelector.

        Args:
            max_iterations (int): Maximum number of selection rounds (experts to add).
            tolerance (float): Minimum improvement in Log Loss required to continue adding experts.
            verbose (bool): Whether to print progress logs.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.weights_ = {}
        self.selected_experts_ = []
        self.best_score_ = float("inf")

    def fit(self, predictions_dict, y_true, labels=None):
        """
        Fits the ensemble weights using the provided validation predictions and labels.

        Args:
            predictions_dict (dict): Dictionary where keys are expert names and values are
                                     numpy arrays of shape (n_samples, n_classes) containing probabilities.
            y_true (array-like): True labels for the validation set.
            labels (list, optional): List of class names corresponding to the columns of the probability matrices.
                                     Required if y_true contains string labels.

        Returns:
            self: The fitted instance.
        """
        # 1. Data Preparation
        # Cast to float64 for maximum precision during accumulation
        expert_preds = {}
        n_samples = len(y_true)

        for name, preds in predictions_dict.items():
            if len(preds) != n_samples:
                raise ValueError(
                    f"Length mismatch for expert {name}: expected {n_samples}, got {len(preds)}"
                )
            expert_preds[name] = np.array(preds, dtype=np.float64)

        available_experts = list(expert_preds.keys())
        if not available_experts:
            raise ValueError("predictions_dict cannot be empty.")

        if self.verbose:
            print(
                f"Starting Greedy Forward Selection with {len(available_experts)} candidates..."
            )
            print("-" * 60)

        # 2. Initialization: Find the single best expert
        best_single_expert = None
        best_single_score = float("inf")

        for name in available_experts:
            # score_predictions handles row-normalization and clipping internally
            score = score_predictions(y_true, expert_preds[name], labels=labels)
            if score < best_single_score:
                best_single_score = score
                best_single_expert = name

        self.best_score_ = best_single_score
        self.selected_experts_ = [best_single_expert]

        # Maintain the sum of predictions for the current ensemble
        # This avoids re-summing the entire history in every iteration
        current_ensemble_sum = expert_preds[best_single_expert].copy()

        if self.verbose:
            print(f"Initial Best Single Model: {best_single_expert}")
            print(f"Initial Score: {self.best_score_:.15f}")

        # 3. Iterative Selection
        for i in range(self.max_iterations):
            iteration_best_score = float("inf")
            iteration_best_expert = None

            # Try adding each expert from the library (sampling with replacement is allowed)
            for name in available_experts:
                # Calculate the sum of the proposed ensemble (Current + Candidate)
                # Note: We pass the sum directly. score_predictions divides by row sums,
                # effectively computing the average: (Sum / N) / (RowSum / N) = Sum / RowSum
                proposed_sum = current_ensemble_sum + expert_preds[name]

                score = score_predictions(y_true, proposed_sum, labels=labels)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_expert = name

            # Check for improvement
            improvement = self.best_score_ - iteration_best_score

            if improvement > self.tolerance:
                self.best_score_ = iteration_best_score
                self.selected_experts_.append(iteration_best_expert)
                current_ensemble_sum += expert_preds[iteration_best_expert]

                if self.verbose:
                    print(
                        f"Round {i+1:02d}: Added {iteration_best_expert:<35} | Score: {self.best_score_:.15f} | Improv: {improvement:.15f}"
                    )
            else:
                if self.verbose:
                    print(
                        f"Round {i+1:02d}: No improvement > {self.tolerance}. Stopping."
                    )
                break

        # 4. Finalize Weights
        self.weights_ = dict(Counter(self.selected_experts_))

        if self.verbose:
            print("-" * 60)
            print("Final Ensemble Weights:")
            for name, weight in self.weights_.items():
                print(f"  {name}: {weight}")
            print(f"Final Best Score: {self.best_score_:.15f}")
            print("-" * 60)

        return self

    def predict(self, predictions_dict):
        """
        Computes the weighted average probabilities using the fitted weights.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to probability matrices (N, C).
                                     Must contain all experts selected during fit.

        Returns:
            np.ndarray: Weighted average probability matrix of shape (N, C).
        """
        if not self.weights_:
            raise RuntimeError(
                "GreedySelector is not fitted. Call fit() before predict()."
            )

        # Determine shape from the first selected expert
        first_expert = next(iter(self.weights_))
        if first_expert not in predictions_dict:
            raise ValueError(
                f"Selected expert '{first_expert}' not found in input predictions."
            )

        input_shape = predictions_dict[first_expert].shape
        weighted_sum = np.zeros(input_shape, dtype=np.float64)
        total_weight = 0.0

        for name, weight in self.weights_.items():
            if name not in predictions_dict:
                raise ValueError(
                    f"Selected expert '{name}' not found in input predictions."
                )

            # Accumulate weighted predictions
            weighted_sum += weight * predictions_dict[name].astype(np.float64)
            total_weight += weight

        # Normalize to valid probabilities [0, 1]
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return np.zeros(input_shape)
