import numpy as np
from sklearn.metrics import log_loss
from library.config import Config


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement to optimize the ensemble composition.
    This selector iteratively adds experts to the ensemble if and only if they maximize
    the improvement in the validation metric (Log Loss).
    """

    def __init__(self):
        self.selected_experts = []
        self.best_score = float("inf")
        self.classes_ = None

    def _normalize_and_clip(self, probas):
        """
        Normalizes probability rows to sum to 1 and clips them to the epsilon range
        defined in the Config to avoid log(0) and match the evaluation metric.

        Args:
            probas (np.ndarray): Probability matrix of shape (N, C).

        Returns:
            np.ndarray: Normalized and clipped probability matrix.
        """
        # 1. Normalize (Row-wise sum to 1)
        # As per task description: "rescaled prior to being scored (each row is divided by the row sum)"
        row_sums = probas.sum(axis=1, keepdims=True)
        # Handle potential zero sums (though unlikely with valid models) to avoid NaN
        row_sums[row_sums == 0] = 1.0
        probas_norm = probas / row_sums

        # 2. Clip
        # As per task description: max(min(p, 1-10^-15), 10^-15)
        eps = Config.PROB_CLIP_EPS
        probas_clipped = np.clip(probas_norm, eps, 1.0 - eps)

        return probas_clipped

    def fit(
        self,
        predictions_dict,
        y_true,
        max_iterations=50,
        tolerance=1e-6,
        verbose=True,
    ):
        """
        Runs the Greedy Forward Selection algorithm.

        Args:
            predictions_dict (dict): Dictionary where keys are expert names and values
                                     are np.ndarrays of shape (N_samples, N_classes)
                                     containing predicted probabilities.
            y_true (np.ndarray): Array of shape (N_samples,) containing true labels.
            max_iterations (int): Maximum number of selection rounds (ensemble size limit).
            tolerance (float): Minimum improvement in log loss required to add an expert.
            verbose (bool): If True, prints progress to stdout.

        Returns:
            self: The fitted selector instance.
        """
        # Identify classes for log_loss calculation
        # We assume predictions columns correspond to sorted unique labels of y_true
        self.classes_ = np.unique(y_true)

        available_experts = list(predictions_dict.keys())
        if not available_experts:
            raise ValueError("predictions_dict is empty. No experts to select from.")

        # State initialization
        self.selected_experts = []
        self.best_score = float("inf")

        # Accumulator for the sum of probabilities of the currently selected ensemble
        # We sum first, then divide by N to get the average
        current_sum_probas = None
        n_selected = 0

        if verbose:
            print(
                f"Starting Greedy Forward Selection (Max Iterations: {max_iterations})..."
            )

        for i in range(max_iterations):
            iteration_best_score = float("inf")
            iteration_best_expert = None

            # Try adding each expert to the current ensemble
            for expert_name in available_experts:
                expert_probas = predictions_dict[expert_name]

                # Calculate tentative ensemble prediction
                if current_sum_probas is None:
                    # Case: First selection
                    tentative_sum = expert_probas
                    tentative_n = 1
                else:
                    # Case: Adding to existing ensemble
                    # Note: + creates a new array, so current_sum_probas is safe
                    tentative_sum = current_sum_probas + expert_probas
                    tentative_n = n_selected + 1

                # Average
                tentative_avg = tentative_sum / tentative_n

                # Normalize and Clip
                tentative_pred = self._normalize_and_clip(tentative_avg)

                # Evaluate
                score = log_loss(y_true, tentative_pred, labels=self.classes_)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_expert = expert_name

            # Check improvement
            improvement = self.best_score - iteration_best_score

            if improvement > tolerance:
                # Commit the selection
                self.best_score = iteration_best_score
                self.selected_experts.append(iteration_best_expert)

                # Update accumulator
                # Use copy on first assignment to avoid modifying the source dictionary array later
                if current_sum_probas is None:
                    current_sum_probas = predictions_dict[iteration_best_expert].copy()
                else:
                    current_sum_probas += predictions_dict[iteration_best_expert]

                n_selected += 1

                if verbose:
                    print(
                        f"Iteration {i+1}: Added '{iteration_best_expert}' | "
                        f"Val Log Loss: {self.best_score:.15f} | "
                        f"Improvement: {improvement:.15f}"
                    )
            else:
                # Stop if no significant improvement
                if verbose:
                    print(
                        f"Iteration {i+1}: No significant improvement "
                        f"({improvement:.15f} <= {tolerance}). Stopping."
                    )
                break

        if not self.selected_experts and verbose:
            print("Warning: No experts were selected during the fitting process.")

        return self

    def predict(self, predictions_dict):
        """
        Aggregates predictions from the selected experts using the weights determined
        implicitly by the selection frequency (selection with replacement).

        Args:
            predictions_dict (dict): Dictionary of expert predictions for the target set.
                                     Keys must include all names in self.selected_experts.

        Returns:
            np.ndarray: Aggregated, normalized, and clipped probability matrix.
        """
        if not self.selected_experts:
            # Fallback strategy: Average all available experts if selection failed or wasn't run
            # This ensures we always return a valid prediction matrix
            print(
                "Warning: No experts selected. Averaging all available experts in predictions_dict."
            )
            all_preds = list(predictions_dict.values())
            if not all_preds:
                raise ValueError("predictions_dict is empty.")
            avg_preds = np.mean(all_preds, axis=0)
            return self._normalize_and_clip(avg_preds)

        # Sum predictions based on the selected list (handling duplicates/weights)
        sum_probas = None
        count = 0

        for expert_name in self.selected_experts:
            if expert_name not in predictions_dict:
                raise KeyError(
                    f"Selected expert '{expert_name}' is missing from the provided predictions_dict."
                )

            preds = predictions_dict[expert_name]

            if sum_probas is None:
                sum_probas = preds.copy()
            else:
                sum_probas += preds
            count += 1

        # Compute Average
        avg_probas = sum_probas / count

        # Final Normalize and Clip
        return self._normalize_and_clip(avg_probas)
