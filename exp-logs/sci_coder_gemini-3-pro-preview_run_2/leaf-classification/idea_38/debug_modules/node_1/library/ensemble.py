import numpy as np
from library.utils import clipped_log_loss
from library.config import Config


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement (Caruana et al., 2004).

    This algorithm starts with the single best model and iteratively adds the model
    (potentially one already selected) that maximizes the ensemble's performance
    on the validation set. This results in a weighted average ensemble where weights
    are proportional to the selection frequency.
    """

    def __init__(self, max_iterations=100, tolerance=1e-6):
        """
        Args:
            max_iterations (int): Maximum number of selection steps (corresponds to sum of weights).
            tolerance (float): Minimum improvement in Log Loss required to continue selection.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.selected_experts = {}  # Dictionary {expert_id: count/weight}
        self.best_score = float("inf")
        self.history = []  # List of (score, expert_id) tuples

    def fit(self, predictions_dict, y_true):
        """
        Fits the ensemble weights using the provided validation predictions and labels.

        Args:
            predictions_dict (dict): Key is expert_id, Value is np.ndarray (N, K) of probabilities.
            y_true (np.ndarray): True class indices (N,).

        Returns:
            dict: The dictionary of selected experts and their integer weights.
        """
        # Ensure y_true is an integer array
        y_true = np.array(y_true, dtype=int)

        expert_ids = list(predictions_dict.keys())
        if not expert_ids:
            raise ValueError("predictions_dict cannot be empty.")

        print(f"Starting Greedy Selection with {len(expert_ids)} experts...")

        # ---------------------------------------------------------------------
        # 1. Initialization: Find the single best expert to start the ensemble
        # ---------------------------------------------------------------------
        best_init_score = float("inf")
        best_init_expert = None

        for eid in expert_ids:
            preds = predictions_dict[eid]
            # Calculate score (lower is better for Log Loss)
            score = clipped_log_loss(y_true, preds)

            if score < best_init_score:
                best_init_score = score
                best_init_expert = eid

        if best_init_expert is None:
            raise ValueError("Could not find a valid starting expert.")

        # Initialize state
        self.selected_experts = {best_init_expert: 1}
        self.best_score = best_init_score
        self.history.append((best_init_score, best_init_expert))

        # Maintain the current sum of unnormalized probabilities (weighted sum)
        # Start with the predictions of the best single expert (weight=1)
        current_sum_preds = (
            predictions_dict[best_init_expert].copy().astype(Config.FLOAT_PRECISION)
        )
        current_total_weight = 1

        print(
            f"Initialization: Best single expert is '{best_init_expert}' with Log Loss: {best_init_score}"
        )

        # ---------------------------------------------------------------------
        # 2. Iterative Selection
        # ---------------------------------------------------------------------
        for i in range(self.max_iterations):
            best_iter_score = float("inf")
            best_iter_expert = None

            # Try adding each expert to the current ensemble
            for eid in expert_ids:
                candidate_preds = predictions_dict[eid]

                # Calculate what the ensemble predictions would be if we added this expert
                # New Avg = (Current Sum + Candidate) / (Total Weight + 1)
                temp_sum = current_sum_preds + candidate_preds
                temp_avg = temp_sum / (current_total_weight + 1)

                score = clipped_log_loss(y_true, temp_avg)

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_expert = eid

            # Check for improvement
            # Note: We use strictly greater than tolerance to avoid adding noise if improvement is infinitesimal
            improvement = self.best_score - best_iter_score

            if improvement > self.tolerance:
                self.best_score = best_iter_score
                self.selected_experts[best_iter_expert] = (
                    self.selected_experts.get(best_iter_expert, 0) + 1
                )

                # Update the running sum
                current_sum_preds += predictions_dict[best_iter_expert]
                current_total_weight += 1

                self.history.append((self.best_score, best_iter_expert))
                print(
                    f"Iteration {i+1}: Added '{best_iter_expert}'. New Log Loss: {self.best_score}"
                )
            else:
                print(
                    f"Iteration {i+1}: Improvement {improvement} <= tolerance {self.tolerance}. Stopping selection."
                )
                break

        print("Selection complete.")
        print(f"Final Ensemble Weights: {self.selected_experts}")
        return self.selected_experts

    def predict(self, predictions_dict):
        """
        Generates ensemble predictions using the fitted weights.

        Args:
            predictions_dict (dict): Key is expert_id, Value is np.ndarray (N, K) of probabilities.
                                     Must contain all experts selected during fit.

        Returns:
            np.ndarray: Weighted average probabilities (N, K).
        """
        if not self.selected_experts:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")

        # Determine shape from the first available array
        first_eid = next(iter(predictions_dict))
        n_samples, n_classes = predictions_dict[first_eid].shape

        # Initialize weighted sum
        weighted_sum = np.zeros((n_samples, n_classes), dtype=Config.FLOAT_PRECISION)
        total_weight = 0

        missing_experts = []

        for eid, weight in self.selected_experts.items():
            if eid in predictions_dict:
                weighted_sum += predictions_dict[eid] * weight
                total_weight += weight
            else:
                missing_experts.append(eid)

        if missing_experts:
            raise KeyError(
                f"The following selected experts are missing from the input predictions: {missing_experts}"
            )

        if total_weight == 0:
            raise ValueError("Total ensemble weight is zero.")

        # Normalize
        final_preds = weighted_sum / total_weight

        return final_preds
