import numpy as np
from sklearn.metrics import log_loss
from collections import Counter


class GreedyForwardSelector:
    """
    Implements Greedy Forward Selection (Ensemble Selection) to find the optimal
    weighted combination of expert predictions that minimizes Log Loss.

    This strategy builds an ensemble by iteratively adding the model that maximizes
    ensemble performance on the validation set. It allows for selection with replacement,
    effectively assigning integer weights to experts.
    """

    def __init__(self, max_iter=100, tol=1e-6, verbose=True):
        """
        Args:
            max_iter (int): Maximum number of selection iterations.
            tol (float): Minimum improvement in Log Loss required to continue adding experts.
            verbose (bool): Whether to print selection progress.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.weights = {}  # Dictionary mapping expert_name -> integer weight
        self.selected_experts = []  # List of selected experts in order
        self.best_score = float("inf")

    def _calculate_score(self, y_true, y_pred):
        """
        Calculates Multi-class Log Loss with specific clipping as per task requirements.

        Args:
            y_true (np.array): True class indices or one-hot encoded labels.
            y_pred (np.array): Predicted probabilities.

        Returns:
            float: Log Loss score.
        """
        # Task requirement: predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        # Normalize rows to sum to 1 (standard practice before scoring)
        row_sums = y_pred_clipped.sum(axis=1)
        y_pred_norm = y_pred_clipped / row_sums[:, np.newaxis]

        return log_loss(y_true, y_pred_norm)

    def fit(self, expert_preds_dict, y_true):
        """
        Runs the Greedy Forward Selection algorithm.

        Args:
            expert_preds_dict (dict): Dictionary {expert_name: np.ndarray(N, n_classes)}.
                                      Contains validation probabilities for each expert.
            y_true (np.ndarray): True validation labels (N,).

        Returns:
            self
        """
        expert_names = list(expert_preds_dict.keys())

        # --- Step 1: Initialization (Find best single model) ---
        best_single_name = None
        best_single_score = float("inf")

        for name, preds in expert_preds_dict.items():
            score = self._calculate_score(y_true, preds)
            if score < best_single_score:
                best_single_score = score
                best_single_name = name

        if self.verbose:
            print(
                f"Best single model: {best_single_name} with Log Loss: {best_single_score:.15f}"
            )

        # Initialize ensemble with the best single model
        self.selected_experts = [best_single_name]
        self.weights = {best_single_name: 1}
        self.best_score = best_single_score

        # Maintain the running sum of predictions to avoid re-summing large arrays repeatedly
        # Ensemble Prediction = current_ensemble_sum / current_k
        current_ensemble_sum = expert_preds_dict[best_single_name].copy()
        current_k = 1  # Total number of models in the ensemble (sum of weights)

        # --- Step 2: Iterative Selection ---
        for i in range(self.max_iter):
            best_iter_score = float("inf")
            best_iter_name = None

            # Try adding each expert from the library (with replacement)
            for name in expert_names:
                candidate_preds = expert_preds_dict[name]

                # Calculate potential new ensemble average
                # New Avg = (Sum_Current + Candidate) / (k + 1)
                temp_ensemble_preds = (current_ensemble_sum + candidate_preds) / (
                    current_k + 1
                )

                score = self._calculate_score(y_true, temp_ensemble_preds)

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_name = name

            # --- Step 3: Check for Improvement ---
            improvement = self.best_score - best_iter_score

            if improvement > self.tol:
                self.best_score = best_iter_score
                self.selected_experts.append(best_iter_name)
                self.weights[best_iter_name] = self.weights.get(best_iter_name, 0) + 1

                # Update state
                current_ensemble_sum += expert_preds_dict[best_iter_name]
                current_k += 1

                if self.verbose:
                    print(
                        f"Iter {i+1}: Added {best_iter_name}. New Log Loss: {self.best_score:.15f}"
                    )
            else:
                if self.verbose:
                    print(
                        f"Iter {i+1}: No significant improvement (Best candidate: {best_iter_name}, Score: {best_iter_score:.15f}). Stopping."
                    )
                break

        if self.verbose:
            print("-" * 30)
            print("Final Selection Weights:")
            for name, weight in self.weights.items():
                print(f"  {name}: {weight}")
            print(f"Final Validation Log Loss: {self.best_score:.15f}")
            print("-" * 30)

        return self

    def predict(self, expert_preds_dict):
        """
        Aggregates predictions from multiple experts using the fitted weights.

        Args:
            expert_preds_dict (dict): Dictionary {expert_name: np.ndarray(N, n_classes)}.
                                      Contains test probabilities for each expert.

        Returns:
            np.ndarray: Weighted average probabilities (N, n_classes).
        """
        if not self.weights:
            raise ValueError("Selector has not been fitted yet.")

        final_preds = None
        total_weight = 0

        for name, weight in self.weights.items():
            if name not in expert_preds_dict:
                raise ValueError(
                    f"Selected expert '{name}' not found in provided predictions dictionary."
                )

            preds = expert_preds_dict[name]

            if final_preds is None:
                final_preds = np.zeros_like(preds)

            final_preds += preds * weight
            total_weight += weight

        if total_weight == 0:
            raise ValueError("Total weight is zero.")

        return final_preds / total_weight
