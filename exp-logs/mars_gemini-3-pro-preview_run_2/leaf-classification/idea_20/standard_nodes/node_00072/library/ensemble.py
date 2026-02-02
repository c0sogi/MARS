import numpy as np
from library.utils import calculate_metric


class GreedySelector:
    """
    Implements Greedy Forward Selection to identify the optimal subset of experts
    for the Gaussianized Metric-Optimized Dynamic Ensemble.
    """

    def __init__(self, tolerance=1e-6):
        """
        Args:
            tolerance (float): The minimum improvement in Log Loss required to
                               add a candidate expert to the ensemble.
        """
        self.selected_experts = []
        self.best_score = float("inf")
        self.tolerance = tolerance

    def fit(self, expert_preds, y_true, classes=None):
        """
        Fits the selector by determining the subset of experts that minimizes
        validation Log Loss.

        Args:
            expert_preds (dict): Dictionary mapping expert names (str) to
                                 prediction arrays of shape (n_samples, n_classes).
            y_true (np.ndarray): Array of true class indices of shape (n_samples,).
            classes (list, optional): List of class labels for metric calculation.
        """
        # Sort keys for deterministic behavior
        available_experts = sorted(list(expert_preds.keys()))

        current_ensemble = []
        best_ensemble_score = float("inf")

        print("Starting Greedy Forward Selection...")

        # Step 1: Initialize with the single best expert
        best_single_expert = None

        for name in available_experts:
            score = calculate_metric(y_true, expert_preds[name], classes=classes)
            print(f"  Expert: {name}, Score: {score:.15f}")

            if score < best_ensemble_score:
                best_ensemble_score = score
                best_single_expert = name

        if best_single_expert is None:
            raise ValueError("No experts provided or scoring failed.")

        current_ensemble.append(best_single_expert)
        available_experts.remove(best_single_expert)
        self.best_score = best_ensemble_score

        print(f"Initialized with: {best_single_expert}, Score: {self.best_score:.15f}")

        # Step 2: Iteratively add experts that improve the score
        while available_experts:
            best_candidate = None
            best_candidate_score = self.best_score

            for name in available_experts:
                # Form a temporary ensemble with the candidate
                temp_ensemble = current_ensemble + [name]

                # Calculate ensemble prediction (simple average)
                # Stack shape: (n_models, n_samples, n_classes)
                preds_stack = np.array([expert_preds[e] for e in temp_ensemble])
                ensemble_pred = np.mean(preds_stack, axis=0)

                score = calculate_metric(y_true, ensemble_pred, classes=classes)

                # Check if this candidate improves the score beyond tolerance
                if score < best_candidate_score - self.tolerance:
                    best_candidate_score = score
                    best_candidate = name

            if best_candidate:
                print(
                    f"Adding {best_candidate}, New Score: {best_candidate_score:.15f}"
                )
                current_ensemble.append(best_candidate)
                available_experts.remove(best_candidate)
                self.best_score = best_candidate_score
            else:
                print("No further improvement found. Stopping selection.")
                break

        self.selected_experts = current_ensemble
        print(f"Selection Complete. Final Experts: {self.selected_experts}")
        print(f"Final Validation Score: {self.best_score:.15f}")

    def predict(self, expert_preds):
        """
        Computes the ensemble prediction using the selected experts.

        Args:
            expert_preds (dict): Dictionary mapping expert names to prediction arrays.
                                 Must contain all keys in self.selected_experts.

        Returns:
            np.ndarray: Averaged probability predictions of shape (n_samples, n_classes).
        """
        if not self.selected_experts:
            raise ValueError("Selector has not been fitted yet.")

        # Check for missing experts
        missing = [e for e in self.selected_experts if e not in expert_preds]
        if missing:
            raise KeyError(f"Missing predictions for selected experts: {missing}")

        # Stack predictions from selected experts and compute mean
        preds_stack = np.array([expert_preds[name] for name in self.selected_experts])
        ensemble_pred = np.mean(preds_stack, axis=0)

        return ensemble_pred
