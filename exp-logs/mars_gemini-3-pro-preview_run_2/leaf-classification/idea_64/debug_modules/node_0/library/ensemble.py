import numpy as np
from collections import Counter
from library.utils import clipped_log_loss


class GreedyForwardSelector:
    """
    Implements Greedy Forward Selection for ensemble optimization.

    This strategy iteratively adds models to the ensemble that maximize the
    improvement in the validation metric (Log Loss). It allows for replacement,
    meaning a model can be selected multiple times, effectively assigning it
    a higher integer weight.
    """

    def __init__(self, max_ensemble_size=50, tol=1e-6, verbose=True):
        """
        Args:
            max_ensemble_size (int): Maximum number of models to include in the ensemble.
            tol (float): Minimum improvement required to continue adding models.
            verbose (bool): Whether to print progress during fitting.
        """
        self.max_ensemble_size = max_ensemble_size
        self.tol = tol
        self.verbose = verbose
        self.selected_experts_ = []
        self.weights_ = {}
        self.best_score_ = float("inf")

    def _compute_ensemble_pred(self, current_selection, y_preds_dict):
        """
        Computes the averaged prediction for a specific list of expert names.

        Args:
            current_selection (list): List of expert names (strings).
            y_preds_dict (dict): Dictionary mapping expert names to probability arrays.

        Returns:
            np.ndarray: Averaged probability matrix.
        """
        if not current_selection:
            return None

        # Sum probabilities
        ensemble_sum = None
        for name in current_selection:
            pred = y_preds_dict[name]
            if ensemble_sum is None:
                ensemble_sum = pred.copy()
            else:
                ensemble_sum += pred

        # Average
        return ensemble_sum / len(current_selection)

    def fit(self, y_preds_dict, y_true):
        """
        Fits the ensemble selection logic using validation predictions.

        Args:
            y_preds_dict (dict): Dictionary where keys are expert names and values
                                 are np.ndarrays of shape (n_samples, n_classes).
            y_true (np.ndarray): True labels for the validation set.

        Returns:
            self
        """
        available_experts = list(y_preds_dict.keys())
        current_selection = []
        best_score = float("inf")

        # Initial baseline: Find single best model
        if self.verbose:
            print(
                f"Starting Greedy Forward Selection (Max Size: {self.max_ensemble_size})..."
            )

        for _ in range(self.max_ensemble_size):
            best_candidate = None
            best_candidate_score = float("inf")

            # Try adding each available expert to the current selection
            for expert in available_experts:
                trial_selection = current_selection + [expert]

                # Compute ensemble prediction
                y_pred_trial = self._compute_ensemble_pred(
                    trial_selection, y_preds_dict
                )

                # Evaluate
                score = clipped_log_loss(y_true, y_pred_trial)

                if score < best_candidate_score:
                    best_candidate_score = score
                    best_candidate = expert

            # Check for improvement
            improvement = best_score - best_candidate_score

            if improvement > self.tol:
                current_selection.append(best_candidate)
                best_score = best_candidate_score
                if self.verbose:
                    print(
                        f"Round {len(current_selection)}: Added '{best_candidate}'. "
                        f"New Score: {best_score}"
                    )
            else:
                if self.verbose:
                    print(
                        f"Stopping early. Improvement {improvement} < tolerance {self.tol}."
                    )
                break

        self.selected_experts_ = current_selection
        self.best_score_ = best_score

        # Calculate weights (counts) for efficient prediction
        self.weights_ = dict(Counter(self.selected_experts_))

        if self.verbose:
            print("Selection Complete.")
            print(f"Final Ensemble Size: {len(self.selected_experts_)}")
            print(f"Final Validation Score: {self.best_score_}")
            print(f"Weights: {self.weights_}")

        return self

    def predict(self, y_preds_dict):
        """
        Computes the weighted average prediction using the fitted ensemble weights.

        Args:
            y_preds_dict (dict): Dictionary where keys are expert names and values
                                 are np.ndarrays of shape (n_samples, n_classes).

        Returns:
            np.ndarray: The final ensemble probability matrix.
        """
        if not self.selected_experts_:
            raise RuntimeError("Ensemble must be fitted before calling predict.")

        ensemble_sum = None
        total_weight = 0

        for name, weight in self.weights_.items():
            if name not in y_preds_dict:
                raise KeyError(
                    f"Expert '{name}' selected during fit not found in prediction dictionary."
                )

            pred = y_preds_dict[name]
            weighted_pred = pred * weight

            if ensemble_sum is None:
                ensemble_sum = weighted_pred
            else:
                ensemble_sum += weighted_pred

            total_weight += weight

        return ensemble_sum / total_weight
