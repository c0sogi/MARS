import numpy as np
from sklearn.metrics import log_loss
from collections import Counter
from library import config


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement to optimize Multi-class Log Loss.

    This selector builds an ensemble by iteratively adding the expert that maximizes
    the validation metric (minimizes Log Loss) when averaged with the currently
    selected experts. This implicitly learns integer weights for the experts.
    """

    def __init__(self, max_experts=50, tolerance=1e-6, verbose=True):
        """
        Args:
            max_experts (int): Maximum number of experts to select (iterations).
            tolerance (float): Minimum improvement required to continue selection.
            verbose (bool): Whether to print progress.
        """
        self.max_experts = max_experts
        self.tolerance = tolerance
        self.verbose = verbose
        self.selected_experts = []  # List of expert names (can contain duplicates)
        self.best_score = float("inf")

    def _compute_metric(self, y_true, y_pred):
        """
        Computes the Multi-class Log Loss with specific clipping as per task description.

        Rules:
        1. Rescale rows to sum to 1.
        2. Clip probabilities to [1e-15, 1-1e-15].
        3. Compute Log Loss.
        """
        # 1. Normalize (rescale prior to scoring)
        row_sums = y_pred.sum(axis=1, keepdims=True)
        # Handle potential zero sums (though unlikely with proper experts)
        row_sums[row_sums == 0] = 1.0
        y_pred_norm = y_pred / row_sums

        # 2. Clip to avoid extremes of log function
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred_norm, eps, 1 - eps)

        # 3. Compute Log Loss
        return log_loss(y_true, y_pred_clipped, labels=list(range(y_pred.shape[1])))

    def fit(self, predictions_dict, y_true):
        """
        Runs the greedy forward selection process on validation data.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to probability matrices (N_samples, N_classes).
            y_true (np.array): True class indices (N_samples,).

        Returns:
            self
        """
        expert_names = list(predictions_dict.keys())
        if not expert_names:
            raise ValueError("predictions_dict cannot be empty.")

        # Ensure input precision
        for k, v in predictions_dict.items():
            predictions_dict[k] = v.astype(config.FLOAT_PRECISION)

        # Initialize ensemble prediction sum
        # We maintain the running sum to avoid re-summing the entire history every iteration.
        # Ensemble Pred at step k = current_sum / k
        sample_shape = list(predictions_dict.values())[0].shape
        current_sum = np.zeros(sample_shape, dtype=config.FLOAT_PRECISION)

        self.selected_experts = []
        self.best_score = float("inf")

        if self.verbose:
            print(f"Starting Greedy Selection with {len(expert_names)} candidates...")

        # Iteratively add experts
        for k in range(1, self.max_experts + 1):
            best_iter_score = float("inf")
            best_iter_expert = None

            # Try adding each candidate to the current ensemble
            for name in expert_names:
                candidate_preds = predictions_dict[name]

                # Calculate trial ensemble prediction
                trial_sum = current_sum + candidate_preds
                trial_preds = trial_sum / k

                score = self._compute_metric(y_true, trial_preds)

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_expert = name

            # Check for improvement
            improvement = self.best_score - best_iter_score

            if improvement > self.tolerance:
                self.best_score = best_iter_score
                self.selected_experts.append(best_iter_expert)
                current_sum += predictions_dict[best_iter_expert]

                if self.verbose:
                    print(
                        f"Step {k}: Added '{best_iter_expert}'. Score: {self.best_score:.15f} (Imp: {improvement:.1e})"
                    )
            else:
                if self.verbose:
                    print(
                        f"Selection stopped at step {k}. Improvement {improvement:.1e} < tolerance {self.tolerance}"
                    )
                break

        if self.verbose:
            print(f"Selection Complete. Ensemble size: {len(self.selected_experts)}")
            print(f"Final Validation Log Loss: {self.best_score:.15f}")

        return self

    def predict(self, predictions_dict):
        """
        Computes the weighted average prediction using the selected experts.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to probability matrices.
                                     Must contain keys for all selected experts.

        Returns:
            np.ndarray: Aggregated probability matrix (N_samples, N_classes).
        """
        if not self.selected_experts:
            raise ValueError(
                "Selector has not been fitted or no experts were selected."
            )

        # Initialize sum with the first selected expert
        first_expert = self.selected_experts[0]
        if first_expert not in predictions_dict:
            raise KeyError(
                f"Selected expert '{first_expert}' not found in predictions_dict."
            )

        ensemble_sum = (
            predictions_dict[first_expert].astype(config.FLOAT_PRECISION).copy()
        )

        # Add the rest
        for name in self.selected_experts[1:]:
            if name not in predictions_dict:
                raise KeyError(
                    f"Selected expert '{name}' not found in predictions_dict."
                )
            ensemble_sum += predictions_dict[name].astype(config.FLOAT_PRECISION)

        # Compute Average
        ensemble_avg = ensemble_sum / len(self.selected_experts)

        return ensemble_avg

    def get_weights(self):
        """
        Returns the calculated weights of the selected experts.

        Returns:
            dict: {expert_name: weight} where weight is the proportion of times
                  the expert was selected.
        """
        counts = Counter(self.selected_experts)
        total = len(self.selected_experts)
        return {k: v / total for k, v in counts.items()}
