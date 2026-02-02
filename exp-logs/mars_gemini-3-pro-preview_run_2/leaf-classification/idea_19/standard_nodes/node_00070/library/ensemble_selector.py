import numpy as np
from sklearn.metrics import log_loss
from library.config import Config


class GreedySelector:
    """
    Implements a Greedy Forward Selection strategy for Dynamic Ensemble Selection.

    This class iteratively selects experts from a pool to add to an ensemble,
    optimizing for the Multi-class Log Loss metric on a validation set.
    It supports weighted ensembles by allowing the same expert to be selected
    multiple times.
    """

    def __init__(self, tolerance=1e-6, max_iterations=100):
        """
        Args:
            tolerance (float): Minimum improvement in log loss required to add an expert.
            max_iterations (int): Maximum number of selection rounds.
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.selected_experts = (
            []
        )  # List of names of selected experts (can contain duplicates)
        self.weights = {}  # Dictionary of {expert_name: weight}
        self.best_score = float("inf")

    def _calculate_log_loss(self, y_true, y_pred, labels=None):
        """
        Calculates the Log Loss with clipping as specified in the task.

        Args:
            y_true (np.array): True class indices or one-hot encoded labels.
            y_pred (np.array): Predicted probabilities.
            labels (np.array, optional): List of all distinct labels (classes) to ensure
                                         log_loss handles sparse validation sets correctly.

        Returns:
            float: The log loss score.
        """
        # Clip probabilities to avoid log(0) and adhere to task metric specifics
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        # Normalize rows to ensure they sum to 1
        row_sums = y_pred_clipped.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        y_pred_norm = y_pred_clipped / row_sums

        return log_loss(y_true, y_pred_norm, labels=labels)

    def fit(self, predictions_dict, y_true, labels=None):
        """
        Fits the ensemble weights based on validation performance.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to prediction arrays
                                     of shape (n_samples, n_classes).
            y_true (np.array): True target labels of shape (n_samples,).
            labels (np.array, optional): List of all distinct labels (classes) the models
                                         were trained on.

        Returns:
            self
        """
        available_experts = list(predictions_dict.keys())
        if not available_experts:
            raise ValueError("predictions_dict cannot be empty.")

        print(
            f"Starting Greedy Forward Selection with {len(available_experts)} experts..."
        )

        # Step 1: Find the single best expert to initialize the ensemble
        best_initial_expert = None
        best_initial_score = float("inf")

        for name, preds in predictions_dict.items():
            score = self._calculate_log_loss(y_true, preds, labels=labels)
            print(f"Expert '{name}' standalone Log Loss: {score}")
            if score < best_initial_score:
                best_initial_score = score
                best_initial_expert = name

        if best_initial_expert is None:
            raise ValueError("Failed to evaluate any experts.")

        # Initialize ensemble
        self.selected_experts = [best_initial_expert]
        current_ensemble_sum = predictions_dict[best_initial_expert].copy()
        current_count = 1
        self.best_score = best_initial_score

        print(
            f"Selected initial expert: {best_initial_expert} with score: {self.best_score}"
        )

        # Step 2: Iteratively add experts
        for i in range(1, self.max_iterations):
            best_iter_expert = None
            best_iter_score = self.best_score

            # Try adding each available expert to the current mix
            for name in available_experts:
                preds = predictions_dict[name]

                # Calculate potential new ensemble prediction
                # New Avg = (Current Sum + Candidate Preds) / (Current Count + 1)
                temp_sum = current_ensemble_sum + preds
                temp_avg = temp_sum / (current_count + 1)

                score = self._calculate_log_loss(y_true, temp_avg, labels=labels)

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_expert = name

            # Evaluate improvement
            improvement = self.best_score - best_iter_score

            if best_iter_expert and improvement > self.tolerance:
                self.selected_experts.append(best_iter_expert)
                current_ensemble_sum += predictions_dict[best_iter_expert]
                current_count += 1
                self.best_score = best_iter_score
                print(
                    f"Iteration {i}: Added '{best_iter_expert}'. New Score: {self.best_score}. Improvement: {improvement}"
                )
            else:
                print(
                    f"Iteration {i}: No sufficient improvement (Best potential: {best_iter_score}). Stopping."
                )
                break

        # Step 3: Calculate final weights
        total_selected = len(self.selected_experts)
        self.weights = {}
        for name in available_experts:
            count = self.selected_experts.count(name)
            if count > 0:
                self.weights[name] = count / total_selected

        print("Final Ensemble Weights:", self.weights)
        return self

    def predict(self, predictions_dict):
        """
        Generates aggregated predictions using the fitted ensemble weights.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to prediction arrays
                                     of shape (n_samples, n_classes).

        Returns:
            np.array: Weighted average probabilities of shape (n_samples, n_classes).
        """
        if not self.weights:
            raise ValueError("Selector has not been fitted. Call fit() first.")

        # Get shape from the first expert present in weights
        first_expert = list(self.weights.keys())[0]
        if first_expert not in predictions_dict:
            raise KeyError(
                f"Expert '{first_expert}' required by ensemble but missing in input."
            )

        n_samples, n_classes = predictions_dict[first_expert].shape
        final_preds = np.zeros((n_samples, n_classes), dtype=np.float64)

        # Weighted sum
        for name, weight in self.weights.items():
            if name not in predictions_dict:
                raise KeyError(
                    f"Expert '{name}' required by ensemble but missing in input."
                )
            final_preds += predictions_dict[name] * weight

        # Ensure numerical stability and range [0, 1]
        # Although mathematically it should sum to 1, floating point errors can occur.
        # We re-normalize.
        row_sums = final_preds.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        final_preds = final_preds / row_sums

        return final_preds
