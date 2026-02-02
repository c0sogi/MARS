import numpy as np
from sklearn.metrics import log_loss
from library.config import FLOAT_PRECISION


class GreedySelector:
    """
    Implements a Greedy Forward Selection strategy (Hill Climbing) with replacement
    to optimize ensemble weights by minimizing Log Loss.
    """

    def __init__(self, iterations=50):
        """
        Args:
            iterations (int): Maximum number of iterations for the greedy selection process.
        """
        self.iterations = iterations
        self.weights = {}
        self.selected_experts = []
        self.best_score = float("inf")

    def fit(self, predictions, y_true):
        """
        Runs the greedy forward selection process to determine optimal expert weights.

        Args:
            predictions (dict): Dictionary mapping expert names to probability matrices
                                of shape (n_samples, n_classes).
            y_true (np.array): True class labels (n_samples,).

        Returns:
            dict: A dictionary mapping expert names to their calculated weights.
        """
        # Filter out invalid or None predictions
        valid_experts = [k for k, v in predictions.items() if v is not None]
        if not valid_experts:
            raise ValueError("No valid predictions provided for selection.")

        # Ensure y_true is a numpy array
        y_true = np.array(y_true)

        # Determine shapes from the first valid prediction
        first_pred = predictions[valid_experts[0]]
        n_samples, n_classes = first_pred.shape

        # Initialize ensemble accumulator
        current_ensemble_sum = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)
        ensemble_count = 0
        self.selected_experts = []
        self.best_score = float("inf")

        print("Starting Greedy Forward Selection...")

        # Step 1: Find the single best expert to initialize the ensemble
        best_single_expert = None

        for name in valid_experts:
            # Clip probabilities to avoid log(0)
            probs = np.clip(predictions[name], 1e-15, 1 - 1e-15)
            score = log_loss(y_true, probs)

            if score < self.best_score:
                self.best_score = score
                best_single_expert = name

        if best_single_expert:
            self.selected_experts.append(best_single_expert)
            current_ensemble_sum += predictions[best_single_expert]
            ensemble_count += 1
            print(f"Iter 1: Added {best_single_expert} (Score: {self.best_score})")
        else:
            raise ValueError("Could not find a valid starting expert.")

        # Step 2: Iterative Hill Climbing
        # Try adding each expert to the current ensemble and keep the one that minimizes loss
        for i in range(self.iterations - 1):
            best_iter_expert = None
            best_iter_score = self.best_score

            for name in valid_experts:
                # Calculate potential new ensemble average
                # New Average = (Current Sum + Candidate Prediction) / (Current Count + 1)
                temp_sum = current_ensemble_sum + predictions[name]
                temp_prob = temp_sum / (ensemble_count + 1)

                # Clip for numerical stability
                temp_prob = np.clip(temp_prob, 1e-15, 1 - 1e-15)

                score = log_loss(y_true, temp_prob)

                # Check for improvement
                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_expert = name

            if best_iter_expert:
                self.selected_experts.append(best_iter_expert)
                current_ensemble_sum += predictions[best_iter_expert]
                ensemble_count += 1
                self.best_score = best_iter_score
                print(
                    f"Iter {i+2}: Added {best_iter_expert} (Score: {self.best_score})"
                )
            else:
                print("No improvement found. Stopping selection.")
                break

        # Step 3: Compute Weights based on selection frequency
        total_selected = len(self.selected_experts)
        self.weights = {}
        for name in self.selected_experts:
            self.weights[name] = self.weights.get(name, 0) + (1.0 / total_selected)

        return self.weights

    def predict(self, predictions):
        """
        Computes the weighted average of expert predictions using the fitted weights.

        Args:
            predictions (dict): Dictionary mapping expert names to probability matrices.

        Returns:
            np.array: The aggregated weighted probability matrix.
        """
        if not self.weights:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")

        # Get dimensions from the first expert in the weights list
        first_expert_name = next(iter(self.weights))
        if first_expert_name not in predictions:
            raise KeyError(
                f"Expert {first_expert_name} not found in predictions dictionary."
            )

        n_samples, n_classes = predictions[first_expert_name].shape
        final_proba = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)

        for name, weight in self.weights.items():
            if name not in predictions:
                raise KeyError(
                    f"Selected expert '{name}' not found in prediction dictionary."
                )

            probs = predictions[name]
            # Ensure probabilities are clipped and correct precision
            probs = np.clip(probs, 1e-15, 1 - 1e-15).astype(FLOAT_PRECISION)

            final_proba += probs * weight

        return final_proba
