import numpy as np
from library.utils import clip_and_score, set_seed


class GreedySelector:
    """
    Implements the Greedy Forward Selection algorithm (Caruana et al., 2004) to
    optimize ensemble weights by iteratively selecting models that minimize
    validation log loss.
    """

    def __init__(self, iterations=100, random_state=42):
        """
        Args:
            iterations (int): Number of iterations for the selection process.
            random_state (int): Seed for reproducibility.
        """
        self.iterations = iterations
        self.random_state = random_state
        self.weights_ = {}
        self.best_score_ = float("inf")
        self.selected_models_ = []

    def fit(self, predictions_dict, y_true):
        """
        Fits the ensemble weights using greedy forward selection.

        Args:
            predictions_dict (dict): Dictionary where keys are model names and values
                                     are numpy arrays of shape (n_samples, n_classes)
                                     containing validation probabilities.
            y_true (np.array): True labels for the validation set.

        Returns:
            self
        """
        set_seed(self.random_state)

        model_names = list(predictions_dict.keys())
        # Ensure inputs are numpy arrays
        for k in model_names:
            predictions_dict[k] = np.array(predictions_dict[k])

        n_samples, n_classes = predictions_dict[model_names[0]].shape

        # Initialize ensemble sum (unscaled)
        ensemble_sum = np.zeros((n_samples, n_classes))
        self.selected_models_ = []
        self.best_score_ = float("inf")

        # Greedy Forward Selection Loop
        # We start with an empty ensemble and add 'iterations' models
        for i in range(1, self.iterations + 1):
            iteration_best_loss = float("inf")
            iteration_best_model = None

            for name in model_names:
                preds = predictions_dict[name]

                # Calculate what the ensemble average would be if we added this model
                # Current size is i-1, adding 1 makes it i
                temp_ensemble_probs = (ensemble_sum + preds) / i

                # Calculate metric
                loss = clip_and_score(y_true, temp_ensemble_probs)

                if loss < iteration_best_loss:
                    iteration_best_loss = loss
                    iteration_best_model = name

            # Update the ensemble with the winner of this iteration
            self.selected_models_.append(iteration_best_model)
            ensemble_sum += predictions_dict[iteration_best_model]
            self.best_score_ = iteration_best_loss

        # Compute final weights based on selection frequency
        total_selected = len(self.selected_models_)
        self.weights_ = {
            name: self.selected_models_.count(name) / total_selected
            for name in model_names
        }

        print(
            f"Greedy Selection Complete. Best Validation Log Loss: {self.best_score_:.20f}"
        )
        return self

    def predict(self, predictions_dict):
        """
        Computes the weighted average prediction using the fitted weights.

        Args:
            predictions_dict (dict): Dictionary of predictions (e.g., on test set).

        Returns:
            np.array: Weighted probability predictions.
        """
        if not self.weights_:
            raise RuntimeError("GreedySelector has not been fitted yet.")

        first_model = list(predictions_dict.keys())[0]
        n_samples, n_classes = predictions_dict[first_model].shape

        weighted_sum = np.zeros((n_samples, n_classes))

        for name, weight in self.weights_.items():
            if weight > 0:
                weighted_sum += weight * np.array(predictions_dict[name])

        # Normalize rows to sum to 1
        row_sums = weighted_sum.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        final_probs = weighted_sum / row_sums[:, np.newaxis]

        return final_probs

    def get_weights(self):
        """
        Returns the calculated weights.
        """
        return self.weights_
