import numpy as np
from sklearn.metrics import log_loss
from collections import Counter


class GreedyEnsembleSelector:
    """
    Implements Greedy Forward Selection for Ensemble Optimization.

    This class iteratively builds an ensemble by adding the model that maximally
    reduces the log loss of the ensemble's weighted average prediction at each step.
    This results in a set of integer weights for the selected models.
    """

    def __init__(self, max_iterations=100, tolerance=1e-6, random_state=42):
        """
        Args:
            max_iterations (int): Maximum number of models to add to the ensemble.
            tolerance (float): Minimum improvement in log loss required to continue adding models.
            random_state (int): Seed for reproducibility (unused in deterministic greedy selection
                                but kept for API consistency).
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.weights_ = {}
        self.selected_models_ = []
        self.best_score_ = float("inf")
        self.n_classes_ = None

    def fit(self, predictions_dict, y_true):
        """
        Fits the ensemble weights using greedy forward selection on validation data.

        Args:
            predictions_dict (dict): Dictionary where keys are model names and values are
                                     numpy arrays of shape (n_samples, n_classes) containing
                                     predicted probabilities.
            y_true (array-like): True class labels for the validation set.

        Returns:
            self: Returns the instance itself.
        """
        # Get list of available models
        available_models = sorted(list(predictions_dict.keys()))
        if not available_models:
            raise ValueError("predictions_dict cannot be empty.")

        # Infer dimensions
        n_samples, self.n_classes_ = predictions_dict[available_models[0]].shape

        # Initialize ensemble sum accumulator
        current_sum_preds = np.zeros((n_samples, self.n_classes_))
        self.selected_models_ = []
        best_loss_global = float("inf")

        # Define labels for log_loss to ensure all classes are accounted for
        labels = np.arange(self.n_classes_)

        print(
            f"Starting Greedy Forward Selection with {len(available_models)} candidate models."
        )

        for i in range(1, self.max_iterations + 1):
            iteration_best_loss = float("inf")
            iteration_best_model = None

            # Try adding each available model to the current ensemble
            for model in available_models:
                preds = predictions_dict[model]

                # Calculate trial ensemble predictions
                # Formula: (Sum of previous selected + Current Candidate) / (Number of selected + 1)
                trial_preds = (current_sum_preds + preds) / i

                # Calculate Log Loss
                # sklearn.metrics.log_loss handles clipping internally (eps=1e-15)
                loss = log_loss(y_true, trial_preds, labels=labels)

                if loss < iteration_best_loss:
                    iteration_best_loss = loss
                    iteration_best_model = model

            # Determine if we should keep this addition
            if i == 1:
                # Always accept the first model
                best_loss_global = iteration_best_loss
                self.selected_models_.append(iteration_best_model)
                current_sum_preds += predictions_dict[iteration_best_model]
                print(
                    f"Iteration {i}: Added {iteration_best_model} | Log Loss: {best_loss_global}"
                )
            else:
                # Check for significant improvement
                if (best_loss_global - iteration_best_loss) > self.tolerance:
                    best_loss_global = iteration_best_loss
                    self.selected_models_.append(iteration_best_model)
                    current_sum_preds += predictions_dict[iteration_best_model]
                    print(
                        f"Iteration {i}: Added {iteration_best_model} | Log Loss: {best_loss_global}"
                    )
                else:
                    print(
                        f"Iteration {i}: No significant improvement (New: {iteration_best_loss}, Old: {best_loss_global}). Stopping."
                    )
                    break

        self.best_score_ = best_loss_global

        # Calculate final integer weights based on selection frequency
        counts = Counter(self.selected_models_)
        self.weights_ = dict(counts)

        print("Ensemble Selection Complete.")
        print(f"Final Weights: {self.weights_}")

        return self

    def predict(self, predictions_dict):
        """
        Generates weighted average predictions using the fitted ensemble weights.

        Args:
            predictions_dict (dict): Dictionary where keys are model names and values are
                                     numpy arrays of shape (n_samples, n_classes) containing
                                     predicted probabilities.

        Returns:
            np.array: Weighted average probabilities of shape (n_samples, n_classes).
        """
        if not self.weights_:
            raise RuntimeError("Selector is not fitted. Call fit() first.")

        # Validate input and initialize accumulator
        first_model_key = next(iter(self.weights_))
        if first_model_key not in predictions_dict:
            raise KeyError(
                f"Model {first_model_key} required by ensemble but not found in input."
            )

        n_samples, n_classes = predictions_dict[first_model_key].shape
        weighted_sum = np.zeros((n_samples, n_classes))
        total_weight = 0

        # Aggregate predictions
        for model, weight in self.weights_.items():
            if model not in predictions_dict:
                raise KeyError(
                    f"Model {model} required by ensemble but not found in input."
                )

            weighted_sum += predictions_dict[model] * weight
            total_weight += weight

        if total_weight == 0:
            raise RuntimeError("Total ensemble weight is zero.")

        # Compute average
        final_preds = weighted_sum / total_weight

        return final_preds
