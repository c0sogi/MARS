import numpy as np
from sklearn.preprocessing import LabelEncoder
from library.config import Config
from library.utils import log_loss_score


class GreedySelector:
    """
    Implements Greedy Forward Selection for the Discriminative-Manifold Generative Ensemble.
    Manages the selection, retraining, and prediction of expert models.
    """

    def __init__(
        self,
        experts,
        max_iter=Config.SELECTION_MAX_ITER,
        tolerance=Config.SELECTION_TOLERANCE,
    ):
        """
        Args:
            experts (list): List of ExpertDefinition objects.
            max_iter (int): Maximum number of experts to add to the ensemble.
            tolerance (float): Minimum log loss improvement required to add an expert.
        """
        self.experts = experts
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.selected_experts = (
            []
        )  # List of ExpertDefinition objects chosen for the ensemble
        self.fitted_models = (
            []
        )  # List of fitted sklearn pipelines corresponding to selected_experts
        self.best_score = float("inf")
        self.le = LabelEncoder()

    def _get_data(self, expert, data_dict):
        """
        Retrieves the specific data view required by the expert.

        Args:
            expert (ExpertDefinition): The expert definition.
            data_dict (dict): Dictionary containing 'X_global' and 'X_morph'.

        Returns:
            np.ndarray: The feature matrix.
        """
        if expert.view_name not in data_dict:
            raise KeyError(
                f"Data view '{expert.view_name}' not found in provided data dictionary."
            )
        return data_dict[expert.view_name]

    def fit(self, data_train, y_train, data_val, y_val):
        """
        Phase 1: Selection.
        Trains all candidates on the training split, predicts on validation split,
        and runs greedy forward selection to determine the optimal ensemble composition.

        Args:
            data_train (dict): Dictionary of training feature views.
            y_train (np.ndarray): Training labels (strings).
            data_val (dict): Dictionary of validation feature views.
            y_val (np.ndarray): Validation labels (strings).
        """
        # Encode labels for metric calculation
        # We fit LE on y_train to ensure consistency with how LDA sees classes.
        # LDA sorts classes alphabetically, and LabelEncoder does the same.
        self.le.fit(y_train)
        y_val_enc = self.le.transform(y_val)

        print(f"Phase 1: Training {len(self.experts)} candidate experts...")

        # 1. Train all candidates and get validation predictions
        val_preds_pool = []

        for i, expert in enumerate(self.experts):
            X_train = self._get_data(expert, data_train)
            X_val = self._get_data(expert, data_val)

            # Build and train pipeline
            pipeline = expert.build_pipeline()
            pipeline.fit(X_train, y_train)

            # Predict probabilities
            # Output shape: (n_samples, n_classes)
            preds = pipeline.predict_proba(X_val).astype(Config.FLOAT_TYPE)
            val_preds_pool.append(preds)

        val_preds_pool = np.array(val_preds_pool)

        # 2. Greedy Forward Selection
        print("Phase 1: Running Greedy Forward Selection...")

        # Initialize ensemble state
        # current_sum accumulates the probability matrices of selected experts
        current_sum = np.zeros_like(val_preds_pool[0])
        best_loss = float("inf")

        # Iteratively add experts
        for k in range(1, self.max_iter + 1):
            iter_best_loss = float("inf")
            iter_best_idx = -1

            # Try adding each candidate from the pool to the current ensemble
            for i in range(len(self.experts)):
                # Calculate trial ensemble prediction: (Current Sum + Candidate) / (Current Count + 1)
                trial_pred = (current_sum + val_preds_pool[i]) / k

                # Calculate score
                loss = log_loss_score(y_val_enc, trial_pred)

                if loss < iter_best_loss:
                    iter_best_loss = loss
                    iter_best_idx = i

            # Check for improvement
            improvement = best_loss - iter_best_loss

            if improvement > self.tolerance:
                best_loss = iter_best_loss

                # Add best candidate to selection
                selected_expert = self.experts[iter_best_idx]
                self.selected_experts.append(selected_expert)

                # Update ensemble state
                current_sum += val_preds_pool[iter_best_idx]

                print(
                    f"Iter {k}: Added '{selected_expert.name}'. Validation Log Loss: {best_loss}"
                )
            else:
                print(
                    f"Iter {k}: Improvement ({improvement:.9f}) <= Tolerance ({self.tolerance}). Stopping."
                )
                break

        self.best_score = best_loss
        print(f"Selection Complete. Ensemble Size: {len(self.selected_experts)}")
        print(f"Final Validation Log Loss: {self.best_score}")

    def refit(self, data_full, y_full):
        """
        Phase 2: Final Retraining.
        Retrains only the selected experts on the full dataset (Train + Val).

        Args:
            data_full (dict): Dictionary of full feature views.
            y_full (np.ndarray): Full target labels.
        """
        if not self.selected_experts:
            print("Warning: No experts selected. Skipping refit.")
            return

        print(
            f"Phase 2: Refitting {len(self.selected_experts)} selected experts on full data..."
        )
        self.fitted_models = []

        for i, expert in enumerate(self.selected_experts):
            print(f"Refitting expert {i+1}/{len(self.selected_experts)}: {expert.name}")
            X_full = self._get_data(expert, data_full)

            pipeline = expert.build_pipeline()
            pipeline.fit(X_full, y_full)
            self.fitted_models.append(pipeline)

    def predict(self, data_test):
        """
        Generates predictions using the refitted ensemble.

        Args:
            data_test (dict): Dictionary of test feature views.

        Returns:
            np.ndarray: Aggregated probability matrix (n_samples, n_classes).
        """
        if not self.fitted_models:
            raise RuntimeError("Ensemble models not fitted. Call refit() first.")

        current_sum = None

        # Aggregate predictions from all fitted models
        for expert, model in zip(self.selected_experts, self.fitted_models):
            X_test = self._get_data(expert, data_test)
            preds = model.predict_proba(X_test).astype(Config.FLOAT_TYPE)

            if current_sum is None:
                current_sum = preds
            else:
                current_sum += preds

        # Compute arithmetic mean
        final_preds = current_sum / len(self.fitted_models)
        return final_preds
