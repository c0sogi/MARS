import numpy as np
from sklearn.metrics import log_loss
from library.config import Config


class EnsembleSelector:
    """
    Implements Greedy Forward Selection for the Precision-Covariance Multi-Resolution Ensemble.
    Manages the lifecycle of expert training, selection, and final retraining.
    """

    def __init__(self, experts):
        """
        Args:
            experts (list[Expert]): List of initialized Expert objects from the ModelFactory.
        """
        self.experts = experts
        self.expert_map = {e.name: e for e in experts}
        self.selected_experts = (
            []
        )  # List of names, allowing duplicates (which act as integer weights)
        self.n_iterations = Config.SELECTION_ITERATIONS

    def _get_view_data(self, dataset, view_name):
        """
        Helper to extract the correct feature matrix based on the expert's view configuration.
        """
        if view_name == "global":
            return dataset["X_global"]
        elif view_name == "macro":
            return dataset["X_macro"]
        elif view_name == "combined":
            return dataset["X_combined"]
        else:
            raise ValueError(f"Unknown view type: {view_name}")

    def fit(self, data):
        """
        Performs the Selection Phase:
        1. Trains all candidate experts on the Training split.
        2. Generates predictions on the Validation split.
        3. Runs Greedy Forward Selection to determine the optimal ensemble composition.
        """
        print(f"Training {len(self.experts)} candidate experts on training split...")

        # Cache validation predictions for all experts to speed up the greedy loop
        val_preds_cache = {}
        y_train = data["train"]["y"]
        y_val = data["val"]["y"]
        n_classes = len(data["classes"])
        # Classes are encoded 0..N-1
        labels = np.arange(n_classes)

        for expert in self.experts:
            # Get specific view for this expert (Global, Macro, or Combined)
            X_train = self._get_view_data(data["train"], expert.view)
            X_val = self._get_view_data(data["val"], expert.view)

            # Train on 80% split
            expert.model.fit(X_train, y_train)

            # Predict on 20% holdout (ensure float64 precision)
            preds = expert.model.predict_proba(X_val).astype(Config.FLOAT_TYPE)
            val_preds_cache[expert.name] = preds

        print(f"Starting Greedy Forward Selection ({self.n_iterations} iterations)...")

        # Initialize ensemble accumulator
        # We accumulate the sum of probabilities to avoid re-averaging the entire list every iteration
        current_ensemble_sum = np.zeros(
            (len(y_val), n_classes), dtype=Config.FLOAT_TYPE
        )

        best_overall_loss = float("inf")

        for i in range(self.n_iterations):
            iter_best_loss = float("inf")
            iter_best_expert_name = None

            # Try adding each available expert to the current ensemble
            for expert in self.experts:
                # Calculate candidate ensemble average
                # New Avg = (Current Sum + Candidate Preds) / (Current Count + 1)
                candidate_sum = current_ensemble_sum + val_preds_cache[expert.name]
                candidate_avg = candidate_sum / (i + 1)

                # Clip probabilities to avoid log(0) and match metric constraints
                candidate_avg = np.clip(candidate_avg, 1e-15, 1 - 1e-15)

                # Calculate Log Loss
                loss = log_loss(y_val, candidate_avg, labels=labels)

                if loss < iter_best_loss:
                    iter_best_loss = loss
                    iter_best_expert_name = expert.name

            # Update Ensemble with the winner of this iteration
            self.selected_experts.append(iter_best_expert_name)
            current_ensemble_sum += val_preds_cache[iter_best_expert_name]
            best_overall_loss = iter_best_loss

            print(
                f"Iteration {i+1}/{self.n_iterations}: Added {iter_best_expert_name}, Val Log Loss: {best_overall_loss:.15f}"
            )

        print("Selection complete.")
        print(f"Selected Ensemble Composition: {self.selected_experts}")

    def refit_and_predict(self, data):
        """
        Performs the Final Retraining & Inference Phase:
        1. Combines Train and Val data into a full dataset.
        2. Retrains ONLY the unique experts selected in the fit phase.
        3. Predicts on the Test data.
        4. Aggregates predictions based on the selection counts (weights).
        """
        if not self.selected_experts:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        print("Preparing for final retraining on full dataset...")

        # 1. Combine Data (Train + Val) for all views to maximize signal
        combined_X = {}
        for view in ["global", "macro", "combined"]:
            combined_X[view] = np.vstack(
                [data["train"][f"X_{view}"], data["val"][f"X_{view}"]]
            )

        combined_y = np.concatenate([data["train"]["y"], data["val"]["y"]])

        # 2. Identify unique experts to retrain
        # We only need to retrain each unique model configuration once, even if selected multiple times
        unique_expert_names = set(self.selected_experts)
        unique_experts = [self.expert_map[name] for name in unique_expert_names]

        test_preds_cache = {}

        print(
            f"Retraining {len(unique_experts)} unique experts on Combined (Train + Val) data..."
        )

        for expert in unique_experts:
            # Get combined view
            X_all = combined_X[expert.view]

            # Retrain on full data
            expert.model.fit(X_all, combined_y)

            # Predict on Test
            X_test = self._get_view_data(data["test"], expert.view)
            preds = expert.model.predict_proba(X_test).astype(Config.FLOAT_TYPE)
            test_preds_cache[expert.name] = preds

        # 3. Aggregate Predictions
        print("Aggregating test predictions based on ensemble weights...")
        n_test_samples = len(data["test"]["ids"])
        n_classes = len(data["classes"])

        final_ensemble_sum = np.zeros(
            (n_test_samples, n_classes), dtype=Config.FLOAT_TYPE
        )

        # Sum up predictions based on how many times each expert was selected in the greedy loop
        for expert_name in self.selected_experts:
            final_ensemble_sum += test_preds_cache[expert_name]

        # Calculate Weighted Average
        final_probs = final_ensemble_sum / len(self.selected_experts)

        # 4. Apply Metric Clipping
        # "predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)"
        final_probs = np.clip(final_probs, 1e-15, 1 - 1e-15)

        return final_probs
