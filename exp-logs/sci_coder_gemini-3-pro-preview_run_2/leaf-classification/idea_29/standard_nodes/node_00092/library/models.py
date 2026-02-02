import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.base import clone
from collections import Counter
from library.utils import set_seed, clipped_log_loss


class ExpertFactory:
    """
    Generates the library of probabilistic experts spanning the covariance complexity spectrum.
    """

    @staticmethod
    def get_experts():
        experts = []

        # --- Tier 1: Diagonal Covariance (GaussianNB) ---
        # High Bias, Low Variance. Serves as a robust anchor.
        # Views: Global (192 feats), Combined (192 + Morph)
        smoothing_grid = [1e-9, 1e-5, 1e-3, 1e-1]
        for smooth in smoothing_grid:
            for view in ["global", "combined"]:
                experts.append(
                    {
                        "name": f"GNB_smooth{smooth}_{view}",
                        "model": GaussianNB(var_smoothing=smooth),
                        "view": view,
                    }
                )

        # --- Tier 2: Shared Covariance (LDA) ---
        # Models correlations but assumes homoscedasticity (shared shape).
        # Views: Global, Combined
        # Shrinkage: 'auto' (Ledoit-Wolf/OAS approximation) and fixed grid.
        shrinkage_grid = ["auto", 0.001, 0.01, 0.1, 0.5]
        for shrink in shrinkage_grid:
            for view in ["global", "combined"]:
                name_shrink = "LW" if shrink == "auto" else f"{shrink}"
                experts.append(
                    {
                        "name": f"LDA_shrink{name_shrink}_{view}",
                        "model": LinearDiscriminantAnalysis(
                            solver="lsqr", shrinkage=shrink
                        ),
                        "view": view,
                    }
                )

        # --- Tier 3: Class-Specific Covariance (QDA) ---
        # Models heteroscedasticity (class-specific shapes). High Variance.
        # View: Macro ONLY (Low dimensionality ~12 feats allows QDA without overfitting).
        reg_grid = [0.0, 0.1, 0.5, 0.9]
        for reg in reg_grid:
            experts.append(
                {
                    "name": f"QDA_reg{reg}_macro",
                    "model": QuadraticDiscriminantAnalysis(reg_param=reg),
                    "view": "macro",
                }
            )

        return experts


class HierarchicalEnsemble:
    """
    Implements the Hierarchical Covariance Multi-Resolution Ensemble (HCMRE).
    Performs Greedy Forward Selection on a validation set, then retrains selected experts on full data.
    """

    def __init__(self, selection_iterations=50, random_seed=42):
        self.experts_def = ExpertFactory.get_experts()
        self.selection_iterations = selection_iterations
        self.random_seed = random_seed
        self.selected_experts = (
            []
        )  # List of dicts: {'model': instance, 'weight': int, 'view': str}
        self.classes_ = None

    def _get_X(self, data, view, split):
        """Helper to retrieve correct X array based on view and split."""
        key = f"X_{split}_{view}"
        if key not in data:
            raise KeyError(f"Data dictionary missing key: {key}")
        return data[key]

    def fit(self, data):
        """
        1. Trains all candidates on Train split.
        2. Evaluates on Val split.
        3. Selects best ensemble composition via Greedy Forward Selection.
        4. Retrains selected composition on Train + Val.

        Args:
            data (dict): Dictionary containing 'X_train_global', 'y_train', etc.
        """
        set_seed(self.random_seed)

        y_train = data["y_train"]
        y_val = data["y_val"]
        self.classes_ = data["classes"]

        print(
            f"Starting HCMRE fitting with {len(self.experts_def)} candidate experts..."
        )

        # --- Phase 1: Train Candidates and Generate Validation Predictions ---
        val_preds = []

        # We don't store trained models from this phase to save memory,
        # as we will retrain them on full data later.
        for exp in self.experts_def:
            model = clone(exp["model"])
            view = exp["view"]

            X_train_view = self._get_X(data, view, "train")
            X_val_view = self._get_X(data, view, "val")

            # Fit on training split
            model.fit(X_train_view, y_train)

            # Predict on validation split
            p_val = model.predict_proba(X_val_view)
            val_preds.append(p_val)

        val_preds = np.array(val_preds)  # Shape: (n_experts, n_samples, n_classes)

        # --- Phase 2: Greedy Forward Selection ---
        print("Running Greedy Forward Selection...")
        n_experts = len(self.experts_def)
        n_val_samples = len(y_val)
        n_classes = len(self.classes_)

        # Initialize ensemble accumulator
        ensemble_sum_probs = np.zeros((n_val_samples, n_classes), dtype=np.float64)
        selected_indices = []  # List of indices into self.experts_def

        best_log_loss = float("inf")

        for i in range(self.selection_iterations):
            iteration_best_loss = float("inf")
            iteration_best_idx = -1

            # Try adding each expert to the current ensemble
            for idx in range(n_experts):
                # Calculate potential new average
                # (current_sum + new_pred) / (current_count + 1)
                temp_sum = ensemble_sum_probs + val_preds[idx]
                temp_avg = temp_sum / (len(selected_indices) + 1)

                loss = clipped_log_loss(y_val, temp_avg)

                if loss < iteration_best_loss:
                    iteration_best_loss = loss
                    iteration_best_idx = idx

            # Update ensemble with best found this iteration
            selected_indices.append(iteration_best_idx)
            ensemble_sum_probs += val_preds[iteration_best_idx]
            best_log_loss = iteration_best_loss

            print(
                f"Selection Iteration {i+1}/{self.selection_iterations}: Added {self.experts_def[iteration_best_idx]['name']}, Val Log Loss: {best_log_loss:.15f}"
            )

        # Consolidate selected experts (calculate integer weights)
        counts = Counter(selected_indices)

        print("\nFinal Selected Ensemble Composition:")
        for idx, count in counts.items():
            print(f"  - {self.experts_def[idx]['name']}: Weight {count}")

        # --- Phase 3: Final Retraining ---
        print("\nRetraining selected experts on Combined (Train + Val) data...")

        self.selected_experts = []

        # Prepare full datasets for each view
        full_data = {}
        for view in ["global", "macro", "combined"]:
            X_tr = self._get_X(data, view, "train")
            X_v = self._get_X(data, view, "val")
            full_data[view] = np.vstack([X_tr, X_v])

        y_full = np.concatenate([y_train, y_val])

        # Retrain only the unique models selected
        for idx, count in counts.items():
            exp_def = self.experts_def[idx]
            view = exp_def["view"]

            # Clone a fresh instance
            model = clone(exp_def["model"])

            # Fit on full data (Train + Val)
            model.fit(full_data[view], y_full)

            self.selected_experts.append(
                {"model": model, "weight": count, "view": view, "name": exp_def["name"]}
            )

        print("Retraining complete.")

    def predict(self, data):
        """
        Generates predictions for the test set using the retrained ensemble.

        Args:
            data (dict): Dictionary containing 'X_test_global', 'test_ids', etc.

        Returns:
            np.ndarray: Probability matrix of shape (n_test, n_classes).
        """
        if not self.selected_experts:
            raise RuntimeError("Model not fitted yet. Call fit() before predict().")

        n_test = len(data["test_ids"])
        n_classes = len(self.classes_)

        weighted_sum_probs = np.zeros((n_test, n_classes), dtype=np.float64)
        total_weight = 0

        for item in self.selected_experts:
            model = item["model"]
            weight = item["weight"]
            view = item["view"]

            X_test = self._get_X(data, view, "test")

            # Predict
            probs = model.predict_proba(X_test)

            # Accumulate weighted probabilities
            weighted_sum_probs += probs * weight
            total_weight += weight

        # Compute final weighted average
        final_probs = weighted_sum_probs / total_weight

        return final_probs
