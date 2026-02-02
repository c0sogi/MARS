import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.base import clone
from library.utils import set_seed, clipped_log_loss


class GreedyEnsembleSelector:
    """
    Implements the Stratified-Manifold Precision-Generative Ensemble (SMPGE).
    Manages expert definition, greedy forward selection, and final retraining.
    """

    def __init__(self, max_ensemble_size=30, tolerance=1e-6):
        self.max_ensemble_size = max_ensemble_size
        self.tolerance = tolerance
        self.selected_experts = (
            []
        )  # List of dicts: {'name': str, 'model': obj, 'view': str, 'weight': int}
        self.candidates = []  # List of dicts: {'name': str, 'model': obj, 'view': str}
        self.n_classes = None

    def _build_library(self):
        """
        Constructs the library of probabilistic experts based on the SMPGE strategy.
        """
        library = []

        # Group A: Global-Rotational Anchors (LDA + Shrinkage)
        # View: 'global'
        shrinkage_values = [0.001, 0.01]
        for s in shrinkage_values:
            library.append(
                {
                    "name": f"Global_LDA_shrink_{s}",
                    "model": LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s),
                    "view": "global",
                }
            )

        # Group B: Stratified-Rotational Experts (LDA + Shrinkage)
        # View: 'stratified'
        for s in shrinkage_values:
            library.append(
                {
                    "name": f"Stratified_LDA_shrink_{s}",
                    "model": LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s),
                    "view": "stratified",
                }
            )

        # Group C: Physical Non-Linear Experts (QDA + Regularization)
        # View: 'physical'
        reg_params = [0.1, 0.5]
        for r in reg_params:
            library.append(
                {
                    "name": f"Physical_QDA_reg_{r}",
                    "model": QuadraticDiscriminantAnalysis(reg_param=r),
                    "view": "physical",
                }
            )

        self.candidates = library

    def fit(self, data):
        """
        Performs the two-phase training process:
        1. Train all candidates on Train split -> Select best subset via Greedy Forward Selection on Val split.
        2. Retrain selected subset on Combined (Train + Val) data.
        """
        set_seed(42)
        self._build_library()

        # Extract data
        y_train = data["y_train"]
        y_val = data["y_val"]
        self.n_classes = len(np.unique(y_train))

        # Map views to data
        views = {
            "global": (data["X_train_global"], data["X_val_global"]),
            "stratified": (data["X_train_stratified"], data["X_val_stratified"]),
            "physical": (data["X_train_physical"], data["X_val_physical"]),
        }

        print(
            f"Starting Phase 1: Candidate Training and Selection (Candidates: {len(self.candidates)})"
        )

        # --- Phase 1: Train Candidates and Generate Val Predictions ---
        candidate_preds = []

        for i, cand in enumerate(self.candidates):
            X_tr, X_v = views[cand["view"]]

            # Clone model to ensure fresh start
            model = clone(cand["model"])
            model.fit(X_tr, y_train)

            # Predict on validation set
            # Ensure float64
            preds = model.predict_proba(X_v).astype(np.float64)
            candidate_preds.append(preds)

            # Store the trained model temporarily (though we will retrain later)
            cand["trained_model_phase1"] = model

            # Initial score check
            loss = clipped_log_loss(y_val, preds)
            # print(f"  Candidate {cand['name']} Val Loss: {loss:.6f}")

        # --- Greedy Forward Selection ---
        ensemble_preds = np.zeros((len(y_val), self.n_classes), dtype=np.float64)
        selected_indices = []  # Indices into self.candidates
        best_loss = float("inf")

        print("Running Greedy Forward Selection...")

        for step in range(self.max_ensemble_size):
            best_step_loss = float("inf")
            best_candidate_idx = -1

            # Try adding each candidate to the current ensemble
            for idx, preds in enumerate(candidate_preds):
                # Current ensemble sum + new candidate
                # We average by (step + 1)
                current_sum = ensemble_preds + preds
                current_avg = current_sum / (step + 1)

                loss = clipped_log_loss(y_val, current_avg)

                if loss < best_step_loss:
                    best_step_loss = loss
                    best_candidate_idx = idx

            # Check for improvement
            if best_step_loss < best_loss - self.tolerance:
                best_loss = best_step_loss
                selected_indices.append(best_candidate_idx)
                ensemble_preds += candidate_preds[best_candidate_idx]

                cand_name = self.candidates[best_candidate_idx]["name"]
                print(
                    f"  Step {step+1}: Added {cand_name} | New Best Val Loss: {best_loss:.15f}"
                )
            else:
                print(
                    f"  Step {step+1}: No significant improvement. Stopping selection."
                )
                break

        if not selected_indices:
            print("Warning: No experts selected. Defaulting to first candidate.")
            selected_indices = [0]

        # Compile selected experts
        # We count occurrences to handle weighting naturally
        from collections import Counter

        counts = Counter(selected_indices)

        self.selected_experts = []
        for idx, count in counts.items():
            cand = self.candidates[idx]
            self.selected_experts.append(
                {
                    "name": cand["name"],
                    "base_model": cand["model"],  # Untrained template
                    "view": cand["view"],
                    "weight": count,
                    "final_model": cand[
                        "trained_model_phase1"
                    ],  # Use Phase 1 model for validation
                }
            )

        print(
            f"Selection Complete. Ensemble Size: {len(selected_indices)} (Unique Experts: {len(self.selected_experts)})"
        )
        return self

    def refit(self, data):
        """
        Phase 2: Retrains selected experts on Combined (Train + Val) data.
        Call this AFTER validation and BEFORE final submission.
        """
        if not self.selected_experts:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        print("Starting Phase 2: Retraining on Combined Data...")

        y_train = data["y_train"]
        y_val = data["y_val"]

        # Map views to data
        views = {
            "global": (data["X_train_global"], data["X_val_global"]),
            "stratified": (data["X_train_stratified"], data["X_val_stratified"]),
            "physical": (data["X_train_physical"], data["X_val_physical"]),
        }

        for expert in self.selected_experts:
            # Prepare Combined Data
            X_tr, X_v = views[expert["view"]]
            X_combined = np.vstack([X_tr, X_v])
            y_combined = np.concatenate([y_train, y_val])

            # Retrain
            model = clone(expert["base_model"])
            model.fit(X_combined, y_combined)
            expert["final_model"] = model

        print("Retraining Complete.")
        return self

    def predict(self, data):
        """
        Generates predictions for the test set using the retrained ensemble.
        """
        if not self.selected_experts:
            raise ValueError("Ensemble not fitted.")

        # Map views to test data
        views = {
            "global": data["X_test_global"],
            "stratified": data["X_test_stratified"],
            "physical": data["X_test_physical"],
        }

        n_samples = views["global"].shape[0]  # Assuming all views have same n_samples

        # Initialize weighted sum
        weighted_sum_preds = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        total_weight = 0

        for expert in self.selected_experts:
            X_test = views[expert["view"]]
            model = expert["final_model"]
            weight = expert["weight"]

            preds = model.predict_proba(X_test).astype(np.float64)
            weighted_sum_preds += preds * weight
            total_weight += weight

        # Normalize
        final_preds = weighted_sum_preds / total_weight

        # Ensure row normalization (just in case of float drift, though division handles it)
        row_sums = final_preds.sum(axis=1)
        final_preds = final_preds / row_sums[:, np.newaxis]

        return final_preds


def get_expert_library():
    """
    Factory function to return the library definition.
    Useful if external inspection is needed, though logic is encapsulated in Selector.
    """
    selector = GreedyEnsembleSelector()
    selector._build_library()
    return selector.candidates
