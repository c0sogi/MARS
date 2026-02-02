import os
import numpy as np
import pandas as pd
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Type

from library.utils import set_seed, calculate_log_loss
from library.preprocessing import get_preprocessed_data
from library.models import LDAWrapper, QDAWrapper

# =============================================================================
# Expert Definition
# =============================================================================


@dataclass
class Expert:
    """
    Represents a single candidate model in the ensemble library.
    """

    name: str
    strategy: str  # Corresponds to preprocessing strategy (e.g., 'global_marginal')
    estimator_class: Type
    estimator_params: Dict[str, Any]
    model: Optional[Any] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the underlying estimator."""
        self.model = self.estimator_class(**self.estimator_params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities."""
        if self.model is None:
            raise RuntimeError(f"Expert {self.name} is not fitted.")
        return self.model.predict_proba(X)


# =============================================================================
# Ensemble Logic
# =============================================================================


class GreedyEnsembleSelector:
    """
    Manages the library of experts, performs greedy forward selection on validation data,
    and handles final retraining and inference.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.candidates: List[Expert] = []
        self.selected_experts: List[Tuple[Expert, int]] = []  # List of (Expert, Weight)
        self.classes_: Optional[np.ndarray] = None

        # Define the Expert Library (Groups A, B, C)
        self._build_library()

    def _build_library(self):
        """Constructs the library of candidate experts."""

        # --- Group A: Global Statistical Anchors ---
        # Strategies: global_marginal, global_rotational, global_robust
        # Algorithm: LDA with Fixed Shrinkage
        shrinkages = [0.001, 0.01]
        strategies_a = ["global_marginal", "global_rotational", "global_robust"]

        for strat in strategies_a:
            for shrink in shrinkages:
                self.candidates.append(
                    Expert(
                        name=f"GroupA_{strat}_LDA_shrink{shrink}",
                        strategy=strat,
                        estimator_class=LDAWrapper,
                        estimator_params={
                            "solver": "lsqr",
                            "shrinkage": shrink,
                            "random_state": self.random_state,
                        },
                    )
                )

        # --- Group B: Stratified Rotational Experts ---
        # Strategy: stratified_rotational
        # Algorithm: LDA with Shrinkage Library (Fixed + Auto + OAS)
        strat_b = "stratified_rotational"
        shrinkages_b = [0.001, 0.01, "auto", "oas"]

        for shrink in shrinkages_b:
            self.candidates.append(
                Expert(
                    name=f"GroupB_{strat_b}_LDA_shrink{shrink}",
                    strategy=strat_b,
                    estimator_class=LDAWrapper,
                    estimator_params={
                        "solver": "lsqr",
                        "shrinkage": shrink,
                        "random_state": self.random_state,
                    },
                )
            )

        # --- Group C: Physical Non-Linear Experts ---
        # Strategy: morph_physical
        # Algorithm: QDA with Regularization
        strat_c = "morph_physical"
        reg_params = [0.1, 0.5]

        for reg in reg_params:
            self.candidates.append(
                Expert(
                    name=f"GroupC_{strat_c}_QDA_reg{reg}",
                    strategy=strat_c,
                    estimator_class=QDAWrapper,
                    estimator_params={
                        "reg_param": reg,
                        "random_state": self.random_state,
                    },
                )
            )

    def fit_selection(self, load_cached_data: bool = True):
        """
        Phase 1: Trains all candidates on Train split, evaluates on Val split,
        and runs Greedy Forward Selection to determine the optimal ensemble.
        """
        print("Starting Phase 1: Expert Selection...")
        set_seed(self.random_state)

        # 1. Cache Validation Predictions for all Candidates
        # We process data strategy-by-strategy to minimize I/O and re-computation

        # Group candidates by strategy to load data once per strategy
        candidates_by_strategy = {}
        for exp in self.candidates:
            if exp.strategy not in candidates_by_strategy:
                candidates_by_strategy[exp.strategy] = []
            candidates_by_strategy[exp.strategy].append(exp)

        val_preds_cache = {}  # name -> proba_matrix
        y_val_true = None

        for strategy, experts in candidates_by_strategy.items():
            # Load Data
            data = get_preprocessed_data(strategy, load_cached_data=load_cached_data)
            X_train = data["X_train"]
            y_train = data["y_train"]
            X_val = data["X_val"]
            y_val = data["y_val"]

            if y_val_true is None:
                y_val_true = y_val

            # Train and Predict
            for exp in experts:
                # print(f"  Training candidate: {exp.name}")
                exp.fit(X_train, y_train)
                preds = exp.predict_proba(X_val)
                val_preds_cache[exp.name] = preds

        # 2. Greedy Forward Selection
        print("Running Greedy Forward Selection...")

        # Initialize
        selected_indices = []  # Indices into self.candidates
        current_ensemble_preds = None
        best_score = float("inf")

        # We allow up to N iterations (e.g., 20) or until no improvement
        max_iter = 20
        tolerance = 1e-6

        for i in range(max_iter):
            iteration_best_score = float("inf")
            iteration_best_idx = -1
            iteration_best_preds = None

            # Try adding each candidate to the current ensemble
            for idx, exp in enumerate(self.candidates):
                candidate_preds = val_preds_cache[exp.name]

                if current_ensemble_preds is None:
                    # First selection
                    temp_preds = candidate_preds
                else:
                    # Weighted average: (current_sum + new_pred) / (current_count + 1)
                    # We maintain the sum for efficiency
                    # current_ensemble_preds stores the SUM of probas so far
                    temp_preds = current_ensemble_preds + candidate_preds

                # Normalize for scoring (divide by count)
                n_members = len(selected_indices) + 1
                score_preds = temp_preds / n_members

                score = calculate_log_loss(y_val_true, score_preds)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_idx = idx
                    iteration_best_preds = temp_preds  # Keep the sum

            # Check for improvement
            if iteration_best_score < best_score - tolerance:
                best_score = iteration_best_score
                selected_indices.append(iteration_best_idx)
                current_ensemble_preds = iteration_best_preds
                best_exp_name = self.candidates[iteration_best_idx].name
                print(
                    f"  Iter {i+1}: Added {best_exp_name}, Val Log Loss: {best_score:.15f}"
                )
            else:
                print(f"  Iter {i+1}: No significant improvement. Stopping.")
                break

        # 3. Store Selected Experts
        # Count occurrences to determine weights
        from collections import Counter

        counts = Counter(selected_indices)

        self.selected_experts = []
        print("\nFinal Selected Ensemble:")
        for idx, weight in counts.items():
            exp = self.candidates[idx]
            self.selected_experts.append((exp, weight))
            print(f"  - {exp.name} (Weight: {weight})")

        if not self.selected_experts:
            print("Warning: No experts selected. Defaulting to first candidate.")
            self.selected_experts.append((self.candidates[0], 1))

    def refit_final(self, load_cached_data: bool = True):
        """
        Phase 2: Retrains the selected experts on the combined Train + Val set.
        """
        print("\nStarting Phase 2: Final Retraining...")
        set_seed(self.random_state)

        # Group selected experts by strategy to optimize data loading
        selected_by_strategy = {}
        for exp, weight in self.selected_experts:
            if exp.strategy not in selected_by_strategy:
                selected_by_strategy[exp.strategy] = []
            selected_by_strategy[exp.strategy].append(exp)

        for strategy, experts in selected_by_strategy.items():
            # Load Data
            data = get_preprocessed_data(strategy, load_cached_data=load_cached_data)

            # Concatenate Train and Val
            X_full = np.vstack([data["X_train"], data["X_val"]])
            y_full = np.concatenate([data["y_train"], data["y_val"]])

            # Refit
            for exp in experts:
                # print(f"  Refitting {exp.name} on full data...")
                exp.fit(X_full, y_full)

    def predict_submission(self, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Generates predictions for the test set using the retrained ensemble.
        """
        print("\nGenerating Submission Predictions...")

        # Group by strategy
        selected_by_strategy = {}
        for exp, weight in self.selected_experts:
            if exp.strategy not in selected_by_strategy:
                selected_by_strategy[exp.strategy] = []
            selected_by_strategy[exp.strategy].append((exp, weight))

        ensemble_preds_sum = None
        total_weight = 0
        test_ids = None

        # Iterate strategies
        for strategy, weighted_experts in selected_by_strategy.items():
            # Load Test Data
            data = get_preprocessed_data(strategy, load_cached_data=load_cached_data)
            X_test = data["X_test"]
            if test_ids is None:
                test_ids = data["ids_test"]

            # Predict
            for exp, weight in weighted_experts:
                preds = exp.predict_proba(X_test)

                if ensemble_preds_sum is None:
                    ensemble_preds_sum = preds * weight
                else:
                    ensemble_preds_sum += preds * weight

                total_weight += weight

        # Average
        final_preds = ensemble_preds_sum / total_weight

        # Get class names (need to load from metadata or assume sorted from training)
        # The data loader returns y as strings. LDA/QDA classes_ attribute will hold the sorted unique labels.
        # We can grab classes from the first fitted model.
        first_model = self.selected_experts[0][0].model
        classes = first_model.classes_

        # Construct DataFrame
        df_sub = pd.DataFrame(final_preds, columns=classes)
        df_sub.insert(0, "id", test_ids)

        return df_sub


# =============================================================================
# Pipeline Execution
# =============================================================================


def run_ensemble_pipeline(output_path: str = "./submission/submission.csv"):
    """
    Main entry point to run the Full-Rank Stratified-Manifold Precision-Generative Ensemble.
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize Selector
    selector = GreedyEnsembleSelector(random_state=42)

    # Phase 1: Selection
    selector.fit_selection(load_cached_data=True)

    # Phase 2: Retraining
    selector.refit_final(load_cached_data=True)

    # Inference
    df_submission = selector.predict_submission(load_cached_data=True)

    # Save
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
