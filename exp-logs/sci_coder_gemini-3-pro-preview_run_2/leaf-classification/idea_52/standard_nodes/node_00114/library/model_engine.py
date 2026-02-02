import numpy as np
import copy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import RANDOM_SEED, FLOAT_PRECISION, LDA_SUBSPACE_COMPONENTS
from library.pipeline_factory import create_preprocessing_pipeline
from library.utils import clipped_log_loss


class ProbabilisticExpert(BaseEstimator, ClassifierMixin):
    """
    Wraps a specific preprocessing pipeline and an LDA estimator.
    Handles feature selection from the data dictionary.
    """

    def __init__(
        self,
        name,
        group,
        feature_source,
        pipeline_type,
        shrinkage,
        pca_variance=None,
        poly_degree=None,
        lda_components=None,
    ):
        self.name = name
        self.group = group
        self.feature_source = feature_source
        self.pipeline_type = pipeline_type
        self.shrinkage = shrinkage

        # Optional overrides for pipeline parameters
        self.pca_variance = pca_variance
        self.poly_degree = poly_degree
        self.lda_components = lda_components

        self.pipeline = None
        self.estimator = None
        self.classes_ = None

    def _build_model(self):
        """Constructs the pipeline and estimator if not already built."""
        # 1. Create Pipeline
        # Pass optional args only if they are not None (relying on factory defaults otherwise)
        kwargs = {}
        if self.pca_variance is not None:
            kwargs["pca_variance"] = self.pca_variance
        if self.poly_degree is not None:
            kwargs["poly_degree"] = self.poly_degree
        if self.lda_components is not None:
            kwargs["lda_components"] = self.lda_components

        self.pipeline = create_preprocessing_pipeline(self.pipeline_type, **kwargs)

        # 2. Create Estimator
        # LDA with shrinkage requires 'lsqr' or 'eigen' solver.
        solver = "lsqr"
        self.estimator = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=self.shrinkage,
            store_covariance=True,  # Useful for debugging, though not strictly required
        )

    def fit(self, X_dict, y):
        """
        Fits the pipeline and estimator.

        Args:
            X_dict (dict): Dictionary containing feature arrays (global, margin, etc.).
            y (array-like): Target labels.
        """
        if self.pipeline is None:
            self._build_model()

        # Select specific feature view
        X_view = X_dict[self.feature_source]

        # Fit Pipeline
        # Note: Some pipelines (subspace_poly) contain LDA steps that require y
        X_transformed = self.pipeline.fit_transform(X_view, y)

        # Fit Estimator
        self.estimator.fit(X_transformed, y)
        self.classes_ = self.estimator.classes_

        return self

    def predict_proba(self, X_dict):
        """
        Predicts class probabilities.
        """
        if self.estimator is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        X_view = X_dict[self.feature_source]
        X_transformed = self.pipeline.transform(X_view)

        return self.estimator.predict_proba(X_transformed).astype(FLOAT_PRECISION)


class GreedySelector:
    """
    Implements Greedy Forward Selection to build an ensemble of experts.
    """

    def __init__(self, expert_configs, max_steps=50, tolerance=1e-6):
        """
        Args:
            expert_configs (list): List of dictionaries defining candidate experts.
            max_steps (int): Maximum number of experts to add to the ensemble.
            tolerance (float): Minimum improvement in log loss required to continue.
        """
        self.expert_configs = expert_configs
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.selected_experts = []  # List of (config, weight)
        self.best_loss = float("inf")

    def fit(self, X_train_dict, y_train, X_val_dict, y_val):
        """
        Trains all candidates on train set, then runs selection on val set.
        """
        print(
            f"Starting Greedy Selection with {len(self.expert_configs)} candidate configurations..."
        )

        # 1. Train all candidates and generate validation probabilities
        candidate_probs = []
        candidate_models = []

        # Flatten the config grid: each config might have a shrinkage_grid
        # We need to instantiate concrete experts for every combination
        concrete_candidates = []

        for config in self.expert_configs:
            shrinkage_options = config.get("shrinkage_grid", ["auto"])

            for shrinkage in shrinkage_options:
                # Create a concrete expert definition
                expert_def = config.copy()
                if "shrinkage_grid" in expert_def:
                    del expert_def["shrinkage_grid"]
                expert_def["shrinkage"] = shrinkage
                expert_def["name"] = f"{config['name']}_s{shrinkage}"

                concrete_candidates.append(expert_def)

        print(
            f"Expanded to {len(concrete_candidates)} concrete experts (via shrinkage grid)."
        )

        # Train and Predict
        for i, expert_def in enumerate(concrete_candidates):
            expert = ProbabilisticExpert(**expert_def)
            expert.fit(X_train_dict, y_train)

            probs = expert.predict_proba(X_val_dict)

            # Calculate individual score for reference
            score = clipped_log_loss(y_val, probs)

            candidate_models.append(expert_def)
            candidate_probs.append(probs)

            # print(f"  Candidate {i}: {expert_def['name']} | Val Loss: {score:.6f}")

        candidate_probs = np.array(
            candidate_probs
        )  # Shape: (n_candidates, n_samples, n_classes)
        n_candidates = len(candidate_models)
        n_samples = len(y_val)
        n_classes = len(
            np.unique(y_val)
        )  # Assuming y_val covers all classes or passed implicitly

        # 2. Greedy Selection Loop
        current_sum_probs = np.zeros(
            (n_samples, candidate_probs.shape[2]), dtype=FLOAT_PRECISION
        )
        current_weights = []  # Indices of selected candidates

        print("\nSelection Loop:")
        for step in range(self.max_steps):
            best_step_loss = float("inf")
            best_candidate_idx = -1

            # Try adding each candidate
            for i in range(n_candidates):
                # Temporary ensemble sum
                temp_sum = current_sum_probs + candidate_probs[i]

                # Normalize (divide by current count + 1)
                # Note: clipped_log_loss handles row-normalization, so we just pass the sum
                # However, for stability, let's average it before passing
                temp_avg = temp_sum / (len(current_weights) + 1)

                loss = clipped_log_loss(y_val, temp_avg)

                if loss < best_step_loss:
                    best_step_loss = loss
                    best_candidate_idx = i

            # Check improvement
            improvement = self.best_loss - best_step_loss

            if improvement > self.tolerance:
                self.best_loss = best_step_loss
                current_weights.append(best_candidate_idx)
                current_sum_probs += candidate_probs[best_candidate_idx]

                selected_name = candidate_models[best_candidate_idx]["name"]
                print(
                    f"  Step {step+1}: Added {selected_name} | New Best Loss: {self.best_loss:.10f} | Improv: {improvement:.10f}"
                )
            else:
                print(
                    f"  Step {step+1}: No significant improvement ({improvement:.10f} <= {self.tolerance}). Stopping."
                )
                break

        # 3. Compile Final Selection
        # Count occurrences of each selected index
        from collections import Counter

        counts = Counter(current_weights)

        final_selection = []
        for idx, count in counts.items():
            final_selection.append(
                {"expert_def": candidate_models[idx], "weight": count}
            )

        self.selected_experts = final_selection
        print(
            f"\nSelection Complete. Ensemble size: {len(current_weights)} experts (composed of {len(final_selection)} unique models)."
        )
        return self.selected_experts


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Final ensemble model that retrains selected experts and aggregates predictions.
    """

    def __init__(self, selected_experts):
        """
        Args:
            selected_experts (list): List of dicts {'expert_def': dict, 'weight': int}
        """
        self.selected_experts_config = selected_experts
        self.experts = []
        self.weights = []

    def fit(self, X_dict, y):
        """
        Retrains all selected experts on the full provided dataset.
        """
        self.experts = []
        self.weights = []

        print(
            f"Retraining {len(self.selected_experts_config)} unique experts on full data..."
        )

        for item in self.selected_experts_config:
            config = item["expert_def"]
            weight = item["weight"]

            # Instantiate new expert
            expert = ProbabilisticExpert(**config)

            # Fit
            expert.fit(X_dict, y)

            self.experts.append(expert)
            self.weights.append(weight)

        return self

    def predict_proba(self, X_dict):
        """
        Aggregates predictions using weighted averaging.
        """
        if not self.experts:
            raise RuntimeError("Ensemble not fitted.")

        # Get predictions from first expert to determine shape
        first_probs = self.experts[0].predict_proba(X_dict)
        weighted_sum = np.zeros_like(first_probs, dtype=FLOAT_PRECISION)
        total_weight = sum(self.weights)

        # Accumulate weighted probabilities
        for expert, weight in zip(self.experts, self.weights):
            probs = expert.predict_proba(X_dict)
            weighted_sum += probs * weight

        # Average
        final_probs = weighted_sum / total_weight

        return final_probs
