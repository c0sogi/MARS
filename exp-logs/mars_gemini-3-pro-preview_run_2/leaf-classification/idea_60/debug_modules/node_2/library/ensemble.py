import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library import config, utils, expert_pipelines


class GreedySelector:
    """
    Implements Greedy Forward Selection with replacement to select an ensemble of experts.
    This strategy iteratively adds the expert that maximizes the improvement in
    validation log loss.
    """

    def __init__(self, max_size=50, tolerance=1e-6):
        """
        Args:
            max_size (int): Maximum number of experts to select.
            tolerance (float): Minimum improvement in log loss required to continue selection.
        """
        self.max_size = max_size
        self.tolerance = tolerance
        self.selected_experts = []  # List of expert keys
        self.best_score = float("inf")

    def fit(self, candidate_preds, y_true):
        """
        Selects experts based on validation log loss.

        Args:
            candidate_preds (dict): Dictionary mapping expert names to prediction matrices
                                    (n_samples, n_classes).
            y_true (array-like): True labels (strings or ints).

        Returns:
            list: List of selected expert keys (strings).
        """
        # Ensure y_true is properly formatted
        y_true = np.array(y_true)

        # Reset state
        self.selected_experts = []
        self.best_score = float("inf")

        available_candidates = list(candidate_preds.keys())
        if not available_candidates:
            print("No candidates provided for selection.")
            return []

        # Current ensemble sum of probabilities (unweighted sum)
        # We maintain the sum to avoid recomputing averages from scratch
        current_sum = None

        print(f"Starting Greedy Forward Selection (Max Size: {self.max_size})...")

        for i in range(self.max_size):
            best_iter_score = float("inf")
            best_iter_candidate = None

            # Try adding each candidate to the current ensemble
            for name in available_candidates:
                pred = candidate_preds[name]

                if current_sum is None:
                    # First iteration: ensemble size will be 1
                    temp_avg = pred
                else:
                    # Current size is i, new size is i+1
                    # New Average = (Sum + New_Pred) / (Count + 1)
                    temp_avg = (current_sum + pred) / (i + 1)

                score = utils.calculate_log_loss(y_true, temp_avg)

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_candidate = name

            # Check for improvement
            # For the first iteration, we always select the best single model
            if i == 0 or best_iter_score < (self.best_score - self.tolerance):
                self.best_score = best_iter_score
                self.selected_experts.append(best_iter_candidate)

                # Update current sum
                if current_sum is None:
                    current_sum = candidate_preds[best_iter_candidate]
                else:
                    current_sum += candidate_preds[best_iter_candidate]

                print(
                    f"Step {i+1}: Added {best_iter_candidate}. Validation Log Loss: {self.best_score}"
                )
            else:
                print(
                    f"Step {i+1}: No sufficient improvement (Best: {best_iter_score}, Current: {self.best_score}). Stopping."
                )
                break

        return self.selected_experts


class HDME_Ensemble:
    """
    Hierarchical Discriminative-Manifold Ensemble.
    Manages candidate generation, selection, and final prediction.
    """

    def __init__(self, max_ensemble_size=50):
        self.selector = GreedySelector(max_size=max_ensemble_size)
        self.selected_config = []  # List of tuples (pipeline_key, shrinkage)

    def _get_pipeline_builders(self):
        """
        Returns a list of configuration tuples for creating pipelines.
        Format: (unique_key, builder_function, args_tuple)
        """
        builders = []

        # Group A: Global Anchors
        for p in config.GROUP_A_CONFIG["pipelines"]:
            key = f"GroupA_{p['name']}"
            builders.append((key, expert_pipelines.build_global_pipeline, (p["name"],)))

        # Group B: Stratified Rotational
        builders.append(
            (
                "GroupB_Stratified",
                expert_pipelines.build_stratified_rotational_pipeline,
                (),
            )
        )

        # Group C: Intra-Domain
        builders.append(
            ("GroupC_Intra", expert_pipelines.build_intra_domain_pipeline, ())
        )

        # Group C: Inter-Domain
        for pair in config.GROUP_C_INTER_CONFIG["pairs"]:
            key = f"GroupC_Inter_{pair[0]}_{pair[1]}"
            builders.append(
                (key, expert_pipelines.build_inter_domain_pipeline, (pair,))
            )

        # Group D: Morphometrics
        builders.append(
            ("GroupD_Morph", expert_pipelines.build_morphometric_pipeline, ())
        )

        return builders

    def fit(self, X_train, y_train, X_val, y_val, feature_subsets):
        """
        Generates candidates, trains them on split data, and runs selection on validation data.

        Args:
            X_train, y_train: Training split.
            X_val, y_val: Validation split.
            feature_subsets: Dictionary of feature columns.
        """
        print("Generating candidate experts...")
        candidate_preds = {}

        builders = self._get_pipeline_builders()

        for pipe_key, builder_func, args in builders:
            # 1. Build and Fit Pipeline
            try:
                pipeline = builder_func(*args, feature_subsets=feature_subsets)
                # Fit on training split
                X_train_trans = pipeline.fit_transform(X_train, y_train)
                # Transform validation split
                X_val_trans = pipeline.transform(X_val)
            except Exception as e:
                print(f"Failed to process pipeline {pipe_key}: {e}")
                continue

            # 2. Train LDA Experts with varying shrinkage
            for shrinkage in config.LDA_SHRINKAGE_CANDIDATES:
                expert_id = f"{pipe_key}__shrinkage_{shrinkage}"

                # Determine solver based on shrinkage
                # 'svd' does not support shrinkage, 'lsqr' and 'eigen' do.
                solver = "lsqr"
                if shrinkage is None:
                    solver = "svd"

                try:
                    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
                    clf.fit(X_train_trans, y_train)

                    # Predict on validation
                    preds = clf.predict_proba(X_val_trans)
                    candidate_preds[expert_id] = utils.enforce_float64(preds)
                except Exception as e:
                    # Some shrinkage values might be incompatible with data dimensionality or solver
                    pass

        print(f"Generated {len(candidate_preds)} candidate experts.")

        # 3. Run Selection
        selected_keys = self.selector.fit(candidate_preds, y_val)

        # 4. Store Configuration
        self.selected_config = []
        for key in selected_keys:
            # Parse key to separate pipeline and shrinkage
            parts = key.split("__shrinkage_")
            p_key = parts[0]
            s_val = parts[1]

            # Convert shrinkage back to appropriate type
            if s_val == "None":
                s_val = None
            elif s_val != "auto":
                try:
                    s_val = float(s_val)
                except ValueError:
                    pass  # Keep as string if not float

            self.selected_config.append((p_key, s_val))

        print(f"Final Ensemble Size: {len(self.selected_config)}")

    def predict(self, X_full, y_full, X_test, feature_subsets):
        """
        Retrains the selected experts on the full dataset and predicts on test data.
        Aggregates predictions using the weights (counts) determined during selection.

        Args:
            X_full, y_full: Combined Training + Validation data.
            X_test: Test data.
            feature_subsets: Dictionary of feature columns.

        Returns:
            np.ndarray: Weighted average probabilities (n_test, n_classes).
        """
        if not self.selected_config:
            raise ValueError("Ensemble must be fit before prediction.")

        # Count occurrences of each (pipeline, shrinkage) pair to determine weight
        expert_counts = {}
        for item in self.selected_config:
            expert_counts[item] = expert_counts.get(item, 0) + 1

        # Group by pipeline to minimize redundant feature processing
        # Map: pipeline_key -> list of (shrinkage, weight)
        pipeline_groups = {}
        for (p_key, shrinkage), weight in expert_counts.items():
            if p_key not in pipeline_groups:
                pipeline_groups[p_key] = []
            pipeline_groups[p_key].append((shrinkage, weight))

        # Map keys to builders
        builders_map = {k: (b, a) for k, b, a in self._get_pipeline_builders()}

        final_sum = None
        total_weight = 0.0

        print("Retraining selected experts on full dataset...")

        for pipe_key, configs in pipeline_groups.items():
            if pipe_key not in builders_map:
                print(f"Warning: Pipeline {pipe_key} not found in builders map.")
                continue

            builder_func, args = builders_map[pipe_key]

            # 1. Re-fit Pipeline on Full Data
            try:
                pipeline = builder_func(*args, feature_subsets=feature_subsets)
                X_full_trans = pipeline.fit_transform(X_full, y_full)
                X_test_trans = pipeline.transform(X_test)
            except Exception as e:
                print(f"Error refitting pipeline {pipe_key}: {e}")
                continue

            # 2. Re-fit Classifiers
            for shrinkage, weight in configs:
                solver = "lsqr"
                if shrinkage is None:
                    solver = "svd"

                try:
                    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
                    clf.fit(X_full_trans, y_full)

                    preds = clf.predict_proba(X_test_trans)
                    preds = utils.enforce_float64(preds)

                    if final_sum is None:
                        final_sum = preds * weight
                    else:
                        final_sum += preds * weight

                    total_weight += weight
                except Exception as e:
                    print(f"Error refitting classifier {pipe_key} (s={shrinkage}): {e}")

        if final_sum is None or total_weight == 0:
            raise RuntimeError(
                "Prediction failed: No experts were successfully retrained."
            )

        return final_sum / total_weight
