import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from library.config import (
    EXPERTS_CONFIG,
    FEATURE_GROUPS,
    LDA_SHRINKAGE_CANDIDATES,
    LDA_SOLVER,
    FLOAT_PRECISION,
    QUANTILE_N_QUANTILES,
    QUANTILE_OUTPUT_DIST,
    POLY_DEGREE,
    BOTTLENECK_K,
    RANDOM_SEED,
)
from library.transformers import (
    make_marginal_pipeline,
    make_alignment_pipeline,
    make_robust_pipeline,
    make_polynomial_pipeline,
    make_bottleneck_pipeline,
)


class Expert(BaseEstimator, ClassifierMixin):
    """
    A self-contained model expert that encapsulates a specific feature subset,
    a preprocessing topology, and an LDA estimator.

    Supports:
    - Single-stream topologies (Marginal, Rotational, Robust, Polynomial)
    - Multi-stream topologies (Pairwise Interaction with Bottlenecks)
    """

    def __init__(
        self, name, group, topology, feature_keys, shrinkage, random_state=RANDOM_SEED
    ):
        self.name = name
        self.group = group
        self.topology = topology
        self.feature_keys = feature_keys
        self.shrinkage = shrinkage
        self.random_state = random_state

        # Internal state
        self.pipelines = []  # List of (pipeline, column_names) tuples
        self.merge_pipeline = None
        self.estimator = None
        self.feature_cols_by_key = []

        self._setup_architecture()

    def _setup_architecture(self):
        """
        Configures the internal pipelines and estimator based on the topology.
        """
        # Resolve feature columns from keys
        self.feature_cols_by_key = [FEATURE_GROUPS[k] for k in self.feature_keys]

        # Flatten columns for single-stream topologies
        feature_cols_flat = [col for cols in self.feature_cols_by_key for col in cols]

        # Initialize Estimator (LDA)
        self.estimator = LinearDiscriminantAnalysis(
            solver=LDA_SOLVER, shrinkage=self.shrinkage
        )

        # Setup Pipelines based on Topology
        if self.topology == "marginal":
            p = make_marginal_pipeline()
            self.pipelines.append((p, feature_cols_flat))

        elif self.topology == "rotational":
            p = make_alignment_pipeline(random_state=self.random_state)
            self.pipelines.append((p, feature_cols_flat))

        elif self.topology == "robust":
            p = make_robust_pipeline(
                n_quantiles=QUANTILE_N_QUANTILES,
                output_distribution=QUANTILE_OUTPUT_DIST,
                random_state=self.random_state,
            )
            self.pipelines.append((p, feature_cols_flat))

        elif self.topology == "polynomial":
            p = make_polynomial_pipeline(
                degree=POLY_DEGREE, random_state=self.random_state
            )
            self.pipelines.append((p, feature_cols_flat))

        elif self.topology == "pairwise_interaction":
            # Expecting exactly 2 feature keys for Pairwise (e.g., Margin + Texture)
            if len(self.feature_keys) != 2:
                raise ValueError(
                    f"Pairwise topology requires exactly 2 feature keys, got {len(self.feature_keys)}"
                )

            # Stream 1: Bottleneck for Feature Set 1
            p1 = make_bottleneck_pipeline(
                n_components=BOTTLENECK_K, random_state=self.random_state
            )
            self.pipelines.append((p1, self.feature_cols_by_key[0]))

            # Stream 2: Bottleneck for Feature Set 2
            p2 = make_bottleneck_pipeline(
                n_components=BOTTLENECK_K, random_state=self.random_state
            )
            self.pipelines.append((p2, self.feature_cols_by_key[1]))

            # Merge Step: Polynomial Expansion on the concatenated bottlenecks
            self.merge_pipeline = make_polynomial_pipeline(
                degree=POLY_DEGREE, random_state=self.random_state
            )

    def fit(self, X, y):
        """
        Fits the expert to the training data.

        Args:
            X (pd.DataFrame): Training features.
            y (array-like): Target labels.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Expert.fit expects a pandas DataFrame with feature columns."
            )

        transformed_parts = []

        # Process each stream
        for pipe, cols in self.pipelines:
            # Select specific columns for this stream
            X_subset = X[cols].values.astype(FLOAT_PRECISION)

            # Fit-Transform pipeline
            # Note: Bottleneck pipelines contain LDA, so they require 'y'
            # Standard pipelines ignore 'y' if not needed
            X_trans = pipe.fit_transform(X_subset, y)
            transformed_parts.append(X_trans)

        # Concatenate stream outputs
        if len(transformed_parts) > 1:
            X_combined = np.hstack(transformed_parts)
        else:
            X_combined = transformed_parts[0]

        # Apply Merge Pipeline (if any)
        if self.merge_pipeline:
            X_combined = self.merge_pipeline.fit_transform(X_combined, y)

        # Fit the final estimator
        self.estimator.fit(X_combined, y)
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Class probabilities.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expert.predict_proba expects a pandas DataFrame.")

        transformed_parts = []

        # Process each stream
        for pipe, cols in self.pipelines:
            X_subset = X[cols].values.astype(FLOAT_PRECISION)
            X_trans = pipe.transform(X_subset)
            transformed_parts.append(X_trans)

        # Concatenate stream outputs
        if len(transformed_parts) > 1:
            X_combined = np.hstack(transformed_parts)
        else:
            X_combined = transformed_parts[0]

        # Apply Merge Pipeline (if any)
        if self.merge_pipeline:
            X_combined = self.merge_pipeline.transform(X_combined)

        return self.estimator.predict_proba(X_combined)


def get_expert_pool():
    """
    Generates the library of candidate experts defined in config.py.
    Creates an Expert instance for each configuration and each shrinkage candidate.

    Returns:
        list[Expert]: A list of initialized Expert objects.
    """
    pool = []

    for key, config in EXPERTS_CONFIG.items():
        # For each expert configuration, create variants for each shrinkage candidate
        for shrinkage in LDA_SHRINKAGE_CANDIDATES:
            # Create a unique name for this specific instantiation
            expert_name = f"{key}_s{shrinkage}"

            expert = Expert(
                name=expert_name,
                group=config["group"],
                topology=config["topology"],
                feature_keys=config["features"],
                shrinkage=shrinkage,
            )
            pool.append(expert)

    return pool
