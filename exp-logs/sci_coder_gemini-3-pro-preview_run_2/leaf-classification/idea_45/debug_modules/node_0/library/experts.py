import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import clone
from library.pipelines import (
    get_marginal_pipeline,
    get_rotational_pipeline,
    get_polynomial_pipeline,
)
from library.config import (
    ANCHOR_SHRINKAGE_VALUES,
    ANCHOR_SOLVERS,
    FLOAT_PRECISION,
    PROB_CLIP_MIN,
    PROB_CLIP_MAX,
)


class Expert:
    """
    A wrapper class for a specific probabilistic model configuration.
    Combines a preprocessing pipeline with a Linear Discriminant Analysis classifier.
    """

    def __init__(self, name, pipeline_factory, solver_params, feature_type):
        """
        Args:
            name (str): Unique identifier for this expert.
            pipeline_factory (callable): Function that returns a scikit-learn Pipeline.
            solver_params (dict): Dictionary of parameters for LinearDiscriminantAnalysis.
            feature_type (str): 'global' or 'physical', indicating which dataset view to use.
        """
        self.name = name
        self.feature_type = feature_type
        self.solver_params = solver_params

        # Construct the full pipeline
        # 1. Get the preprocessing steps
        self.pipeline = pipeline_factory()

        # 2. Append the Classifier
        # We use LDA as the core density estimator
        lda = LinearDiscriminantAnalysis(**solver_params)
        self.pipeline.steps.append(("clf", lda))

    def fit(self, X, y):
        """
        Fits the expert pipeline to the training data.

        Args:
            X (np.ndarray): Feature matrix (float64).
            y (np.ndarray): Target vector.
        """
        # Ensure precision
        X = X.astype(FLOAT_PRECISION)
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.

        Args:
            X (np.ndarray): Feature matrix (float64).

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        # Ensure precision
        X = X.astype(FLOAT_PRECISION)

        # Get raw probabilities
        probs = self.pipeline.predict_proba(X)

        # Apply strict clipping to avoid log-loss extremes
        # max(min(p, 1-1e-15), 1e-15)
        probs = np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)

        return probs

    def __repr__(self):
        return f"Expert(name='{self.name}', type='{self.feature_type}')"


def build_expert_library():
    """
    Constructs the library of candidate experts for the ensemble.

    Returns:
        list[Expert]: A list of initialized Expert objects.
    """
    experts = []

    # =========================================================================
    # Group A: Marginal Statistical Anchors (The Baseline)
    # Input: Global View
    # Pipeline: Marginal Gaussianization (PowerTransformer)
    # Algorithm: LDA with Fixed Shrinkage
    # =========================================================================
    for solver in ANCHOR_SOLVERS:  # ['lsqr', 'eigen']
        for shrinkage in ANCHOR_SHRINKAGE_VALUES:  # [0.001, 0.01]
            name = f"GroupA_Marginal_{solver}_shrink{shrinkage}"
            params = {"solver": solver, "shrinkage": shrinkage}

            expert = Expert(
                name=name,
                pipeline_factory=get_marginal_pipeline,
                solver_params=params,
                feature_type="global",
            )
            experts.append(expert)

    # =========================================================================
    # Group B: Rotational Statistical Experts (The Innovation)
    # Input: Global View
    # Pipeline: Rotation (PCA no-whiten) -> Gaussianization
    # Algorithm: LDA with Fixed Shrinkage
    # =========================================================================
    for solver in ANCHOR_SOLVERS:
        for shrinkage in ANCHOR_SHRINKAGE_VALUES:
            name = f"GroupB_Rotational_{solver}_shrink{shrinkage}"
            params = {"solver": solver, "shrinkage": shrinkage}

            expert = Expert(
                name=name,
                pipeline_factory=get_rotational_pipeline,
                solver_params=params,
                feature_type="global",
            )
            experts.append(expert)

    # =========================================================================
    # Group C: Polynomial Physical Experts (The Synergy)
    # Input: Physical View (Morphometrics)
    # Pipeline: Polynomial Expansion -> Gaussianization
    # Algorithm: LDA with Ledoit-Wolf Shrinkage ('auto')
    # =========================================================================
    # We use 'auto' shrinkage which corresponds to the Ledoit-Wolf lemma
    # This is robust for the expanded feature space.

    # Solver 'lsqr' supports 'auto' shrinkage
    name_lsqr = "GroupC_PolyPhysical_lsqr_auto"
    params_lsqr = {"solver": "lsqr", "shrinkage": "auto"}
    expert_lsqr = Expert(
        name=name_lsqr,
        pipeline_factory=get_polynomial_pipeline,
        solver_params=params_lsqr,
        feature_type="physical",
    )
    experts.append(expert_lsqr)

    # Solver 'eigen' supports 'auto' shrinkage
    name_eigen = "GroupC_PolyPhysical_eigen_auto"
    params_eigen = {"solver": "eigen", "shrinkage": "auto"}
    expert_eigen = Expert(
        name=name_eigen,
        pipeline_factory=get_polynomial_pipeline,
        solver_params=params_eigen,
        feature_type="physical",
    )
    experts.append(expert_eigen)

    return experts
