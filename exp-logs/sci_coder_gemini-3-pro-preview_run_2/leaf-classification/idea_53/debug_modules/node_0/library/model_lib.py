import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from library.config import LDA_SHRINKAGE_CANDIDATES
from library.transformations import (
    make_global_marginal_pipeline,
    make_global_rotational_pipeline,
    make_stratified_rotational_pipeline,
    make_discriminative_interaction_pipeline,
    make_poly_physical_pipeline,
)


class Float64LDA(LinearDiscriminantAnalysis):
    """
    A wrapper around sklearn's LinearDiscriminantAnalysis that enforces
    float64 precision for inputs to minimize numerical noise.
    """

    def __init__(
        self,
        solver="lsqr",
        shrinkage=None,
        priors=None,
        n_components=None,
        store_covariance=False,
        tol=1e-4,
        covariance_estimator=None,
    ):
        super().__init__(
            solver=solver,
            shrinkage=shrinkage,
            priors=priors,
            n_components=n_components,
            store_covariance=store_covariance,
            tol=tol,
            covariance_estimator=covariance_estimator,
        )

    def fit(self, X, y):
        """
        Fit LinearDiscriminantAnalysis model according to the given training data and parameters.
        Casts X to float64 before fitting.
        """
        X_64 = np.asarray(X, dtype=np.float64)
        return super().fit(X_64, y)

    def predict_proba(self, X):
        """
        Estimate probability.
        Casts X to float64 before prediction.
        """
        X_64 = np.asarray(X, dtype=np.float64)
        return super().predict_proba(X_64)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Casts X to float64 before prediction.
        """
        X_64 = np.asarray(X, dtype=np.float64)
        return super().predict(X_64)


class Expert:
    """
    A container for a specific model configuration within the ensemble.
    Holds the pipeline (preprocessing + classifier) and the key for the
    feature view it consumes.
    """

    def __init__(self, name, pipeline, view_name):
        self.name = name
        self.pipeline = pipeline
        self.view_name = view_name

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def __repr__(self):
        return f"Expert(name='{self.name}', view='{self.view_name}')"


def generate_expert_library():
    """
    Generates a list of Expert objects covering the defined topologies and
    hyperparameter candidates.

    Topologies:
    A. Global Marginal Anchors (Global View)
    B. Global Rotational Experts (Global View)
    C. Stratified Rotational Experts (Global View - split internally)
    D. Discriminative-Interaction Experts (Global View)
    E. Polynomial Physical Experts (Morphometric View)

    Returns:
        list[Expert]: A list of instantiated, untrained Expert objects.
    """
    experts = []

    # Helper to create LDA with specific shrinkage
    def get_lda(shrinkage):
        return Float64LDA(solver="lsqr", shrinkage=shrinkage)

    # -------------------------------------------------------------------------
    # Topology A: Global Marginal Anchors
    # -------------------------------------------------------------------------
    # Uses Global View (192 features)
    for shrinkage in LDA_SHRINKAGE_CANDIDATES:
        pipeline = make_global_marginal_pipeline()
        pipeline.steps.append(("clf", get_lda(shrinkage)))

        experts.append(
            Expert(
                name=f"Topo_A_GlobalMarginal_Shrinkage_{shrinkage}",
                pipeline=pipeline,
                view_name="global",
            )
        )

    # -------------------------------------------------------------------------
    # Topology B: Global Rotational Experts
    # -------------------------------------------------------------------------
    # Uses Global View (192 features)
    for shrinkage in LDA_SHRINKAGE_CANDIDATES:
        pipeline = make_global_rotational_pipeline()
        pipeline.steps.append(("clf", get_lda(shrinkage)))

        experts.append(
            Expert(
                name=f"Topo_B_GlobalRotational_Shrinkage_{shrinkage}",
                pipeline=pipeline,
                view_name="global",
            )
        )

    # -------------------------------------------------------------------------
    # Topology C: Stratified Rotational Experts
    # -------------------------------------------------------------------------
    # Uses Global View (192 features), split internally by the pipeline
    for shrinkage in LDA_SHRINKAGE_CANDIDATES:
        pipeline = make_stratified_rotational_pipeline()
        pipeline.steps.append(("clf", get_lda(shrinkage)))

        experts.append(
            Expert(
                name=f"Topo_C_StratifiedRotational_Shrinkage_{shrinkage}",
                pipeline=pipeline,
                view_name="global",
            )
        )

    # -------------------------------------------------------------------------
    # Topology D: Discriminative-Interaction Experts
    # -------------------------------------------------------------------------
    # Uses Global View (192 features)
    for shrinkage in LDA_SHRINKAGE_CANDIDATES:
        pipeline = make_discriminative_interaction_pipeline()
        pipeline.steps.append(("clf", get_lda(shrinkage)))

        experts.append(
            Expert(
                name=f"Topo_D_DiscriminativeInteraction_Shrinkage_{shrinkage}",
                pipeline=pipeline,
                view_name="global",
            )
        )

    # -------------------------------------------------------------------------
    # Topology E: Polynomial Physical Experts
    # -------------------------------------------------------------------------
    # Uses Morphometric View (Hu Moments + Geometric Scalars)
    for shrinkage in LDA_SHRINKAGE_CANDIDATES:
        pipeline = make_poly_physical_pipeline()
        pipeline.steps.append(("clf", get_lda(shrinkage)))

        experts.append(
            Expert(
                name=f"Topo_E_PolyPhysical_Shrinkage_{shrinkage}",
                pipeline=pipeline,
                view_name="morph",
            )
        )

    return experts
