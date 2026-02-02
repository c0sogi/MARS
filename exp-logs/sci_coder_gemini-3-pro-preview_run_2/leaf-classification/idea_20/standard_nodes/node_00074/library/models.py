import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.base import clone


def get_expert_pipeline(expert_name, random_state=42):
    """
    Returns an untrained Scikit-Learn Pipeline for the specified expert architecture.

    Args:
        expert_name (str): One of 'Expert_A', 'Expert_B', 'Expert_C'.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # Note: Input data is already Gaussianized by library.utils.preprocess_data
    # We remove PowerTransformer from here to avoid redundant double-transformation.
    # Cite solution_lesson_node_00072

    if expert_name == "Expert_A":
        # Expert A: Gaussianized Generative Anchor
        # Architecture: LDA (Input is already PT transformed)
        # Config: LDA with Ledoit-Wolf shrinkage (solver='lsqr', shrinkage='auto')
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        return Pipeline([("lda", lda)])

    elif expert_name == "Expert_B":
        # Expert B: Metric-Discriminative Expert
        # Architecture: NCA -> Logistic Regression (Input is already PT transformed)
        # Phase 1 Config: LogisticRegressionCV for hyperparameter search

        # NCA Config: 99 components (n_classes), init='pca', max_iter=500
        nca = NeighborhoodComponentsAnalysis(
            n_components=99, init="pca", max_iter=500, random_state=random_state
        )

        # Logistic Regression Config
        # Cs: Dense Grid np.logspace(-4, 4, 100)
        # Scoring: neg_log_loss
        # Solver: lbfgs with L2 penalty
        # Multi_class: multinomial (to optimize global log loss)
        lr_cv = LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 100),
            cv=5,
            scoring="neg_log_loss",
            solver="lbfgs",
            penalty="l2",
            multi_class="multinomial",
            max_iter=1000,
            n_jobs=-1,
            random_state=random_state,
        )

        return Pipeline([("nca", nca), ("lr", lr_cv)])

    elif expert_name == "Expert_C":
        # Expert C: Metric-Generative Expert
        # Architecture: NCA -> LDA (Input is already PT transformed)

        nca = NeighborhoodComponentsAnalysis(
            n_components=99, init="pca", max_iter=500, random_state=random_state
        )

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        return Pipeline([("nca", nca), ("lda", lda)])

    else:
        raise ValueError(f"Unknown expert name: {expert_name}")


def get_fixed_pipeline(trained_pipeline):
    """
    Converts a trained pipeline (Phase 1) into a fixed pipeline (Phase 2).
    Specifically, it extracts the optimal C from LogisticRegressionCV and
    replaces it with a standard LogisticRegression with fixed C.
    For other steps (PT, NCA, LDA), it returns fresh untrained clones.

    Args:
        trained_pipeline (sklearn.pipeline.Pipeline): The fitted pipeline from Phase 1.

    Returns:
        sklearn.pipeline.Pipeline: A new untrained pipeline with fixed hyperparameters.
    """
    new_steps = []

    for name, step in trained_pipeline.steps:
        if isinstance(step, LogisticRegressionCV):
            # Extract the best C found during CV
            # For multi_class='multinomial', C_ is typically shape (1,) containing the best C
            best_c = step.C_[0]

            # Create a standard LogisticRegression with this fixed C
            fixed_lr = LogisticRegression(
                C=best_c,
                solver="lbfgs",
                penalty="l2",
                multi_class="multinomial",
                max_iter=1000,
                n_jobs=-1,
                random_state=step.random_state,
            )
            new_steps.append((name, fixed_lr))
        else:
            # For other steps (PowerTransformer, NCA, LDA), we want fresh instances
            # clone() creates a new estimator with the same parameters but not fitted
            new_steps.append((name, clone(step)))

    return Pipeline(new_steps)
