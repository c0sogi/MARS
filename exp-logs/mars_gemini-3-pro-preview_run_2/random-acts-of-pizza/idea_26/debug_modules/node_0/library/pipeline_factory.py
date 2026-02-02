import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from library.config import Config


def _map_param_grid(raw_grid: dict, prefix: str = "clf__estimator__") -> dict:
    """
    Maps a raw parameter grid (e.g., for LogisticRegression) to the pipeline's
    nested parameter naming convention (e.g., inside BaggingClassifier).
    """
    mapped_grid = {}
    for key, values in raw_grid.items():
        mapped_grid[f"{prefix}{key}"] = values
    return mapped_grid


def create_branch_a_pipeline(
    embedding_dim: int, meta_dim: int, param_grid: dict = None
) -> GridSearchCV:
    """
    Constructs the pipeline for Branch A (MiniLM Backbone).

    Architecture:
    1. Preprocessing:
       - Embeddings (0 to embedding_dim): L2 Normalization.
       - Metadata (embedding_dim to end): QuantileTransformer (RankGauss).
    2. Classifier:
       - Bagged Logistic Regression.
    3. Tuning:
       - GridSearchCV over the base LogisticRegression parameters.

    Args:
        embedding_dim (int): Number of embedding features (384 for MiniLM).
        meta_dim (int): Number of metadata features.
        param_grid (dict, optional): Hyperparameter grid for LogisticRegression.
                                     Defaults to Config.LR_PARAM_GRID.

    Returns:
        GridSearchCV: A configured grid search object ready to be fitted.
    """
    # 1. Define Preprocessor
    # Slice(0, embedding_dim) selects embedding columns
    # Slice(embedding_dim, embedding_dim + meta_dim) selects metadata columns
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "emb_norm",
                Normalizer(norm="l2"),
                slice(0, embedding_dim),
            ),
            (
                "meta_scaler",
                QuantileTransformer(
                    output_distribution="normal", random_state=Config.SEED
                ),
                slice(embedding_dim, embedding_dim + meta_dim),
            ),
        ],
        remainder="drop",  # Drop any unexpected columns
    )

    # 2. Define Classifier
    # BaggingClassifier wrapping LogisticRegression
    # Note: In sklearn >= 1.2, 'base_estimator' is 'estimator'
    clf = BaggingClassifier(
        estimator=LogisticRegression(random_state=Config.SEED),
        n_estimators=Config.N_ESTIMATORS_BAGGING,
        random_state=Config.SEED,
        n_jobs=1,  # Avoid oversubscription; parallelization handled by GridSearchCV
    )

    # 3. Create Pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

    # 4. Configure Grid Search
    if param_grid is None:
        raw_grid = Config.LR_PARAM_GRID
    else:
        raw_grid = param_grid

    mapped_grid = _map_param_grid(raw_grid)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=mapped_grid,
        scoring="roc_auc",
        cv=5,  # Robust inner CV
        n_jobs=Config.N_JOBS,
        verbose=0,
    )

    return grid_search


def create_branch_b_pipeline(
    embedding_dim: int, meta_dim: int, param_grid: dict = None
) -> GridSearchCV:
    """
    Constructs the pipeline for Branch B (MPNet Backbone).

    Architecture:
    1. Preprocessing:
       - Embeddings (0 to embedding_dim): PCA (200 dim) -> L2 Normalization.
       - Metadata (embedding_dim to end): QuantileTransformer (RankGauss).
    2. Classifier:
       - Bagged Logistic Regression.
    3. Tuning:
       - GridSearchCV over the base LogisticRegression parameters.

    Args:
        embedding_dim (int): Number of embedding features (768 for MPNet).
        meta_dim (int): Number of metadata features.
        param_grid (dict, optional): Hyperparameter grid for LogisticRegression.
                                     Defaults to Config.LR_PARAM_GRID.

    Returns:
        GridSearchCV: A configured grid search object ready to be fitted.
    """
    # 1. Define Embedding Sub-Pipeline (PCA -> Norm)
    # PCA is fitted inside the CV loop via the Pipeline
    emb_pipeline = Pipeline(
        [
            (
                "pca",
                PCA(
                    n_components=Config.MODEL_B_PCA_COMPONENTS,
                    random_state=Config.SEED,
                ),
            ),
            ("norm", Normalizer(norm="l2")),
        ]
    )

    # 2. Define Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "emb_pca_norm",
                emb_pipeline,
                slice(0, embedding_dim),
            ),
            (
                "meta_scaler",
                QuantileTransformer(
                    output_distribution="normal", random_state=Config.SEED
                ),
                slice(embedding_dim, embedding_dim + meta_dim),
            ),
        ],
        remainder="drop",
    )

    # 3. Define Classifier
    clf = BaggingClassifier(
        estimator=LogisticRegression(random_state=Config.SEED),
        n_estimators=Config.N_ESTIMATORS_BAGGING,
        random_state=Config.SEED,
        n_jobs=1,
    )

    # 4. Create Pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

    # 5. Configure Grid Search
    if param_grid is None:
        raw_grid = Config.LR_PARAM_GRID
    else:
        raw_grid = param_grid

    mapped_grid = _map_param_grid(raw_grid)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=mapped_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=Config.N_JOBS,
        verbose=0,
    )

    return grid_search
