from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("pipeline_factory")


def create_model_pipeline(
    schema, pca_components=None, n_estimators=None, base_params=None
):
    """
    Constructs the Multi-Field Asymmetric Dual-Backbone Ensemble (MF-ADBE) pipeline.

    This pipeline implements Early Fusion by processing distinct feature views
    (Title, Body, Global, Meta) separately before concatenating them for the classifier.

    Args:
        schema (dict): Feature schema mapping view names to index ranges (start, end).
                       Must contain keys: 'title', 'body', 'global', 'meta'.
        pca_components (int, optional): Number of PCA components for the global view.
                                        Defaults to Config.PCA_COMPONENTS_AUX.
        n_estimators (int, optional): Number of estimators for Bagging.
                                      Defaults to Config.N_BAGGING_ESTIMATORS.
        base_params (dict, optional): Parameters for the base LogisticRegression estimator
                                      (e.g., {'C': 1.0, 'class_weight': 'balanced'}).

    Returns:
        sklearn.pipeline.Pipeline: The compiled scikit-learn pipeline ready for fitting.
    """
    # Set defaults based on Config if not provided
    if pca_components is None:
        pca_components = Config.PCA_COMPONENTS_AUX
    if n_estimators is None:
        n_estimators = Config.N_BAGGING_ESTIMATORS
    if base_params is None:
        base_params = {}

    logger.info(
        f"Building MF-ADBE Pipeline. PCA: {pca_components}, Bagging Est: {n_estimators}"
    )

    # -------------------------------------------------------------------------
    # 1. Define Column Slices
    # -------------------------------------------------------------------------
    # The schema provides (start, end) tuples. We generate explicit index lists
    # for ColumnTransformer.
    try:
        idx_title = list(range(schema["title"][0], schema["title"][1]))
        idx_body = list(range(schema["body"][0], schema["body"][1]))
        idx_global = list(range(schema["global"][0], schema["global"][1]))
        idx_meta = list(range(schema["meta"][0], schema["meta"][1]))
    except KeyError as e:
        logger.error(f"Schema missing required key: {e}")
        raise ValueError(f"Invalid schema provided. Missing view: {e}")

    # -------------------------------------------------------------------------
    # 2. Define Transformers for Each View
    # -------------------------------------------------------------------------

    # View 1: Title Semantics (High Res) -> L2 Normalization
    # Projects embeddings onto the hypersphere to focus on directional semantics.
    title_transformer = Normalizer(norm="l2")

    # View 2: Body Semantics (High Res) -> L2 Normalization
    body_transformer = Normalizer(norm="l2")

    # View 3: Global Context (Low Res) -> PCA -> L2 Normalization
    # Compresses the 768d MPNet embedding to capture dominant semantic variance
    # and "world knowledge" without overfitting.
    global_transformer = Pipeline(
        [
            ("pca", PCA(n_components=pca_components, random_state=Config.SEED)),
            ("norm", Normalizer(norm="l2")),
        ]
    )

    # View 4: Metadata -> RankGauss (QuantileTransformer)
    # Enforces a normal distribution to align numerical metadata with the
    # normalized embedding space, neutralizing outliers.
    meta_transformer = QuantileTransformer(
        output_distribution="normal", random_state=Config.SEED
    )

    # Combine all views via ColumnTransformer (Early Fusion)
    # n_jobs=-1 allows parallel processing of transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("title_view", title_transformer, idx_title),
            ("body_view", body_transformer, idx_body),
            ("global_view", global_transformer, idx_global),
            ("meta_view", meta_transformer, idx_meta),
        ],
        n_jobs=-1,
    )

    # -------------------------------------------------------------------------
    # 3. Define Classifier
    # -------------------------------------------------------------------------

    # Base Estimator: Logistic Regression (Ridge)
    # We use 'lbfgs' which supports L2 penalty and is robust for dense features.
    lr_defaults = {
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 2000,  # Increased to ensure convergence
        "random_state": Config.SEED,
    }

    # Override defaults with any provided base_params (e.g., C, class_weight)
    lr_params = {**lr_defaults, **base_params}

    base_clf = LogisticRegression(**lr_params)

    # Ensemble: Bagging Classifier
    # Wraps the linear model to reduce variance and improve stability.
    classifier = BaggingClassifier(
        estimator=base_clf,
        n_estimators=n_estimators,
        random_state=Config.SEED,
        n_jobs=-1,  # Parallelize training of ensemble members
    )

    # -------------------------------------------------------------------------
    # 4. Assemble Full Pipeline
    # -------------------------------------------------------------------------
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    return pipeline
