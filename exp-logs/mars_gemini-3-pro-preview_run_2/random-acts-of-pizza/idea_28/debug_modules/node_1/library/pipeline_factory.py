import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import QuantileTransformer, Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("pipeline_factory")


class PipelineFactory:
    """
    Constructs the machine learning pipeline for the Dual-Resolution Semantic Early Fusion strategy.
    Integrates feature preprocessing (Metadata Scaling, Embedding Normalization, PCA)
    and the Bagged Logistic Regression classifier.
    """

    @staticmethod
    def create_pipeline(params: dict) -> Pipeline:
        """
        Creates a scikit-learn pipeline with specified hyperparameters.

        Args:
            params (dict): Hyperparameters for the Logistic Regression base estimator.
                           Expected keys: 'C', 'class_weight', 'penalty', 'solver', 'max_iter'.

        Returns:
            Pipeline: A configured scikit-learn pipeline ready for fitting.
        """
        # ---------------------------------------------------------------------
        # 1. Define Preprocessing Steps
        # ---------------------------------------------------------------------

        # View 3: Robust Metadata
        # Apply RankGauss (QuantileTransformer) to normalize distributions
        meta_transformer = QuantileTransformer(
            output_distribution="normal", random_state=Config.RANDOM_SEED
        )

        # View 1: Primary Semantics (High-Resolution MiniLM)
        # Apply L2 Normalization
        minilm_transformer = Normalizer(norm="l2")

        # View 2: Auxiliary Semantics (Low-Resolution MPNet)
        # Apply PCA for compression, then L2 Normalization
        # Note: Normalization happens AFTER PCA to ensure unit norm of the reduced features
        mpnet_transformer = Pipeline(
            [
                (
                    "pca",
                    PCA(n_components=Config.PCA_DIMS, random_state=Config.RANDOM_SEED),
                ),
                ("norm", Normalizer(norm="l2")),
            ]
        )

        # Column Transformer to route features to appropriate transformers
        # We use make_column_selector for embeddings to handle dynamic column names
        preprocessor = ColumnTransformer(
            transformers=[
                ("meta", meta_transformer, Config.METADATA_COLS),
                (
                    "minilm",
                    minilm_transformer,
                    make_column_selector(pattern="^minilm_"),
                ),
                ("mpnet", mpnet_transformer, make_column_selector(pattern="^mpnet_")),
            ],
            remainder="drop",  # Drop any columns not explicitly handled
            n_jobs=-1,
        )

        # ---------------------------------------------------------------------
        # 2. Define Classifier
        # ---------------------------------------------------------------------

        # Extract hyperparameters with defaults
        c_val = params.get("C", 1.0)
        class_weight = params.get("class_weight", None)
        penalty = params.get("penalty", "l2")
        solver = params.get("solver", "lbfgs")
        max_iter = params.get("max_iter", 1000)

        # Base Estimator: Logistic Regression
        base_estimator = LogisticRegression(
            C=c_val,
            class_weight=class_weight,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=Config.RANDOM_SEED,
        )

        # Ensemble: Bagging Classifier
        # Reduces variance of the linear model
        clf = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=Config.N_ESTIMATORS,
            random_state=Config.RANDOM_SEED,
            n_jobs=-1,  # Parallelize bagging
        )

        # ---------------------------------------------------------------------
        # 3. Assemble Pipeline
        # ---------------------------------------------------------------------
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

        return pipeline
