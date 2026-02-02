import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, QuantileTransformer, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("pipeline_builder")


class PipelineBuilder:
    """
    Constructs the Discretized-Augmented Asymmetric Dual-Backbone Ensemble (DAADBE) pipeline.
    """

    @staticmethod
    def build_daadbe_pipeline(
        anchor_cols: list,
        aux_cols: list,
        continuous_cols: list,
        discrete_cols: list,
        pca_components: int = Config.PCA_COMPONENTS,
        n_bins: int = Config.N_BINS,
        bin_strategy: str = Config.BIN_STRATEGY,
        n_estimators: int = Config.N_ESTIMATORS,
        logistic_params: dict = None,
        random_state: int = Config.SEED,
    ) -> Pipeline:
        """
        Builds the Scikit-Learn pipeline.

        Args:
            anchor_cols (list): List of column names for the Anchor (MiniLM) embeddings.
            aux_cols (list): List of column names for the Aux (MPNet) embeddings.
            continuous_cols (list): List of continuous metadata column names.
            discrete_cols (list): List of discrete metadata column names.
            pca_components (int): Number of PCA components for the Aux view.
            n_bins (int): Number of bins for discretization.
            bin_strategy (str): Strategy for KBinsDiscretizer (e.g., 'quantile').
            n_estimators (int): Number of estimators for BaggingClassifier.
            logistic_params (dict): Parameters for the base LogisticRegression.
            random_state (int): Seed for reproducibility.

        Returns:
            Pipeline: The complete DAADBE pipeline.
        """

        # Default Logistic Regression parameters if none provided
        if logistic_params is None:
            logistic_params = {
                "solver": "lbfgs",
                "max_iter": Config.MAX_ITER,
                "class_weight": "balanced",
                "C": 1.0,
            }

        # Ensure max_iter is set if not present
        if "max_iter" not in logistic_params:
            logistic_params["max_iter"] = Config.MAX_ITER

        logger.info("Building DAADBE Pipeline...")
        logger.info(f"  Anchor Cols: {len(anchor_cols)}")
        logger.info(f"  Aux Cols: {len(aux_cols)} -> PCA({pca_components})")
        logger.info(f"  Continuous Cols: {len(continuous_cols)}")
        logger.info(
            f"  Discrete Cols: {len(discrete_cols)} -> KBins({n_bins}, {bin_strategy})"
        )

        # ---------------------------------------------------------
        # 1. Feature Engineering Blocks
        # ---------------------------------------------------------

        # View 1: Semantic Anchor (384d) -> L2 Normalization
        # Projects embeddings onto the hypersphere
        anchor_transformer = Normalizer(norm="l2")

        # View 2: Deep Semantics (768d) -> PCA(50) -> L2 Normalization
        # Compresses world knowledge and normalizes
        aux_transformer = Pipeline(
            [
                ("pca", PCA(n_components=pca_components, random_state=random_state)),
                ("norm", Normalizer(norm="l2")),
            ]
        )

        # View 3: Continuous Metadata -> RankGauss
        # Handles monotonic trends and outliers
        continuous_transformer = QuantileTransformer(
            output_distribution="normal", random_state=random_state
        )

        # View 4: Discretized Metadata -> Quantile Binning -> OneHot
        # Captures non-monotonic relationships (e.g., U-shaped curves)
        # encode='onehot-dense' ensures we get a dense array, avoiding sparse/dense mix issues
        discrete_transformer = KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy=bin_strategy,
            random_state=random_state,
        )

        # ---------------------------------------------------------
        # 2. Column Transformer (Early Fusion)
        # ---------------------------------------------------------
        preprocessor = ColumnTransformer(
            transformers=[
                ("anchor", anchor_transformer, anchor_cols),
                ("aux", aux_transformer, aux_cols),
                ("continuous", continuous_transformer, continuous_cols),
                ("discrete", discrete_transformer, discrete_cols),
            ],
            remainder="drop",  # Drop any columns not explicitly listed
        )

        # ---------------------------------------------------------
        # 3. Classifier (Ensemble)
        # ---------------------------------------------------------
        # Base Learner: Logistic Regression (Ridge)
        base_lr = LogisticRegression(random_state=random_state, **logistic_params)

        # Ensemble: Bagging to reduce variance
        classifier = BaggingClassifier(
            estimator=base_lr,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,  # Parallelize training
        )

        # ---------------------------------------------------------
        # 4. Final Pipeline
        # ---------------------------------------------------------
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", classifier)]
        )

        return pipeline
