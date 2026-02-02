import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from library import config


class ViewFactory:
    """
    Factory class to generate scikit-learn preprocessing pipelines for the
    Compressed-Interaction Precision-Generative Ensemble (CIPGE).

    Each view corresponds to a specific transformation strategy designed to
    expose different structural properties of the data (Marginal, Rotational,
    Physical, and Compressed-Interaction).
    """

    @staticmethod
    def get_pipeline(view_code):
        """
        Constructs and returns the preprocessing pipeline for a specific view.

        Args:
            view_code (str): The identifier for the view ('A', 'B', 'C', or 'D').

        Returns:
            sklearn.pipeline.Pipeline: The constructed pipeline.

        Raises:
            ValueError: If an invalid view_code is provided.
        """
        view_code = view_code.upper()

        if view_code == "A":
            return ViewFactory._create_view_a()
        elif view_code == "B":
            return ViewFactory._create_view_b()
        elif view_code == "C":
            return ViewFactory._create_view_c()
        elif view_code == "D":
            return ViewFactory._create_view_d()
        else:
            raise ValueError(
                f"Unknown view code: {view_code}. Expected 'A', 'B', 'C', or 'D'."
            )

    @staticmethod
    def _create_view_a():
        """
        View A: Marginal Statistical Anchors (The Baseline)
        Pipeline: Global Features -> PowerTransformer

        Role: Preserves the state-of-the-art baseline, utilizing the full
        high-dimensional statistical signal with robust covariance estimation.
        """
        steps = [
            (
                "power_transform",
                PowerTransformer(method="yeo-johnson", standardize=True),
            )
        ]
        return Pipeline(steps)

    @staticmethod
    def _create_view_b():
        """
        View B: Rotational Statistical Experts (The Alignment)
        Pipeline: Global Features -> PowerTransformer -> PCA(whiten=False) -> PowerTransformer

        Role: Aligns the data with its principal axes of variation without the
        noise-amplifying effects of Whitening.
        """
        steps = [
            # Initial Gaussianization
            ("pt_1", PowerTransformer(method="yeo-johnson", standardize=True)),
            # Rotation (keep all components, no whitening)
            (
                "pca_rot",
                PCA(n_components=None, whiten=False, random_state=config.RANDOM_STATE),
            ),
            # Re-Gaussianization after rotation
            ("pt_2", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
        return Pipeline(steps)

    @staticmethod
    def _create_view_c():
        """
        View C: Polynomial Physical Experts (The Domain Signal)
        Pipeline: Morphometrics -> PowerTransformer -> Poly(2) -> PowerTransformer

        Role: Captures non-linear physical constraints (e.g., Solidity * Eccentricity)
        from the raw images.
        """
        steps = [
            # Initial Gaussianization of physical scalars
            ("pt_1", PowerTransformer(method="yeo-johnson", standardize=True)),
            # Polynomial expansion (Quadratic)
            ("poly", PolynomialFeatures(degree=config.POLY_DEGREE, include_bias=False)),
            # Re-Gaussianization of the expanded feature space
            ("pt_2", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
        return Pipeline(steps)

    @staticmethod
    def _create_view_d():
        """
        View D: Compressed-Interaction Global Experts (The Innovation)
        Pipeline: Global Features -> PowerTransformer -> PCA(k) -> Poly(2) -> PowerTransformer

        Role: Explicitly models quadratic interactions between global features by
        first compressing to a dense latent space to avoid the curse of dimensionality.
        """
        steps = [
            # Initial Gaussianization
            ("pt_1", PowerTransformer(method="yeo-johnson", standardize=True)),
            # Compression to latent space
            (
                "pca_compress",
                PCA(
                    n_components=config.PCA_COMPONENTS_INTERACTION,
                    whiten=False,
                    random_state=config.RANDOM_STATE,
                ),
            ),
            # Quadratic expansion in latent space
            ("poly", PolynomialFeatures(degree=config.POLY_DEGREE, include_bias=False)),
            # Re-Gaussianization of the interaction features
            ("pt_2", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
        return Pipeline(steps)

    @staticmethod
    def get_feature_type(view_code):
        """
        Returns the type of input features required for the view.

        Args:
            view_code (str): 'A', 'B', 'C', or 'D'.

        Returns:
            str: 'global' or 'morphometric'
        """
        view_code = view_code.upper()
        if view_code in ["A", "B", "D"]:
            return "global"
        elif view_code == "C":
            return "morphometric"
        else:
            raise ValueError(f"Unknown view code: {view_code}")
