import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from library.config import Config


class ModelFactory:
    """
    Factory class for creating the Modality-Balanced Bagged Linear Ensemble.
    """

    @staticmethod
    def create_bagged_ensemble(
        C: float = 1.0,
        class_weight: str = None,
        n_estimators: int = Config.BAGGING_N_ESTIMATORS,
        random_state: int = Config.SEED,
        n_jobs: int = -1,
    ) -> BaggingClassifier:
        """
        Constructs a BaggingClassifier with a LogisticRegression base estimator.

        This architecture implements the linear core of the solution. By wrapping
        LogisticRegression in a BaggingClassifier, we stabilize the predictions
        and reduce variance, which is particularly important when combining
        high-dimensional text embeddings with dense metadata.

        Args:
            C (float): Inverse of regularization strength for Logistic Regression.
                       Smaller values specify stronger regularization.
                       Default is 1.0.
            class_weight (str or dict, optional): Weights associated with classes.
                                                  'balanced' or None.
            n_estimators (int): The number of base estimators in the ensemble.
                                Defaults to Config.BAGGING_N_ESTIMATORS.
            random_state (int): Seed used by the random number generator.
                                Defaults to Config.SEED.
            n_jobs (int): The number of jobs to run in parallel for both fit and predict.
                          Defaults to -1 (use all processors).

        Returns:
            BaggingClassifier: The configured ensemble model ready for training.
        """

        # Base Estimator: Logistic Regression
        # - solver='lbfgs': Standard solver, robust and memory efficient.
        # - max_iter=2000: Increased from default (100) to ensure convergence
        #   on the high-dimensional fused feature space (384 dim text + metadata).
        base_estimator = LogisticRegression(
            C=C,
            class_weight=class_weight,
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
        )

        # Bagging Ensemble
        # - max_samples=1.0: Use the full dataset size for each bootstrap sample.
        # - bootstrap=True: Standard bagging (sampling with replacement).
        # - estimator: The LogisticRegression instance defined above.
        model = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        return model
