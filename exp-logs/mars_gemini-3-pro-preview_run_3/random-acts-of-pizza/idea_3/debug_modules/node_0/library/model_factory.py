from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library.config import Config


class ModelFactory:
    """
    Factory class for instantiating the machine learning models used in the
    Multi-Paradigm Stacking Ensemble.

    This class provides methods to create the specific Level 1 base learners
    (Lexical Bagger, Semantic Bagger, Semantic Booster) and the Level 2
    Meta-Learner, using parameters defined in the Config while allowing
    runtime overrides.
    """

    @staticmethod
    def create_lexical_bagger(**kwargs) -> RandomForestClassifier:
        """
        Creates the Lexical Bagger model (Random Forest) intended for the
        sparse TF-IDF feature view.

        Args:
            **kwargs: Arbitrary keyword arguments to override defaults in
                      Config.L1_RF_LEXICAL_PARAMS.

        Returns:
            RandomForestClassifier: The configured Random Forest model.
        """
        params = Config.L1_RF_LEXICAL_PARAMS.copy()
        params.update(kwargs)
        return RandomForestClassifier(**params)

    @staticmethod
    def create_semantic_bagger(**kwargs) -> RandomForestClassifier:
        """
        Creates the Semantic Bagger model (Random Forest) intended for the
        dense SBERT embedding feature view.

        Args:
            **kwargs: Arbitrary keyword arguments to override defaults in
                      Config.L1_RF_SEMANTIC_PARAMS.

        Returns:
            RandomForestClassifier: The configured Random Forest model.
        """
        params = Config.L1_RF_SEMANTIC_PARAMS.copy()
        params.update(kwargs)
        return RandomForestClassifier(**params)

    @staticmethod
    def create_semantic_booster(**kwargs) -> XGBClassifier:
        """
        Creates the Semantic Booster model (XGBoost) intended for the
        dense SBERT embedding feature view.

        Args:
            **kwargs: Arbitrary keyword arguments to override defaults in
                      Config.L1_XGB_SEMANTIC_PARAMS.

        Returns:
            XGBClassifier: The configured XGBoost model.
        """
        params = Config.L1_XGB_SEMANTIC_PARAMS.copy()
        params.update(kwargs)
        return XGBClassifier(**params)

    @staticmethod
    def create_meta_learner(**kwargs) -> LogisticRegression:
        """
        Creates the Meta-Learner (Logistic Regression) intended for the
        Level 2 Stacking layer.

        Args:
            **kwargs: Arbitrary keyword arguments to override defaults in
                      Config.L2_META_PARAMS.

        Returns:
            LogisticRegression: The configured Logistic Regression model.
        """
        params = Config.L2_META_PARAMS.copy()
        params.update(kwargs)
        return LogisticRegression(**params)
