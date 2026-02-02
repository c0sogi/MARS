import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from library.config import Config


class ModelFactory:
    """
    Factory class to instantiate configured models for the Hex-View Stacking Ensemble.
    """

    @staticmethod
    def get_base_learner(learner_name, **kwargs):
        """
        Returns an instance of a Level 1 base learner configured with parameters from Config.

        Args:
            learner_name (str): The identifier for the model (e.g., 'lexical_bagger').
            **kwargs: Optional arguments to override default configuration (e.g., n_estimators).

        Returns:
            model: An instantiated scikit-learn compatible model.
        """
        # 1. Lexical Bagger (Sparse Text -> Random Forest)
        if learner_name == "lexical_bagger":
            params = copy.deepcopy(Config.PARAMS_LEXICAL_BAGGER)
            params.update(kwargs)
            return RandomForestClassifier(**params)

        # 2. Community Bagger (Sparse History -> Random Forest)
        elif learner_name == "community_bagger":
            params = copy.deepcopy(Config.PARAMS_COMMUNITY_BAGGER)
            params.update(kwargs)
            return RandomForestClassifier(**params)

        # 3. Semantic Booster (Dense Embeddings -> XGBoost)
        elif learner_name == "semantic_booster":
            params = copy.deepcopy(Config.PARAMS_SEMANTIC_BOOSTER)
            params.update(kwargs)
            return XGBClassifier(**params)

        # 4. Semantic Bagger (Dense Embeddings -> Random Forest)
        elif learner_name == "semantic_bagger":
            params = copy.deepcopy(Config.PARAMS_SEMANTIC_BAGGER)
            params.update(kwargs)
            return RandomForestClassifier(**params)

        # 5. Temporal Booster (Metadata -> LightGBM)
        elif learner_name == "temporal_booster":
            params = copy.deepcopy(Config.PARAMS_TEMPORAL_BOOSTER)
            params.update(kwargs)
            return LGBMClassifier(**params)

        # 6. Metadata Anchor (Metadata -> Logistic Regression)
        elif learner_name == "metadata_anchor":
            params = copy.deepcopy(Config.PARAMS_METADATA_ANCHOR)
            params.update(kwargs)
            return LogisticRegression(**params)

        else:
            raise ValueError(f"Unknown base learner name: {learner_name}")

    @staticmethod
    def get_meta_learner(**kwargs):
        """
        Returns an instance of the Level 2 Meta-Learner (Logistic Regression).

        Args:
            **kwargs: Optional arguments to override default configuration.

        Returns:
            model: An instantiated LogisticRegression model.
        """
        params = copy.deepcopy(Config.PARAMS_META_LEARNER)
        params.update(kwargs)
        return LogisticRegression(**params)
