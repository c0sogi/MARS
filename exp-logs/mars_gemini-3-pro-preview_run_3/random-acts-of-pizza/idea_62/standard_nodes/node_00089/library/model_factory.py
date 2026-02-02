import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("model_factory")


class ModelFactory:
    """
    Factory class to instantiate the 7 Level-1 base learners and the Level-2 meta-learner
    for the Conservative Granular Hept-View Stacking Ensemble.

    Retrieves hyperparameters from the Config class and allows for dynamic overrides.
    """

    @staticmethod
    def get_lexical_bagger(**kwargs):
        """
        Returns the Granular Lexical Bagger (Random Forest).
        Operates on Sparse Lexical Features (TF-IDF of Title + Body).
        """
        params = Config.LEXICAL_BAGGER_PARAMS.copy()
        params.update(kwargs)
        logger.info(f"Initializing Lexical Bagger (RF) with params: {params}")
        return RandomForestClassifier(**params)

    @staticmethod
    def get_community_bagger(**kwargs):
        """
        Returns the Community Bagger (Random Forest).
        Operates on Sparse Behavioral Features (TF-IDF of Subreddit History).
        """
        params = Config.COMMUNITY_BAGGER_PARAMS.copy()
        params.update(kwargs)
        logger.info(f"Initializing Community Bagger (RF) with params: {params}")
        return RandomForestClassifier(**params)

    @staticmethod
    def get_semantic_booster(scale_pos_weight=None, **kwargs):
        """
        Returns the Semantic Booster (XGBoost).
        Operates on Dense Semantic Features (Embeddings).

        Args:
            scale_pos_weight (float, optional): Weight for positive class to handle imbalance.
                                                If None, uses the value in kwargs or default.
        """
        params = Config.SEMANTIC_BOOSTER_PARAMS.copy()
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
        params.update(kwargs)

        logger.info(f"Initializing Semantic Booster (XGB) with params: {params}")
        return xgb.XGBClassifier(**params)

    @staticmethod
    def get_semantic_gradient(**kwargs):
        """
        Returns the Semantic Gradient (LightGBM).
        Operates on Dense Semantic Features (Embeddings).
        """
        params = Config.SEMANTIC_GRADIENT_PARAMS.copy()
        params.update(kwargs)
        logger.info(f"Initializing Semantic Gradient (LGBM) with params: {params}")
        return lgb.LGBMClassifier(**params)

    @staticmethod
    def get_semantic_bagger(**kwargs):
        """
        Returns the Semantic Bagger (Random Forest).
        Operates on Dense Semantic Features (Embeddings).
        """
        params = Config.SEMANTIC_BAGGER_PARAMS.copy()
        params.update(kwargs)
        logger.info(f"Initializing Semantic Bagger (RF) with params: {params}")
        return RandomForestClassifier(**params)

    @staticmethod
    def get_metadata_anchor(**kwargs):
        """
        Returns the Metadata Anchor (Logistic Regression).
        Operates on Augmented Global Metadata.
        """
        params = Config.METADATA_ANCHOR_PARAMS.copy()
        params.update(kwargs)
        logger.info(f"Initializing Metadata Anchor (LR) with params: {params}")
        return LogisticRegression(**params)

    @staticmethod
    def get_temporal_booster(**kwargs):
        """
        Returns the Temporal Booster (LightGBM).
        Operates on Augmented Global Metadata.
        """
        params = Config.TEMPORAL_BOOSTER_PARAMS.copy()
        params.update(kwargs)
        logger.info(f"Initializing Temporal Booster (LGBM) with params: {params}")
        return lgb.LGBMClassifier(**params)

    @staticmethod
    def get_meta_learner(**kwargs):
        """
        Returns the Level-2 Meta Learner (Logistic Regression).
        Operates on the stack of probability predictions from Level-1 models.
        """
        params = Config.META_LEARNER_PARAMS.copy()
        params.update(kwargs)
        logger.info(f"Initializing Meta Learner (LR) with params: {params}")
        return LogisticRegression(**params)
