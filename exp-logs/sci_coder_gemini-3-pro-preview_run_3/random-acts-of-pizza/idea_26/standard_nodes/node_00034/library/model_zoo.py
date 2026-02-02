import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from library import config


class ModelFactory:
    """
    Factory class to instantiate specific learner configurations for the
    Pent-View architecture.
    """

    @staticmethod
    def get_lexical_bagger():
        """
        Returns the Sparse Lexical Branch learner (Random Forest).
        Trained on TF-IDF vectors of request text + Metadata.
        """
        return RandomForestClassifier(**config.LEXICAL_BAGGER_PARAMS)

    @staticmethod
    def get_community_bagger():
        """
        Returns the Sparse Behavioral Branch learner (Random Forest).
        Trained on TF-IDF vectors of subreddit history + Metadata.
        """
        return RandomForestClassifier(**config.COMMUNITY_BAGGER_PARAMS)

    @staticmethod
    def get_semantic_booster():
        """
        Returns the Dense Semantic Branch booster (XGBoost).
        Trained on Dense Embeddings + Metadata.

        Note: Early stopping parameters (config.SEMANTIC_BOOSTER_FIT_PARAMS)
        should be passed to the fit() method by the training pipeline.
        """
        return xgb.XGBClassifier(**config.SEMANTIC_BOOSTER_PARAMS)

    @staticmethod
    def get_semantic_bagger():
        """
        Returns the Dense Semantic Branch bagger (Random Forest).
        Trained on Dense Embeddings + Metadata.
        """
        return RandomForestClassifier(**config.SEMANTIC_BAGGER_PARAMS)

    @staticmethod
    def get_metadata_anchor():
        """
        Returns the Contextual Branch learner (Logistic Regression).
        Trained on Metadata only.
        """
        return LogisticRegression(**config.METADATA_ANCHOR_PARAMS)

    @staticmethod
    def get_meta_learner():
        """
        Returns the Level 2 Meta-Learner (Logistic Regression).
        Trained on the probability outputs of the Level 1 learners.
        """
        return LogisticRegression(**config.META_LEARNER_PARAMS)
