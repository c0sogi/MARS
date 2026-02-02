import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from library import config


class ModelFactory:
    """
    Factory class to instantiate Level 1 base learners and Level 2 meta-learners
    with configurations defined in library.config.
    """

    @staticmethod
    def _get_params(default_params, overrides):
        """
        Helper to merge default parameters with overrides.
        """
        params = copy.deepcopy(default_params)
        if overrides:
            params.update(overrides)
        return params

    @classmethod
    def get_interaction_bagger(cls, **kwargs):
        """
        Returns the Interaction Bagger (Holistic Sparse Branch).
        Model: Random Forest
        Input: Concatenated Text + Subreddits + Metadata
        """
        params = cls._get_params(config.INTERACTION_BAGGER_PARAMS, kwargs)
        return RandomForestClassifier(**params)

    @classmethod
    def get_lexical_bagger(cls, **kwargs):
        """
        Returns the Lexical Bagger (Specialized Sparse Branch).
        Model: Random Forest
        Input: Title + Body + Metadata
        """
        params = cls._get_params(config.LEXICAL_BAGGER_PARAMS, kwargs)
        return RandomForestClassifier(**params)

    @classmethod
    def get_community_bagger(cls, **kwargs):
        """
        Returns the Community Bagger (Specialized Sparse Branch).
        Model: Random Forest
        Input: Subreddit History + Metadata
        """
        params = cls._get_params(config.COMMUNITY_BAGGER_PARAMS, kwargs)
        return RandomForestClassifier(**params)

    @classmethod
    def get_semantic_booster(cls, **kwargs):
        """
        Returns the Semantic Booster (Dense Semantic Branch).
        Model: XGBoost
        Input: Embeddings + Metadata
        """
        params = cls._get_params(config.SEMANTIC_BOOSTER_PARAMS, kwargs)
        return XGBClassifier(**params)

    @classmethod
    def get_semantic_bagger(cls, **kwargs):
        """
        Returns the Semantic Bagger (Dense Semantic Branch).
        Model: Random Forest
        Input: Embeddings + Metadata
        """
        params = cls._get_params(config.SEMANTIC_BAGGER_PARAMS, kwargs)
        return RandomForestClassifier(**params)

    @classmethod
    def get_metadata_anchor(cls, **kwargs):
        """
        Returns the Metadata Anchor (Contextual Branch).
        Model: Logistic Regression
        Input: Metadata Only
        """
        params = cls._get_params(config.METADATA_ANCHOR_PARAMS, kwargs)
        return LogisticRegression(**params)

    @classmethod
    def get_meta_learner(cls, **kwargs):
        """
        Returns the Level 2 Meta-Learner.
        Model: Logistic Regression
        Input: Predictions from Level 1 models
        """
        params = cls._get_params(config.META_LEARNER_PARAMS, kwargs)
        return LogisticRegression(**params)
