import library.config as config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ModelFactory:
    """
    Factory class to instantiate configured model objects for the
    Clean-Signal Hex-View Stacking Ensemble.

    Retrieves hyperparameters from library.config.
    """

    @staticmethod
    def get_lexical_bagger():
        """
        Returns the Random Forest model for the Sparse Lexical Branch.
        Trained on TF-IDF vectors + Metadata.
        """
        return RandomForestClassifier(**config.RF_PARAMS_LEXICAL)

    @staticmethod
    def get_community_bagger():
        """
        Returns the Random Forest model for the Sparse Behavioral Branch.
        Trained on Subreddit History (Bag-of-Concepts) + Metadata.
        """
        return RandomForestClassifier(**config.RF_PARAMS_COMMUNITY)

    @staticmethod
    def get_semantic_booster():
        """
        Returns the XGBoost model for the Dense Semantic Branch.
        Trained on Embeddings + Metadata.
        """
        return XGBClassifier(**config.XGB_PARAMS_SEMANTIC)

    @staticmethod
    def get_semantic_bagger():
        """
        Returns the Random Forest model for the Dense Semantic Branch.
        Trained on Embeddings + Metadata. Provides structural diversity.
        """
        return RandomForestClassifier(**config.RF_PARAMS_SEMANTIC)

    @staticmethod
    def get_metadata_anchor():
        """
        Returns the Logistic Regression model for the Contextual Branch.
        Trained on Metadata only. Acts as a high-bias regularizer.
        """
        return LogisticRegression(**config.LR_PARAMS_ANCHOR)

    @staticmethod
    def get_temporal_booster():
        """
        Returns the LightGBM model for the Contextual Branch.
        Trained on Metadata only. Captures non-linear temporal drift.
        """
        return LGBMClassifier(**config.LGBM_PARAMS_TEMPORAL)

    @staticmethod
    def get_meta_learner():
        """
        Returns the Logistic Regression Meta-Learner (Level 2).
        Calibrates the ensemble weights.
        """
        return LogisticRegression(**config.META_LEARNER_PARAMS)
