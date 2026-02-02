from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from library.config import (
    RF_PARAMS,
    XGB_PARAMS,
    LOGREG_ANCHOR_PARAMS,
    META_LEARNER_PARAMS,
)


class ModelFactory:
    """
    Factory class to instantiate the base learners and meta-learner
    for the Symmetric Dual-Topology Stacking Ensemble architecture.

    This class encapsulates the configuration of the six Level 1 base learners
    and the Level 2 meta-learner, ensuring consistent hyperparameter application
    and easing the management of the ensemble structure.
    """

    @staticmethod
    def get_lexical_sparse_rf(**kwargs):
        """
        Returns the Random Forest classifier for the Lexical Sparse View (TF-IDF).

        Base Params: RF_PARAMS (Balanced class weights, high regularization)
        """
        params = RF_PARAMS.copy()
        params.update(kwargs)
        return RandomForestClassifier(**params)

    @staticmethod
    def get_lexical_dense_rf(**kwargs):
        """
        Returns the Random Forest classifier for the Lexical Dense View (MPNet Embeddings).

        Base Params: RF_PARAMS
        """
        params = RF_PARAMS.copy()
        params.update(kwargs)
        return RandomForestClassifier(**params)

    @staticmethod
    def get_lexical_dense_xgb(**kwargs):
        """
        Returns the XGBoost classifier for the Lexical Dense View (MPNet Embeddings).

        Base Params: XGB_PARAMS (Scale pos weight, specific learning rate)
        Note: Early stopping should be handled in the training loop using the validation set.
        """
        params = XGB_PARAMS.copy()
        params.update(kwargs)
        return XGBClassifier(**params)

    @staticmethod
    def get_behavioral_sparse_rf(**kwargs):
        """
        Returns the Random Forest classifier for the Behavioral Sparse View (Subreddit TF-IDF).

        Base Params: RF_PARAMS
        """
        params = RF_PARAMS.copy()
        params.update(kwargs)
        return RandomForestClassifier(**params)

    @staticmethod
    def get_behavioral_dense_xgb(**kwargs):
        """
        Returns the XGBoost classifier for the Behavioral Dense View (Subreddit Embeddings).

        Base Params: XGB_PARAMS
        Note: Early stopping should be handled in the training loop using the validation set.
        """
        params = XGB_PARAMS.copy()
        params.update(kwargs)
        return XGBClassifier(**params)

    @staticmethod
    def get_contextual_anchor_lr(**kwargs):
        """
        Returns the Logistic Regression anchor model for the Contextual View (Metadata).

        Base Params: LOGREG_ANCHOR_PARAMS (L2 penalty, balanced weights)
        """
        params = LOGREG_ANCHOR_PARAMS.copy()
        params.update(kwargs)
        return LogisticRegression(**params)

    @staticmethod
    def get_meta_learner(**kwargs):
        """
        Returns the Logistic Regression Meta-Learner for Level 2 Stacking.

        Base Params: META_LEARNER_PARAMS
        """
        params = META_LEARNER_PARAMS.copy()
        params.update(kwargs)
        return LogisticRegression(**params)
