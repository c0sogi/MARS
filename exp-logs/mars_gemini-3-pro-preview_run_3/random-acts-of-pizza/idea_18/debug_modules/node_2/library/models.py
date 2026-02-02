import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library.config import Config


class ModelFactory:
    """
    Factory class to instantiate and configure base learners and the meta-learner
    for the Symmetric Multi-Modal Stacking Ensemble.
    """

    @staticmethod
    def get_text_lexical_model():
        """
        Returns the Lexical Bagger (Random Forest) for the Text Modality (Sparse View).
        Uses TF-IDF + Metadata.
        """
        return RandomForestClassifier(**Config.RF_LEXICAL_PARAMS)

    @staticmethod
    def get_text_semantic_models(scale_pos_weight=1.0):
        """
        Returns the Semantic Ensemble models for the Text Modality (Dense View).
        Includes:
            1. XGBoost Classifier (Gradient Boosting)
            2. Random Forest Classifier (Bagging)
        Both use Embeddings + Metadata.

        Args:
            scale_pos_weight (float): Weight for positive class in XGBoost to handle imbalance.
                                      Should be calculated as count(neg) / count(pos).

        Returns:
            dict: {'xgb': XGBClassifier, 'rf': RandomForestClassifier}
        """
        # Configure XGBoost
        xgb_params = Config.XGB_SEMANTIC_PARAMS.copy()
        xgb_params["scale_pos_weight"] = scale_pos_weight
        xgb_model = XGBClassifier(**xgb_params)

        # Configure Random Forest
        rf_model = RandomForestClassifier(**Config.RF_SEMANTIC_PARAMS)

        return {"xgb": xgb_model, "rf": rf_model}

    @staticmethod
    def get_behavior_sparse_model():
        """
        Returns the Community Bagger (Random Forest) for the Behavioral Modality (Sparse View).
        Uses Subreddit TF-IDF + Metadata.
        """
        return RandomForestClassifier(**Config.RF_BEHAVIORAL_PARAMS)

    @staticmethod
    def get_behavior_dense_model(scale_pos_weight=1.0):
        """
        Returns the Persona Booster (XGBoost) for the Behavioral Modality (Dense View).
        Uses Subreddit Embeddings + Metadata.

        Args:
            scale_pos_weight (float): Weight for positive class in XGBoost.
        """
        xgb_params = Config.XGB_BEHAVIORAL_PARAMS.copy()
        xgb_params["scale_pos_weight"] = scale_pos_weight
        return XGBClassifier(**xgb_params)

    @staticmethod
    def get_context_model():
        """
        Returns the Metadata Anchor (Logistic Regression) for the Contextual Modality.
        Uses Metadata only.
        """
        return LogisticRegression(**Config.LR_CONTEXTUAL_PARAMS)

    @staticmethod
    def get_meta_learner():
        """
        Returns the Level 2 Meta-Learner (Logistic Regression).
        Calibrates the contributions of the Level 1 base learners.
        """
        return LogisticRegression(**Config.META_LEARNER_PARAMS)
