import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from library.config import Config


class ModelFactory:
    """
    Factory class to instantiate base learners and the meta-learner
    for the Hex-View Hybrid-Topology Stacking Ensemble.
    """

    @staticmethod
    def get_base_models():
        """
        Initializes and returns the dictionary of Level 1 base learners.

        Returns:
            dict: A dictionary where keys are model names and values are
                  unfitted estimator instances.
        """
        models = {}

        # 1. Sparse Lexical Branch (Text Modality)
        # Random Forest on TF-IDF of Request Text
        # Captures specific high-impact keywords.
        models["LexicalBagger"] = RandomForestClassifier(**Config.RF_LEXICAL_PARAMS)

        # 2. Sparse Behavioral Branch (History Modality)
        # Random Forest on TF-IDF of Subreddit History
        # Captures niche community signals using a bag-of-concepts approach.
        models["BehavioralBagger"] = RandomForestClassifier(
            **Config.RF_BEHAVIORAL_PARAMS
        )

        # 3. Dense Semantic Branch (Text Modality - The Diversity Trio)

        # A. Semantic Booster: XGBoost on Embeddings
        # Extracts non-linear signals from dense embeddings.
        # Note: early_stopping_rounds is passed to __init__ in modern XGBoost/Scikit-Learn APIs
        # or handled during fit via the params stored here.
        models["SemanticBooster"] = XGBClassifier(**Config.XGB_SEMANTIC_PARAMS)

        # B. Semantic Bagger: Random Forest on Embeddings
        # Provides diversity by using Bagging on the same dense view.
        models["SemanticBagger"] = RandomForestClassifier(**Config.RF_SEMANTIC_PARAMS)

        # C. Manifold Neighbor: kNN on PCA-Reduced Embeddings
        # Exploits local manifold structure (retrieval-like signal).
        models["ManifoldNeighbor"] = KNeighborsClassifier(**Config.KNN_MANIFOLD_PARAMS)

        # 4. Contextual Branch (Metadata Modality)
        # Metadata Anchor: Logistic Regression on Global Metadata
        # Acts as a high-bias regularizer.
        models["ContextualAnchor"] = LogisticRegression(
            **Config.LOGREG_CONTEXTUAL_PARAMS
        )

        return models

    @staticmethod
    def get_meta_model():
        """
        Initializes and returns the Level 2 Meta-Learner.

        Returns:
            sklearn.base.BaseEstimator: The unfitted meta-learner instance.
        """
        # Logistic Regression to calibrate ensemble weights
        return LogisticRegression(**Config.META_LEARNER_PARAMS)
