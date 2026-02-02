import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from library.config import RF_PARAMS, XGB_PARAMS, KNN_PARAMS, LR_PARAMS, SEED
from library.utils import set_seed


class ModelRegistry:
    """
    Factory class to instantiate the specific base learners and meta-learner
    for the Hex-View Hybrid-Topology Stacking Ensemble.

    This registry ensures that all models are initialized with the hyperparameters
    defined in library.config and consistent random seeds for reproducibility.
    """

    @staticmethod
    def get_lexical_bagger():
        """
        Level 1 Base Learner: Sparse Lexical Branch.
        Algorithm: Random Forest Classifier.
        Input: TF-IDF Text Vectors + Global Metadata.
        Rationale: Captures high-impact keywords using sparse representations.
        """
        set_seed(SEED)
        return RandomForestClassifier(**RF_PARAMS)

    @staticmethod
    def get_community_bagger():
        """
        Level 1 Base Learner: Sparse Behavioral Branch.
        Algorithm: Random Forest Classifier.
        Input: TF-IDF Subreddit History + Global Metadata.
        Rationale: Treats user history as a bag-of-concepts to capture niche community signals.
        """
        set_seed(SEED)
        return RandomForestClassifier(**RF_PARAMS)

    @staticmethod
    def get_semantic_booster():
        """
        Level 1 Base Learner: Dense Semantic Branch.
        Algorithm: XGBoost Classifier.
        Input: Dense Embeddings + Global Metadata.
        Rationale: Extracts non-linear signals from continuous embedding space.
        Note: Early stopping rounds should be passed dynamically during the fit call.
        """
        set_seed(SEED)
        return XGBClassifier(**XGB_PARAMS)

    @staticmethod
    def get_semantic_bagger():
        """
        Level 1 Base Learner: Dense Semantic Branch.
        Algorithm: Random Forest Classifier.
        Input: Dense Embeddings + Global Metadata.
        Rationale: Provides algorithmic diversity (Bagging vs Boosting) on the semantic view.
        """
        set_seed(SEED)
        return RandomForestClassifier(**RF_PARAMS)

    @staticmethod
    def get_manifold_neighbor():
        """
        Level 1 Base Learner: Manifold Learning Branch.
        Algorithm: k-Nearest Neighbors Classifier.
        Input: PCA-Reduced Embeddings + Global Metadata.
        Rationale: Exploits local data density and retrieves semantically similar requests.
        """
        set_seed(SEED)
        return KNeighborsClassifier(**KNN_PARAMS)

    @staticmethod
    def get_metadata_anchor():
        """
        Level 1 Base Learner: Contextual Branch.
        Algorithm: Logistic Regression.
        Input: Global Metadata Vector only.
        Rationale: Acts as a high-bias regularizer based on robust user/post statistics.
        """
        set_seed(SEED)
        return LogisticRegression(**LR_PARAMS)

    @staticmethod
    def get_meta_learner():
        """
        Level 2 Meta-Learner.
        Algorithm: Logistic Regression.
        Input: Predictions (probabilities) from all Level 1 base learners.
        Rationale: Calibrates the ensemble weights to minimize log loss / maximize AUC.
        """
        set_seed(SEED)
        return LogisticRegression(**LR_PARAMS)

    @staticmethod
    def create_base_models():
        """
        Instantiates and returns a dictionary containing all Level 1 base learners.

        Returns:
            dict: Keys are model identifiers (e.g., 'lexical_bagger'),
                  Values are instantiated model objects.
        """
        return {
            "lexical_bagger": ModelRegistry.get_lexical_bagger(),
            "community_bagger": ModelRegistry.get_community_bagger(),
            "semantic_booster": ModelRegistry.get_semantic_booster(),
            "semantic_bagger": ModelRegistry.get_semantic_bagger(),
            "manifold_neighbor": ModelRegistry.get_manifold_neighbor(),
            "metadata_anchor": ModelRegistry.get_metadata_anchor(),
        }
