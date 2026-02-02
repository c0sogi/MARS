import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import set_seed


class ModelZoo:
    """
    ModelZoo provides initialized instances of the 7 base learners and the meta-learner
    for the High-Fidelity Hept-View Stacking Ensemble.

    It categorizes models into 'volatile' (Gradient Boosting) and 'stable' (Bagging/Linear)
    types to support the Consistent Hybrid Inference Protocol.
    """

    def __init__(self):
        set_seed(Config.SEED)

    @staticmethod
    def get_lexical_bagger():
        """
        Returns the High-Fidelity Lexical Bagger (Random Forest).
        Branch: Sparse Lexical (Text Modality).
        """
        return RandomForestClassifier(**Config.LEXICAL_RF_PARAMS)

    @staticmethod
    def get_community_bagger():
        """
        Returns the Community Bagger (Random Forest).
        Branch: Sparse Behavioral (History Modality).
        """
        return RandomForestClassifier(**Config.COMMUNITY_RF_PARAMS)

    @staticmethod
    def get_semantic_booster():
        """
        Returns the Semantic Booster (XGBoost).
        Branch: Dense Semantic (Text Modality).
        Type: Volatile (Supports Early Stopping).
        """
        return xgb.XGBClassifier(**Config.SEMANTIC_XGB_PARAMS)

    @staticmethod
    def get_semantic_gradient():
        """
        Returns the Semantic Gradient (LightGBM).
        Branch: Dense Semantic (Text Modality).
        Type: Volatile (Supports Early Stopping).
        """
        return lgb.LGBMClassifier(**Config.SEMANTIC_LGBM_PARAMS)

    @staticmethod
    def get_semantic_bagger():
        """
        Returns the Semantic Bagger (Random Forest).
        Branch: Dense Semantic (Text Modality).
        """
        return RandomForestClassifier(**Config.SEMANTIC_RF_PARAMS)

    @staticmethod
    def get_metadata_anchor():
        """
        Returns the Metadata Anchor (Logistic Regression).
        Branch: Contextual (Metadata Modality).
        """
        return LogisticRegression(**Config.METADATA_ANCHOR_PARAMS)

    @staticmethod
    def get_temporal_booster():
        """
        Returns the Temporal Booster (LightGBM).
        Branch: Contextual (Metadata Modality).
        Type: Volatile (Supports Early Stopping).
        """
        return lgb.LGBMClassifier(**Config.METADATA_BOOSTER_PARAMS)

    @staticmethod
    def get_meta_learner():
        """
        Returns the Level 2 Meta-Learner (Logistic Regression).
        """
        return LogisticRegression(**Config.META_LEARNER_PARAMS)

    @classmethod
    def get_models_dict(cls):
        """
        Returns a dictionary of all Level 1 base learners with their metadata.

        Structure:
        {
            "model_name": {
                "model": instance,
                "type": "volatile" | "stable",
                "feature_set": "lexical" | "behavioral" | "semantic" | "metadata"
            }
        }

        - 'volatile': Models that use gradient boosting and require validation sets for early stopping.
        - 'stable': Models that use bagging or linear equations and are trained on the full fold.
        - 'feature_set': The primary modality to be loaded and (optionally) concatenated with metadata.
        """
        return {
            "lexical_bagger": {
                "model": cls.get_lexical_bagger(),
                "type": "stable",
                "feature_set": "lexical",
            },
            "community_bagger": {
                "model": cls.get_community_bagger(),
                "type": "stable",
                "feature_set": "behavioral",
            },
            "semantic_booster": {
                "model": cls.get_semantic_booster(),
                "type": "volatile",
                "feature_set": "semantic",
            },
            "semantic_gradient": {
                "model": cls.get_semantic_gradient(),
                "type": "volatile",
                "feature_set": "semantic",
            },
            "semantic_bagger": {
                "model": cls.get_semantic_bagger(),
                "type": "stable",
                "feature_set": "semantic",
            },
            "metadata_anchor": {
                "model": cls.get_metadata_anchor(),
                "type": "stable",
                "feature_set": "metadata",
            },
            "temporal_booster": {
                "model": cls.get_temporal_booster(),
                "type": "volatile",
                "feature_set": "metadata",
            },
        }
