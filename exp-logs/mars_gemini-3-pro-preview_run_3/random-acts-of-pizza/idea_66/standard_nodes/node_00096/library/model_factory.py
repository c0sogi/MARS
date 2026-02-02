import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from library.config import Config


class ModelFactory:
    """
    Factory class for instantiating the machine learning models used in the
    High-Fidelity Symmetric Stacking Ensemble.

    This class ensures that all models are initialized with the hyperparameters
    defined in the central Config, enforcing the architectural decisions regarding
    regularization, boosting conservatism, and differential sparsity.
    """

    @staticmethod
    def get_base_models():
        """
        Instantiates the 9 base learners (Level 1 models) across the 4 modality branches.

        Returns:
            dict: A dictionary mapping model names to their instantiated sklearn/xgb/lgbm objects.
        """
        models = {
            # 1. Open-Vocabulary Lexical Branch (Text Modality)
            # Uses Granular Tokenization + Open Vocabulary TF-IDF
            "lexical_bagger": RandomForestClassifier(**Config.LEXICAL_BAGGER_PARAMS),
            "lexical_anchor": LogisticRegression(**Config.LEXICAL_ANCHOR_PARAMS),
            # 2. Closed-Vocabulary Behavioral Branch (History Modality)
            # Uses Constrained Vocabulary TF-IDF
            "community_bagger": RandomForestClassifier(
                **Config.COMMUNITY_BAGGER_PARAMS
            ),
            "community_anchor": LogisticRegression(**Config.COMMUNITY_ANCHOR_PARAMS),
            # 3. Dense Semantic Branch (Text Modality)
            # Uses Frozen Dense Embeddings
            "semantic_booster": xgb.XGBClassifier(**Config.SEMANTIC_BOOSTER_PARAMS),
            "semantic_gradient": lgb.LGBMClassifier(**Config.SEMANTIC_GRADIENT_PARAMS),
            "semantic_bagger": RandomForestClassifier(**Config.SEMANTIC_BAGGER_PARAMS),
            # 4. Contextual Branch (Metadata Modality)
            # Uses Allow-Listed Metadata
            "metadata_anchor": LogisticRegression(**Config.METADATA_ANCHOR_PARAMS),
            "temporal_booster": lgb.LGBMClassifier(**Config.TEMPORAL_BOOSTER_PARAMS),
        }

        return models

    @staticmethod
    def get_meta_learner():
        """
        Instantiates the Level 2 Meta-Learner.

        Returns:
            sklearn.linear_model.LogisticRegression: The meta-learner model.
        """
        return LogisticRegression(**Config.META_LEARNER_PARAMS)
