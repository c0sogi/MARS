import copy
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from library.config import Config


class ModelRegistry:
    """
    Factory class for instantiating machine learning models with configurations
    defined in library.config.Config.
    """

    # Define model categories for training strategy
    VOLATILE_MODELS = {
        "semantic_booster",
        "semantic_gradient",
        "temporal_booster",
    }

    STABLE_MODELS = {
        "lexical_bagger",
        "lexical_randomizer",
        "lexical_anchor",
        "community_bagger",
        "community_anchor",
        "semantic_bagger",
        "metadata_anchor",
        "meta_learner",
    }

    @staticmethod
    def get_model_type(model_name):
        """
        Returns 'volatile' or 'stable' based on the model name.
        Volatile models support early stopping and require validation sets during fit.
        Stable models are trained on the full available data for the fold.
        """
        if model_name in ModelRegistry.VOLATILE_MODELS:
            return "volatile"
        elif model_name in ModelRegistry.STABLE_MODELS:
            return "stable"
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    # --- Branch 1: Sparse Lexical (Text) ---

    @staticmethod
    def get_lexical_bagger():
        params = copy.deepcopy(Config.LEXICAL_BAGGER_PARAMS)
        return RandomForestClassifier(**params)

    @staticmethod
    def get_lexical_randomizer():
        params = copy.deepcopy(Config.LEXICAL_RANDOMIZER_PARAMS)
        return ExtraTreesClassifier(**params)

    @staticmethod
    def get_lexical_anchor():
        params = copy.deepcopy(Config.LEXICAL_ANCHOR_PARAMS)
        return LogisticRegression(**params)

    # --- Branch 2: Sparse Behavioral (History) ---

    @staticmethod
    def get_community_bagger():
        params = copy.deepcopy(Config.COMMUNITY_BAGGER_PARAMS)
        return RandomForestClassifier(**params)

    @staticmethod
    def get_community_anchor():
        params = copy.deepcopy(Config.COMMUNITY_ANCHOR_PARAMS)
        return LogisticRegression(**params)

    # --- Branch 3: Dense Semantic (Text Embeddings) ---

    @staticmethod
    def get_semantic_booster():
        params = copy.deepcopy(Config.SEMANTIC_BOOSTER_PARAMS)
        # XGBClassifier from xgboost package
        return XGBClassifier(**params)

    @staticmethod
    def get_semantic_gradient():
        params = copy.deepcopy(Config.SEMANTIC_GRADIENT_PARAMS)
        # LGBMClassifier from lightgbm package
        return LGBMClassifier(**params)

    @staticmethod
    def get_semantic_bagger():
        params = copy.deepcopy(Config.SEMANTIC_BAGGER_PARAMS)
        return RandomForestClassifier(**params)

    # --- Branch 4: Contextual (Metadata) ---

    @staticmethod
    def get_metadata_anchor():
        params = copy.deepcopy(Config.METADATA_ANCHOR_PARAMS)
        return LogisticRegression(**params)

    @staticmethod
    def get_temporal_booster():
        params = copy.deepcopy(Config.TEMPORAL_BOOSTER_PARAMS)
        return LGBMClassifier(**params)

    # --- Level 2: Meta-Learner ---

    @staticmethod
    def get_meta_learner():
        params = copy.deepcopy(Config.META_LEARNER_PARAMS)
        return LogisticRegression(**params)

    @staticmethod
    def create_model(model_name):
        """
        Generic dispatcher to create a model by name.
        """
        dispatch = {
            "lexical_bagger": ModelRegistry.get_lexical_bagger,
            "lexical_randomizer": ModelRegistry.get_lexical_randomizer,
            "lexical_anchor": ModelRegistry.get_lexical_anchor,
            "community_bagger": ModelRegistry.get_community_bagger,
            "community_anchor": ModelRegistry.get_community_anchor,
            "semantic_booster": ModelRegistry.get_semantic_booster,
            "semantic_gradient": ModelRegistry.get_semantic_gradient,
            "semantic_bagger": ModelRegistry.get_semantic_bagger,
            "metadata_anchor": ModelRegistry.get_metadata_anchor,
            "temporal_booster": ModelRegistry.get_temporal_booster,
            "meta_learner": ModelRegistry.get_meta_learner,
        }

        if model_name not in dispatch:
            raise ValueError(f"Model '{model_name}' not found in registry.")

        return dispatch[model_name]()
