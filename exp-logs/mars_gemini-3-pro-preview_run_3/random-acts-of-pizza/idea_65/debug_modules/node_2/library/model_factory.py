import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from library.config import (
    LEXICAL_BAGGER_PARAMS,
    LEXICAL_ANCHOR_PARAMS,
    COMMUNITY_BAGGER_PARAMS,
    COMMUNITY_ANCHOR_PARAMS,
    SEMANTIC_BOOSTER_PARAMS,
    SEMANTIC_GRADIENT_PARAMS,
    SEMANTIC_BAGGER_PARAMS,
    METADATA_ANCHOR_PARAMS,
    TEMPORAL_BOOSTER_PARAMS,
    META_LEARNER_PARAMS,
)
from library.utils import setup_logger

logger = setup_logger("model_factory")


class ModelFactory:
    """
    Centralized registry for model instantiation.
    Enforces specific architectural configurations defined in library.config.
    """

    @staticmethod
    def get_model(model_key: str, **kwargs):
        """
        Instantiates and returns a configured model object based on the provided key.

        Args:
            model_key (str): The identifier for the model (e.g., 'lexical_bagger').
            **kwargs: Dynamic parameter overrides. These take precedence over config defaults.

        Returns:
            sklearn.base.BaseEstimator: A scikit-learn compatible model instance.

        Raises:
            ValueError: If the model_key is not recognized.
        """
        if model_key == "lexical_bagger":
            # Sparse Lexical Branch: Random Forest
            params = LEXICAL_BAGGER_PARAMS.copy()
            params.update(kwargs)
            return RandomForestClassifier(**params)

        elif model_key == "lexical_anchor":
            # Sparse Lexical Branch: Logistic Regression
            params = LEXICAL_ANCHOR_PARAMS.copy()
            params.update(kwargs)
            return LogisticRegression(**params)

        elif model_key == "community_bagger":
            # Sparse Behavioral Branch: Random Forest
            params = COMMUNITY_BAGGER_PARAMS.copy()
            params.update(kwargs)
            return RandomForestClassifier(**params)

        elif model_key == "community_anchor":
            # Sparse Behavioral Branch: Logistic Regression
            params = COMMUNITY_ANCHOR_PARAMS.copy()
            params.update(kwargs)
            return LogisticRegression(**params)

        elif model_key == "semantic_booster":
            # Dense Semantic Branch: XGBoost
            # Enforces Conservative Boosting
            params = SEMANTIC_BOOSTER_PARAMS.copy()
            params.update(kwargs)
            return xgb.XGBClassifier(**params)

        elif model_key == "semantic_gradient":
            # Dense Semantic Branch: LightGBM
            # Provides algorithmic diversity
            params = SEMANTIC_GRADIENT_PARAMS.copy()
            params.update(kwargs)
            return lgb.LGBMClassifier(**params)

        elif model_key == "semantic_bagger":
            # Dense Semantic Branch: Random Forest
            # Structural diversity with depth constraints
            params = SEMANTIC_BAGGER_PARAMS.copy()
            params.update(kwargs)
            return RandomForestClassifier(**params)

        elif model_key == "metadata_anchor":
            # Contextual Branch: Logistic Regression (L1)
            params = METADATA_ANCHOR_PARAMS.copy()
            params.update(kwargs)
            return LogisticRegression(**params)

        elif model_key == "temporal_booster":
            # Contextual Branch: LightGBM
            # Captures non-linear temporal drift
            params = TEMPORAL_BOOSTER_PARAMS.copy()
            params.update(kwargs)
            return lgb.LGBMClassifier(**params)

        elif model_key == "meta_learner":
            # Level 2: Logistic Regression
            params = META_LEARNER_PARAMS.copy()
            params.update(kwargs)
            return LogisticRegression(**params)

        else:
            raise ValueError(f"Unknown model key: {model_key}")
