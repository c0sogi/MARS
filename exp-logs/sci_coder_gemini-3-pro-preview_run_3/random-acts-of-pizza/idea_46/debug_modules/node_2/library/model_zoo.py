import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from library.config import Config
from library.utils import set_seed


class ModelZoo:
    """
    Factory class for initializing the models used in the Hex-View Stacking Ensemble.
    Provides methods to instantiate Level 1 base learners and the Level 2 meta-learner
    with hyperparameters defined in the configuration.
    """

    @staticmethod
    def get_base_models():
        """
        Initializes and returns the dictionary of Level 1 base learners.

        The ensemble consists of:
        1. Lexical_RF: Random Forest on Text TF-IDF + Metadata
        2. Community_RF: Random Forest on Subreddit TF-IDF + Metadata
        3. Semantic_XGB: XGBoost on Embeddings + Metadata
        4. Semantic_RF: Random Forest on Embeddings + Metadata
        5. Metadata_LR: Logistic Regression on Metadata (Linear Anchor)
        6. Temporal_LGBM: LightGBM on Metadata (Non-linear Temporal Booster)

        Returns:
            dict: A dictionary where keys are model names and values are initialized model instances.
        """
        # Ensure reproducibility
        set_seed(Config.RANDOM_SEED)

        models = {}

        # 1. Lexical Bagger
        # Random Forest trained on Sparse Lexical features + Dense Metadata
        lexical_params = Config.L1_LEXICAL_RF_PARAMS.copy()
        models["Lexical_RF"] = RandomForestClassifier(**lexical_params)

        # 2. Community Bagger
        # Random Forest trained on Sparse Behavioral features + Dense Metadata
        community_params = Config.L1_COMMUNITY_RF_PARAMS.copy()
        models["Community_RF"] = RandomForestClassifier(**community_params)

        # 3. Semantic Booster
        # XGBoost trained on Dense Semantic Embeddings + Dense Metadata
        # Note: early_stopping_rounds is handled by the XGBClassifier constructor in recent versions
        semantic_xgb_params = Config.L1_SEMANTIC_XGB_PARAMS.copy()
        models["Semantic_XGB"] = xgb.XGBClassifier(**semantic_xgb_params)

        # 4. Semantic Bagger
        # Random Forest trained on Dense Semantic Embeddings + Dense Metadata
        # Provides structural diversity to the gradient booster
        semantic_rf_params = Config.L1_SEMANTIC_RF_PARAMS.copy()
        models["Semantic_RF"] = RandomForestClassifier(**semantic_rf_params)

        # 5. Metadata Anchor
        # Logistic Regression trained on Dense Metadata
        # Acts as a high-bias linear baseline
        meta_lr_params = Config.L1_META_LR_PARAMS.copy()
        models["Metadata_LR"] = LogisticRegression(**meta_lr_params)

        # 6. Temporal Booster
        # LightGBM trained on Dense Metadata
        # Captures non-linear temporal regimes via raw timestamp splits
        temporal_lgbm_params = Config.L1_META_LGBM_PARAMS.copy()
        models["Temporal_LGBM"] = lgb.LGBMClassifier(**temporal_lgbm_params)

        return models

    @staticmethod
    def get_meta_learner():
        """
        Initializes and returns the Level 2 Meta-Learner.

        The meta-learner is a Logistic Regression model that calibrates the
        predictions from the Level 1 base learners.

        Returns:
            sklearn.linear_model.LogisticRegression: The initialized meta-learner.
        """
        # Ensure reproducibility
        set_seed(Config.RANDOM_SEED)

        l2_params = Config.L2_LOGREG_PARAMS.copy()
        return LogisticRegression(**l2_params)
