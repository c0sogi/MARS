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

        # 1. Community Bagger (The Winner)
        # Random Forest trained on Sparse Behavioral features + Dense Metadata
        # consistently showed the highest OOF AUC (~0.68).
        community_params = Config.L1_COMMUNITY_RF_PARAMS.copy()
        models["Community_RF"] = RandomForestClassifier(**community_params)

        # 2. Metadata Anchor
        # Logistic Regression trained on Dense Metadata
        # Acts as a high-bias linear baseline.
        meta_lr_params = Config.L1_META_LR_PARAMS.copy()
        models["Metadata_LR"] = LogisticRegression(**meta_lr_params)

        # Note: We disabled Lexical_RF, Semantic_XGB, Semantic_RF, and Temporal_LGBM
        # as they were performing near random chance (AUC ~0.50) and degrading the ensemble.

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
