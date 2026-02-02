import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library.config import (
    RF_PARAMS,
    XGB_PARAMS,
    RF_DENSE_PARAMS,
    LR_PARAMS,
    META_LEARNER_PARAMS,
)


class ModelFactory:
    """
    Factory class to create and configure the 7 Base Learners (Level 1)
    and the Meta-Learner (Level 2) for the Dual-Topology Stacking architecture.
    """

    # Define model keys for consistent reference
    KEY_TEXT_SPARSE_RF = "text_sparse_rf"
    KEY_TEXT_DENSE_XGB = "text_dense_xgb"
    KEY_TEXT_DENSE_RF = "text_dense_rf"

    KEY_BEH_SPARSE_RF = "beh_sparse_rf"
    KEY_BEH_DENSE_XGB = "beh_dense_xgb"
    KEY_BEH_DENSE_RF = "beh_dense_rf"

    KEY_META_LR = "meta_lr"

    @staticmethod
    def get_level_1_models():
        """
        Instantiates the 7 base learners with hyperparameters from config.

        Returns:
            dict: A dictionary mapping model keys to initialized sklearn-compatible model objects.
        """
        models = {}

        # --- Text Branch ---
        # 1. Lexical Bagger (Sparse RF)
        models[ModelFactory.KEY_TEXT_SPARSE_RF] = RandomForestClassifier(**RF_PARAMS)

        # 2. Semantic Hybrid (Dense XGB)
        # Separate fit params from init params
        xgb_init_params = XGB_PARAMS.copy()
        if "early_stopping_rounds" in xgb_init_params:
            xgb_init_params.pop("early_stopping_rounds")
        models[ModelFactory.KEY_TEXT_DENSE_XGB] = XGBClassifier(**xgb_init_params)

        # 3. Semantic Hybrid (Dense RF)
        models[ModelFactory.KEY_TEXT_DENSE_RF] = RandomForestClassifier(
            **RF_DENSE_PARAMS
        )

        # --- Behavioral Branch ---
        # 4. Community Bagger (Sparse RF)
        models[ModelFactory.KEY_BEH_SPARSE_RF] = RandomForestClassifier(**RF_PARAMS)

        # 5. Persona Hybrid (Dense XGB)
        models[ModelFactory.KEY_BEH_DENSE_XGB] = XGBClassifier(**xgb_init_params)

        # 6. Persona Hybrid (Dense RF)
        models[ModelFactory.KEY_BEH_DENSE_RF] = RandomForestClassifier(
            **RF_DENSE_PARAMS
        )

        # --- Context Branch ---
        # 7. Metadata Anchor (LR)
        models[ModelFactory.KEY_META_LR] = LogisticRegression(**LR_PARAMS)

        return models

    @staticmethod
    def get_level_2_model():
        """
        Instantiates the Meta-Learner.

        Returns:
            sklearn.linear_model.LogisticRegression: The initialized meta-learner.
        """
        return LogisticRegression(**META_LEARNER_PARAMS)

    @staticmethod
    def prepare_features(feature_dict, model_key):
        """
        Assembles the correct feature matrix X for a given model key.
        Handles concatenation of specific views with metadata.

        Args:
            feature_dict (dict): Dictionary containing 'meta', 'text_sparse', 'text_dense',
                                 'beh_sparse', 'beh_dense' features.
            model_key (str): The key identifying the model (e.g., 'text_sparse_rf').

        Returns:
            np.ndarray or scipy.sparse.csr_matrix: The combined feature matrix.
        """
        meta = feature_dict["meta"]

        # --- Text Branch ---
        if model_key == ModelFactory.KEY_TEXT_SPARSE_RF:
            # Sparse Text + Dense Meta -> Sparse
            return sparse.hstack([feature_dict["text_sparse"], meta]).tocsr()

        elif model_key == ModelFactory.KEY_TEXT_DENSE_XGB:
            # Dense Text + Dense Meta -> Dense
            return np.hstack([feature_dict["text_dense"], meta])

        elif model_key == ModelFactory.KEY_TEXT_DENSE_RF:
            # Dense Text + Dense Meta -> Dense
            return np.hstack([feature_dict["text_dense"], meta])

        # --- Behavioral Branch ---
        elif model_key == ModelFactory.KEY_BEH_SPARSE_RF:
            # Sparse Subreddits + Dense Meta -> Sparse
            return sparse.hstack([feature_dict["beh_sparse"], meta]).tocsr()

        elif model_key == ModelFactory.KEY_BEH_DENSE_XGB:
            # Dense Subreddits + Dense Meta -> Dense
            return np.hstack([feature_dict["beh_dense"], meta])

        elif model_key == ModelFactory.KEY_BEH_DENSE_RF:
            # Dense Subreddits + Dense Meta -> Dense
            return np.hstack([feature_dict["beh_dense"], meta])

        # --- Context Branch ---
        elif model_key == ModelFactory.KEY_META_LR:
            # Meta only
            return meta

        else:
            raise ValueError(f"Unknown model key: {model_key}")

    @staticmethod
    def get_xgb_fit_params():
        """
        Returns the parameters intended for the fit() method of XGBoost models.
        Useful for retrieving 'early_stopping_rounds'.
        """
        params = {}
        if "early_stopping_rounds" in XGB_PARAMS:
            params["early_stopping_rounds"] = XGB_PARAMS["early_stopping_rounds"]
        return params
