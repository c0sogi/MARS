import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from library import config


class ModelFactory:
    """
    Factory class for creating models and preparing features for the Hept-View Ensemble.
    Encapsulates the architecture definition, hyperparameter loading, and feature stacking logic.
    """

    # Definition of the Hept-View Ensemble Architecture
    # Maps model names to their algorithm type, required feature views, and volatility status.
    MODEL_REGISTRY = {
        # 1. Sparse Lexical Branch
        "lexical_bagger": {
            "type": "rf",
            "features": ["lexical", "contextual"],
            "volatility": "stable",
        },
        # 2. Sparse Behavioral Branch
        "community_bagger": {
            "type": "rf",
            "features": ["behavioral", "contextual"],
            "volatility": "stable",
        },
        # 3. Dense Semantic Branch
        "semantic_booster": {
            "type": "xgb",
            "features": ["semantic", "contextual"],
            "volatility": "volatile",
        },
        "semantic_gradient": {
            "type": "lgbm",
            "features": ["semantic", "contextual"],
            "volatility": "volatile",
        },
        "semantic_bagger": {
            "type": "rf",
            "features": ["semantic", "contextual"],
            "volatility": "stable",
        },
        # 4. Contextual Branch
        "metadata_anchor": {
            "type": "lr",
            "features": ["contextual"],
            "volatility": "stable",
        },
        "temporal_booster": {
            "type": "lgbm",
            "features": ["contextual"],
            "volatility": "volatile",
        },
    }

    @staticmethod
    def get_model_config(model_name: str) -> dict:
        """Retrieves the architectural configuration for a specific model."""
        if model_name not in ModelFactory.MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found in registry.")
        return ModelFactory.MODEL_REGISTRY[model_name]

    @staticmethod
    def get_model(model_name: str):
        """
        Instantiates a Level 1 base learner with parameters from config.
        Removes 'early_stopping_rounds' from init params to avoid TypeError.
        """
        if model_name not in config.MODEL_PARAMS:
            raise ValueError(f"Parameters for '{model_name}' not found in config.")

        # Copy params to avoid mutating global config
        params = config.MODEL_PARAMS[model_name].copy()

        # Remove fit-time parameters if present
        if "early_stopping_rounds" in params:
            _ = params.pop("early_stopping_rounds")

        # Get model type
        model_conf = ModelFactory.get_model_config(model_name)
        model_type = model_conf["type"]

        if model_type == "rf":
            return RandomForestClassifier(**params)
        elif model_type == "xgb":
            return XGBClassifier(**params)
        elif model_type == "lgbm":
            return LGBMClassifier(**params)
        elif model_type == "lr":
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_meta_learner():
        """Instantiates the Level 2 Meta-Learner (Logistic Regression)."""
        params = config.MODEL_PARAMS["meta_learner"].copy()
        return LogisticRegression(**params)

    @staticmethod
    def prepare_features(data: dict, model_name: str, split: str = "train"):
        """
        Stacks the required feature views for a given model.
        Handles combination of Sparse and Dense matrices.

        Args:
            data (dict): Dictionary containing all feature matrices (from FeaturePipeline).
            model_name (str): Name of the model to prepare features for.
            split (str): 'train' or 'test'.

        Returns:
            Combined feature matrix (numpy array or scipy sparse matrix).
        """
        conf = ModelFactory.get_model_config(model_name)
        required_views = conf["features"]

        matrices_to_stack = []
        is_sparse = False

        for view in required_views:
            # Construct key, e.g., "X_train_lexical"
            key = f"X_{split}_{view}"

            if key not in data:
                raise KeyError(f"Feature key '{key}' not found in data dictionary.")

            matrix = data[key]
            matrices_to_stack.append(matrix)

            if sp.issparse(matrix):
                is_sparse = True

        if not matrices_to_stack:
            raise ValueError(f"No features defined for model {model_name}")

        # Stack matrices
        if is_sparse:
            # If any matrix is sparse, we must convert all dense matrices to sparse
            # and use scipy.sparse.hstack
            sparse_matrices = []
            for m in matrices_to_stack:
                if sp.issparse(m):
                    sparse_matrices.append(m)
                else:
                    sparse_matrices.append(sp.csr_matrix(m))
            return sp.hstack(sparse_matrices, format="csr")
        else:
            # If all are dense, use numpy hstack
            return np.hstack(matrices_to_stack)
