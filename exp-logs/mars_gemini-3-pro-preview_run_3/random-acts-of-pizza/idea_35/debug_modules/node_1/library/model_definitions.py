import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from library.config import Config


class ModelZoo:
    """
    Manages the instantiation, data formatting, and training of the models
    defined in the Unified Interaction-Aware Stacking Ensemble.
    """

    def __init__(self):
        self.config = Config
        # Map model names to (Class, Config_Dict_Name)
        self.model_registry = {
            "unified_rf": (RandomForestClassifier, "MODEL_UNIFIED_RF"),
            "lexical_rf": (RandomForestClassifier, "MODEL_LEXICAL_RF"),
            "community_rf": (RandomForestClassifier, "MODEL_COMMUNITY_RF"),
            "semantic_xgb": (XGBClassifier, "MODEL_SEMANTIC_XGB"),
            "semantic_rf": (RandomForestClassifier, "MODEL_SEMANTIC_RF"),
            "metadata_lr": (LogisticRegression, "MODEL_METADATA_LR"),
            "meta_lr": (LogisticRegression, "MODEL_META_LR"),
        }

    def get_model(self, model_name, **kwargs):
        """
        Instantiates a model by name with parameters from Config, optionally overridden by kwargs.

        Args:
            model_name (str): Name of the model (e.g., 'unified_rf').
            **kwargs: Hyperparameters to override defaults (e.g., scale_pos_weight).

        Returns:
            object: The instantiated model.
        """
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry.")

        model_class, config_attr = self.model_registry[model_name]

        # Get default params from Config
        params = getattr(self.config, config_attr).copy()

        # Override with kwargs
        params.update(kwargs)

        # Special handling for XGBoost
        if model_class == XGBClassifier:
            # Extract fit parameters that shouldn't be in __init__
            early_stopping_rounds = params.pop("early_stopping_rounds", None)

            model = model_class(**params)

            # Attach early_stopping_rounds to the instance for use in train_model
            if early_stopping_rounds is not None:
                model._early_stopping_rounds_custom = early_stopping_rounds
        else:
            model = model_class(**params)

        return model

    def format_data(self, model_name, feature_dict, split="train"):
        """
        Constructs the feature matrix X and target y for a specific model and split.
        Handles the concatenation of Sparse/Dense features with Metadata.

        Args:
            model_name (str): Name of the model.
            feature_dict (dict): Dictionary containing all feature matrices.
            split (str): 'train', 'val', or 'test'.

        Returns:
            tuple: (X, y) where y is None for 'test' split or if not found.
        """
        # Retrieve metadata (used by all Level 1 models)
        key_meta = f"X_{split}_metadata"
        X_meta = feature_dict[key_meta]

        X = None

        # Logic for each model branch
        if model_name == "unified_rf":
            # Sparse Unified + Dense Metadata -> Sparse
            key_feat = f"X_{split}_unified"
            X_feat = feature_dict[key_feat]
            X = sparse.hstack([X_feat, X_meta], format="csr")

        elif model_name == "lexical_rf":
            # Sparse Lexical + Dense Metadata -> Sparse
            key_feat = f"X_{split}_lexical"
            X_feat = feature_dict[key_feat]
            X = sparse.hstack([X_feat, X_meta], format="csr")

        elif model_name == "community_rf":
            # Sparse Community + Dense Metadata -> Sparse
            key_feat = f"X_{split}_community"
            X_feat = feature_dict[key_feat]
            X = sparse.hstack([X_feat, X_meta], format="csr")

        elif model_name in ["semantic_xgb", "semantic_rf"]:
            # Dense Semantic + Dense Metadata -> Dense
            key_feat = f"X_{split}_semantic"
            X_feat = feature_dict[key_feat]
            X = np.hstack([X_feat, X_meta])

        elif model_name == "metadata_lr":
            # Metadata only
            X = X_meta

        elif model_name == "meta_lr":
            # Meta learner expects predictions from Level 1, which are passed directly
            # as X in the pipeline. This block is just a placeholder or pass-through.
            # Usually, the pipeline constructs the OOF matrix separately.
            pass

        else:
            raise ValueError(f"Unknown model formatting rule for {model_name}")

        # Retrieve Target
        y = None
        if split != "test":
            key_y = f"y_{split}"
            if key_y in feature_dict:
                y = feature_dict[key_y]

        return X, y

    def train_model(
        self, model, X_train, y_train, X_val=None, y_val=None, verbose=True
    ):
        """
        Trains the model. Handles Early Stopping for XGBoost if validation data is provided.

        Args:
            model: The instantiated model object.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional).
            y_val: Validation targets (optional).
            verbose (bool): Whether to print metrics.

        Returns:
            model: The trained model.
        """
        # XGBoost Specific Handling
        if isinstance(model, XGBClassifier):
            fit_params = {}

            # Check if early stopping is configured and validation data is available
            if (
                hasattr(model, "_early_stopping_rounds_custom")
                and X_val is not None
                and y_val is not None
            ):
                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["verbose"] = False  # Suppress XGB internal logs
                model.set_params(
                    early_stopping_rounds=model._early_stopping_rounds_custom
                )
            else:
                model.set_params(early_stopping_rounds=None)

            model.fit(X_train, y_train, **fit_params)

        else:
            # Standard Sklearn Models
            model.fit(X_train, y_train)

        # Validation Metrics
        if verbose and X_val is not None and y_val is not None:
            try:
                y_pred_prob = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_prob)
                print(f"Validation AUC: {auc}")  # Full precision
            except Exception as e:
                print(f"Could not calculate validation metric: {e}")

        return model

    def predict_proba(self, model, X):
        """
        Wrapper for predict_proba to return only the positive class probability.
        """
        return model.predict_proba(X)[:, 1]
