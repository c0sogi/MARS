import lightgbm as lgb
import torch
import torch.nn as nn
import timm
from sklearn.linear_model import Ridge
from library.config import Config


class LGBMRegressorWrapper:
    """
    Wrapper for LightGBM Regressor (Branch A).
    Handles configuration loading and training with early stopping.
    """

    def __init__(self):
        # Create a copy of params to avoid modifying the global Config
        self.params = Config.LGBM_PARAMS.copy()

        # Extract early_stopping_rounds as it is passed via callbacks in newer LightGBM versions
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 100)

        # Initialize the model
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model with Early Stopping.
        """
        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            # Configure callbacks for monitoring and early stopping
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=True
                )
            )
            callbacks.append(lgb.log_evaluation(period=100))

        self.model.fit(
            X_train, y_train, eval_set=eval_set, eval_metric="mae", callbacks=callbacks
        )

    def predict(self, X):
        """
        Generates predictions using the trained model.
        """
        return self.model.predict(X)


class EfficientNet10Ch(nn.Module):
    """
    EfficientNet-B0 modified for 10-channel input (Branch B).
    Uses 'timm' library to handle weight adaptation for non-RGB inputs.
    """

    def __init__(self, model_name=Config.CNN_MODEL_NAME, pretrained=True):
        super(EfficientNet10Ch, self).__init__()

        # Map Config params to timm arguments
        # in_channels -> in_chans
        # dropout -> drop_rate
        in_chans = Config.CNN_PARAMS["in_channels"]
        num_classes = Config.CNN_PARAMS["num_classes"]
        drop_rate = Config.CNN_PARAMS["dropout"]

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (N, 10, H, W)
        Returns:
            torch.Tensor: Output logits of shape (N, 1)
        """
        return self.backbone(x)


class RidgeStacker:
    """
    Meta-Learner using Ridge Regression.
    Allows unconstrained weighting of base model predictions.
    """

    def __init__(self):
        self.model = Ridge(alpha=Config.RIDGE_ALPHA, random_state=Config.SEED)

    def fit(self, X, y):
        """
        Fits the meta-learner on OOF predictions.
        Args:
            X (np.ndarray): Matrix of shape (N_samples, N_models)
            y (np.ndarray): Target values
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts final target values.
        Args:
            X (np.ndarray): Matrix of shape (N_samples, N_models)
        """
        return self.model.predict(X)
