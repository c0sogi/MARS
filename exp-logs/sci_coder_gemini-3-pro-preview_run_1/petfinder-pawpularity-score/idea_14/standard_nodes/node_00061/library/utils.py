import os
import random
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rmse_score(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE).
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def logit_transform(y):
    """
    Applies Logit transformation to the target variable to map bounded [1, 100] range
    to unbounded real numbers (-inf, inf).

    Formula:
    y' = clip(y / SCALE, MIN, MAX)
    z = log(y' / (1 - y'))
    """
    y = np.array(y, dtype=np.float64)

    # Scale to [0, 1]
    y_scaled = y / Config.TARGET_SCALE

    # Clip to avoid infinity in logit calculation
    y_clipped = np.clip(y_scaled, Config.TARGET_MIN, Config.TARGET_MAX)

    # Logit transform: log(p / (1-p))
    z = np.log(y_clipped / (1.0 - y_clipped))
    return z


def inverse_logit_transform(z):
    """
    Applies Inverse Logit (Sigmoid) transformation to map unbounded predictions
    back to the original [1, 100] range.

    Formula:
    y' = 1 / (1 + exp(-z))
    y = y' * SCALE
    """
    # Sigmoid function
    y_scaled = 1.0 / (1.0 + np.exp(-z))

    # Scale back to original range
    y = y_scaled * Config.TARGET_SCALE
    return y


def save_to_cache(data, path):
    """
    Saves a numpy array to the specified path.
    Automatically creates the parent directory if it does not exist.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(path, data)


def load_from_cache(path):
    """
    Loads a numpy array from the specified path if it exists.
    Returns None if the file does not exist or cannot be loaded.
    """
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            return None
    return None
