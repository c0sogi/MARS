import os
import random
import numpy as np
import joblib
import pandas as pd
import logging
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logger(name="pipeline"):
    """
    Configures a simple logger that outputs to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if the logger is retrieved multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def save_model(model, model_name, fold=None):
    """
    Saves a model to the configured model directory using joblib.

    Args:
        model: The trained model object.
        model_name (str): Base name for the model.
        fold (int, optional): Fold number. If provided, appended to the filename.
    """
    if fold is not None:
        filename = f"{model_name}_fold_{fold}.joblib"
    else:
        filename = f"{model_name}.joblib"

    path = os.path.join(Config.MODEL_DIR, filename)
    joblib.dump(model, path)
    return path


def load_model(model_name, fold=None):
    """
    Loads a model from the configured model directory using joblib.

    Args:
        model_name (str): Base name for the model.
        fold (int, optional): Fold number.

    Returns:
        The loaded model object.
    """
    if fold is not None:
        filename = f"{model_name}_fold_{fold}.joblib"
    else:
        filename = f"{model_name}.joblib"

    path = os.path.join(Config.MODEL_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    return joblib.load(path)


def compute_metric(y_true, y_pred):
    """
    Computes the ROC AUC score.

    Args:
        y_true: Array-like of ground truth labels.
        y_pred: Array-like of predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    return roc_auc_score(y_true, y_pred)


def print_metric(metric_name, value):
    """
    Prints the metric value with full precision as required.
    """
    print(f"{metric_name}: {value}")


def save_submission(request_ids, predictions, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the required submission format.

    Args:
        request_ids: Array-like of request IDs.
        predictions: Array-like of predicted probabilities.
        filename (str): Name of the output file. Defaults to 'submission.csv'.
    """
    # Ensure predictions are flattened
    if hasattr(predictions, "flatten"):
        predictions = predictions.flatten()
    elif isinstance(predictions, list):
        predictions = np.array(predictions).flatten()

    df = pd.DataFrame({Config.ID_COL: request_ids, Config.TARGET_COL: predictions})

    # Use the directory from Config
    path = os.path.join(Config.SUBMISSION_DIR, filename)

    df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
