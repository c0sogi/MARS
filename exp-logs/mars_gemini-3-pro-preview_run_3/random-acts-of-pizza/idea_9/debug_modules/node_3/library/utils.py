import os
import random
import numpy as np
import pandas as pd
import joblib
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(model, name):
    """
    Saves a trained model to the working directory using joblib.

    Args:
        model: The model object to save.
        name (str): The filename for the model (e.g., 'lexical_rf.joblib').
    """
    file_path = os.path.join(Config.WORKING_DIR, name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)


def load_model(name):
    """
    Loads a trained model from the working directory.

    Args:
        name (str): The filename of the model to load.

    Returns:
        The loaded model object.
    """
    file_path = os.path.join(Config.WORKING_DIR, name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at {file_path}")
    return joblib.load(file_path)


def save_submission(request_ids, probabilities):
    """
    Generates and saves the submission CSV file in the required format.

    Args:
        request_ids (array-like): Sequence of request IDs.
        probabilities (array-like): Sequence of predicted probabilities for the positive class.
    """
    submission_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": probabilities}
    )

    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
