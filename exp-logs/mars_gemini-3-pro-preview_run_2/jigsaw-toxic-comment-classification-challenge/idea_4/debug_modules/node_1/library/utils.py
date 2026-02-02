import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
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


def get_device() -> torch.device:
    """
    Returns the PyTorch device (CUDA or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean column-wise ROC AUC.

    Args:
        y_true: Ground truth binary labels (N, num_classes).
        y_pred: Predicted probabilities (N, num_classes).

    Returns:
        Mean ROC AUC score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate AUC for each column
    aucs = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # Handle edge case where a class might not be present in the batch
        if len(np.unique(y_true[:, i])) > 1:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(score)
        else:
            # If only one class is present, ROC AUC is undefined.
            # In context of large validation sets, this shouldn't happen often.
            # We skip or assign 0.5 (random guess) depending on strategy.
            # Here we skip to keep the mean accurate for valid columns.
            pass

    if not aucs:
        return 0.5

    return np.mean(aucs)


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    """

    def __init__(
        self,
        patience=Config.TRAIN_PARAMS["patience"],
        mode="max",
        delta=0.0,
        save_path=None,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the
                quantity monitored has stopped decreasing; in 'max' mode it will stop when
                the quantity monitored has stopped increasing.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the best model.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.val_score_max = -np.Inf

        if self.mode == "min":
            self.check_func = lambda current, best: current < best - self.delta
        else:
            self.check_func = lambda current, best: current > best + self.delta

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif not self.check_func(score, self.best_score):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation score improves."""
        if self.save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), self.save_path)
            else:
                torch.save(model.state_dict(), self.save_path)


# =========================================================================
# Data Caching Utilities (Parquet/Numpy)
# =========================================================================


def save_npy(data: np.ndarray, path: str) -> None:
    """
    Saves a numpy array to a file, ensuring the directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)


def load_npy(path: str) -> np.ndarray:
    """
    Loads a numpy array from a file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Saves a pandas DataFrame to a parquet file, ensuring the directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from a parquet file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)


def save_submission(
    ids: list, predictions: np.ndarray, output_path: str = Config.SUBMISSION_PATH
) -> None:
    """
    Saves the predictions to a CSV file in the required submission format.

    Args:
        ids: List of ID strings.
        predictions: Numpy array of probabilities (N, 6).
        output_path: Path to save the submission file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df = pd.DataFrame(predictions, columns=Config.LABEL_COLS)
    submission_df.insert(0, "id", ids)

    submission_df.to_csv(output_path, index=False)
