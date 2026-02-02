import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Crucially, this function explicitly disables strict CuDNN determinism
    (cudnn.deterministic = False) and enables CuDNN benchmarking. This aligns
    with the strategy to maximize kernel performance for the deep architecture,
    accepting bit-level non-determinism for training speed.

    Args:
        seed (int): The random seed to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Strategy Requirement: Disable strict determinism for performance (Lesson 00070)
        torch.backends.cudnn.deterministic = False
        # Enable benchmark to let CuDNN select the fastest kernels
        torch.backends.cudnn.benchmark = True


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes the multi-class classification accuracy.

    Args:
        outputs (torch.Tensor): Model predictions (logits or probabilities)
                                of shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels of shape (batch_size).

    Returns:
        float: The accuracy score (0.0 to 1.0).
    """
    with torch.no_grad():
        # Get the index of the max log-probability
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)

        if total == 0:
            return 0.0

        return correct / total


def save_submission(
    predictions: np.ndarray,
    test_ids: np.ndarray,
    output_path: str = Config.SUBMISSION_PATH,
):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        predictions (np.ndarray): Array of predicted class labels (integers).
        test_ids (np.ndarray): Array of corresponding test set IDs.
        output_path (str): File path where the submission CSV will be saved.
                           Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create the DataFrame matching the sample submission format
    submission_df = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: predictions}
    )

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
