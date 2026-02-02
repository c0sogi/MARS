import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RobustAUC:
    """
    Accumulates predictions and targets to calculate the macro-averaged AUC-ROC.
    Explicitly handles cases where specific classes are absent or constant in the
    validation set by skipping them in the average calculation.
    """

    def __init__(self):
        self.y_preds = []
        self.y_trues = []

    def update(self, y_pred, y_true):
        """
        Add a batch of predictions and targets.

        Args:
            y_pred: Predicted probabilities (batch_size, num_classes).
                    Can be numpy array or torch Tensor.
            y_true: Ground truth labels (batch_size, num_classes).
                    Can be numpy array or torch Tensor.
        """
        # Detach and move to cpu if necessary
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        self.y_preds.append(y_pred)
        self.y_trues.append(y_true)

    def compute(self):
        """
        Compute the macro-averaged AUC across all accumulated batches.

        Returns:
            float: The mean AUC score. Returns 0.0 if no valid classes are found.
        """
        if not self.y_preds:
            return 0.0

        y_pred = np.vstack(self.y_preds)
        y_true = np.vstack(self.y_trues)

        # Ensure shapes match
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
            )

        num_classes = y_true.shape[1]
        auc_scores = []

        for i in range(num_classes):
            # Check if class is present in ground truth (must have at least two unique values)
            # roc_auc_score requires both positive and negative samples.
            if len(np.unique(y_true[:, i])) > 1:
                try:
                    auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                    auc_scores.append(auc)
                except ValueError:
                    # Fallback for edge cases where sklearn might still raise an error
                    continue

        if not auc_scores:
            return 0.0

        return float(np.mean(auc_scores))


def average_checkpoints(checkpoint_paths, output_path):
    """
    Averages the state dictionaries of the provided checkpoints
    and saves the result to output_path.

    Args:
        checkpoint_paths (list): List of file paths to .pth checkpoints.
        output_path (str): Destination path for the averaged checkpoint.
    """
    if not checkpoint_paths:
        print("No checkpoints to average.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load the first checkpoint to initialize the average
    # Map to CPU to avoid GPU OOM during averaging
    avg_state_dict = torch.load(checkpoint_paths[0], map_location="cpu")

    # Handle case where checkpoint is a dict containing 'model_state_dict'
    if "model_state_dict" in avg_state_dict:
        avg_state_dict = avg_state_dict["model_state_dict"]

    # Convert parameters to float for precise averaging
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key].float()

    # Sum subsequent checkpoints
    for path in checkpoint_paths[1:]:
        state_dict = torch.load(path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        for key in avg_state_dict:
            # Ensure keys match; assuming identical architectures
            if key in state_dict:
                avg_state_dict[key] += state_dict[key].float()

    # Divide by number of checkpoints
    num_checkpoints = len(checkpoint_paths)
    for key in avg_state_dict:
        avg_state_dict[key] /= num_checkpoints

    # Save the averaged state dict
    torch.save(avg_state_dict, output_path)
    print(f"Averaged {num_checkpoints} checkpoints and saved to {output_path}")
