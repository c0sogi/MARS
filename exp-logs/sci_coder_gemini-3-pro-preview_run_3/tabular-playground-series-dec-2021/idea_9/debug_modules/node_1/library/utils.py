import os
import random
import copy
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelCheckpoint:
    """
    Handles saving the best model during training.
    Crucially uses copy.deepcopy to store the state_dict in memory to avoid
    issues with optimizer updates mutating the reference before saving.
    """

    def __init__(self, mode="max"):
        """
        Args:
            mode (str): One of 'min' or 'max'.
                        'min' for metrics like loss where lower is better.
                        'max' for metrics like accuracy where higher is better.
        """
        if mode not in ["min", "max"]:
            raise ValueError("Mode must be 'min' or 'max'")

        self.mode = mode
        if self.mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = float("-inf")

        self.best_state = None

    def step(self, score, model):
        """
        Updates the best score and caches the model state if the current score is better.

        Args:
            score (float): The current metric value.
            model (torch.nn.Module): The model to checkpoint.

        Returns:
            bool: True if the model was improved and checkpointed, False otherwise.
        """
        improved = False
        if self.mode == "min":
            if score < self.best_score:
                self.best_score = score
                improved = True
        else:
            if score > self.best_score:
                self.best_score = score
                improved = True

        if improved:
            # Critical: deepcopy to prevent reference mutation by optimizer
            # This ensures we hold the exact weights at the time of the best score
            self.best_state = copy.deepcopy(model.state_dict())

        return improved

    def save_best(self, path):
        """
        Saves the best cached state_dict to the specified file path.

        Args:
            path (str): The file path to save the model weights to.
        """
        if self.best_state is not None:
            torch.save(self.best_state, path)

    def load_best(self, model):
        """
        Loads the best cached state_dict into the provided model instance.

        Args:
            model (torch.nn.Module): The model to load weights into.

        Returns:
            model: The model with loaded weights.
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model
