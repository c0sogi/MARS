import os
import random
import numpy as np
import torch


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


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard score between two strings.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity score.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denominator = len(a) + len(b) - len(c)

    if denominator == 0:
        return 0.0

    return float(len(c)) / denominator


class FGM:
    """
    Fast Gradient Method (FGM) for Adversarial Training.

    This class manages the perturbation of input embeddings during training to
    improve model robustness.
    """

    def __init__(self, model):
        """
        Initialize the FGM attacker.

        Args:
            model (torch.nn.Module): The model to attack.
        """
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        """
        Perturbs the embeddings based on the gradient to generate adversarial examples.

        Args:
            epsilon (float): The scaling factor for the perturbation.
            emb_name (str): The substring to identify embedding layers in named_parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        """
        Restores the original embeddings from backup.

        Args:
            emb_name (str): The substring to identify embedding layers in named_parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}
