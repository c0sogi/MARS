import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    Ensures deterministic behavior for CuDNN backends.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FGM:
    """
    Fast Gradient Method (FGM) for Adversarial Training.
    Perturbs embeddings based on gradients to improve model robustness.
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        """
        Applies perturbation to the embeddings.

        Args:
            epsilon (float): Magnitude of the perturbation.
            emb_name (str): Substring to identify embedding parameters.
        """
        for name, param in self.model.named_parameters():
            # Apply only to parameters that require gradients and match the embedding name
            if param.requires_grad and emb_name in name and param.grad is not None:
                # Save original data
                self.backup[name] = param.data.clone()

                # Calculate norm of the gradient
                norm = torch.norm(param.grad)

                # Apply perturbation if norm is valid (avoid division by zero)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        """
        Restores the original embeddings from backup.

        Args:
            emb_name (str): Substring to identify embedding parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


def get_optimizer_grouped_parameters(model, config):
    """
    Configures Differential Learning Rates (DLR) with explicit grouping.

    Strategy:
    - Backbone: Base LR
    - Heads: Base LR * 5 (Cite {solution_lesson_node_00031})
    - Weight Decay: Applied to all parameters (Cite {solution_lesson_node_00036})

    This replaces aggressive LLRD with a simpler, more robust split (Cite {solution_lesson_node_00044})
    and uses explicit module references (Cite {solution_lesson_node_00033}).
    """

    # Explicitly group parameters using module references
    # Backbone parameters
    backbone_params = model.roberta.parameters()

    # Task-specific Head parameters
    head_params = list(model.qa_outputs.parameters()) + list(
        model.relevance_classifier.parameters()
    )

    optimizer_grouped_parameters = [
        {
            "params": backbone_params,
            "lr": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
        },
        {
            "params": head_params,
            "lr": config.LEARNING_RATE * 5,  # Higher LR for initialized heads
            "weight_decay": config.WEIGHT_DECAY,
        },
    ]

    return optimizer_grouped_parameters
