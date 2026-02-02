import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.utils import Config, ensure_dirs
from library.dataset import load_and_process_taxonomy


def get_class_weights(load_cached_data=True):
    """
    Calculates or loads class weights for the species classification task
    to handle class imbalance.

    Logic:
    1. Checks for cached weights in ./working/idea_4/class_weights.npy
    2. If not found or load_cached_data is False:
       - Loads train metadata and taxonomy mappings.
       - Maps original category_ids to model indices.
       - Computes inverse frequency weights: N / (num_classes * count_c).
       - Saves to cache.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: Class weights of shape (NUM_CLASSES_SPECIES,).
    """
    ensure_dirs()
    cache_path = os.path.join(Config.WORKING_DIR, "class_weights.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32)
        except Exception as e:
            print(f"Failed to load cached class weights: {e}. Recomputing...")

    # Load taxonomy to get mapping from category_id to contiguous index
    maps = load_and_process_taxonomy(load_cached_data=True)
    species_to_idx = {int(k): v for k, v in maps["species_to_idx"].items()}

    # Load training data
    train_df = pd.read_csv(Config.TRAIN_CSV)

    # Map category_id to model index
    # Note: We assume all category_ids in train_df exist in the mapping
    train_indices = train_df["category_id"].map(species_to_idx).values

    # Count frequencies
    num_classes = Config.NUM_CLASSES_SPECIES
    counts = np.bincount(train_indices, minlength=num_classes)

    # Handle classes with 0 samples (though unlikely in this specific dataset split)
    # by setting count to 1 to avoid division by zero, or masking.
    # Given the dataset analysis, min samples is 3, so we are safe.
    counts = np.maximum(counts, 1)

    # Compute weights: w_j = n_samples / (n_classes * n_samples_j)
    total_samples = len(train_indices)
    weights = total_samples / (num_classes * counts)

    # Normalize weights so they sum to num_classes (optional, but keeps scale consistent)
    # weights = weights / weights.sum() * num_classes

    # Save to cache
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32)


class HierarchicalLoss(nn.Module):
    """
    Computes the weighted sum of losses for Species, Genus, and Family predictions.

    Formula:
    L_total = L_species + lambda_genus * L_genus + lambda_family * L_family
    """

    def __init__(self, device, class_weights=None):
        """
        Args:
            device (torch.device): Device to move weights to.
            class_weights (torch.Tensor, optional): Pre-computed class weights for species.
        """
        super(HierarchicalLoss, self).__init__()

        # Species Loss: Weighted Cross Entropy with Label Smoothing
        if class_weights is not None:
            class_weights = class_weights.to(device)

        self.species_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
        )

        # Genus and Family Loss: Standard Cross Entropy
        # We assume these are less imbalanced or the structural regularization is sufficient without explicit weights
        self.genus_loss_fn = nn.CrossEntropyLoss()
        self.family_loss_fn = nn.CrossEntropyLoss()

        # Weights for the multi-task objective
        self.lambda_species = Config.LAMBDA_SPECIES
        self.lambda_genus = Config.LAMBDA_GENUS
        self.lambda_family = Config.LAMBDA_FAMILY

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing logits:
                - 'species': (B, Num_Species)
                - 'genus': (B, Num_Genera)
                - 'family': (B, Num_Families)
            targets (dict): Dictionary containing ground truth labels:
                - 'species': (B,)
                - 'genus': (B,)
                - 'family': (B,)

        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss (torch.Tensor): Scalar loss for backpropagation.
                - loss_dict (dict): Components of the loss for logging.
        """

        l_species = self.species_loss_fn(outputs["species"], targets["species"])
        l_genus = self.genus_loss_fn(outputs["genus"], targets["genus"])
        l_family = self.family_loss_fn(outputs["family"], targets["family"])

        total_loss = (
            self.lambda_species * l_species
            + self.lambda_genus * l_genus
            + self.lambda_family * l_family
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_species": l_species.item(),
            "loss_genus": l_genus.item(),
            "loss_family": l_family.item(),
        }

        return total_loss, loss_dict
