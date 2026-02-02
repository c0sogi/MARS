import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.data import get_taxonomy_mapping
from library.utils import get_logger

logger = get_logger("loss")


class TaxonomicFocalLoss(nn.Module):
    """
    Loss function that combines Focal Loss with Taxonomic Label Smoothing.

    Targets are smoothed such that probability mass epsilon is distributed
    among sibling species (same Genus) instead of uniformly across all classes.
    """

    def __init__(
        self, gamma=Config.FOCAL_LOSS_GAMMA, epsilon=Config.LABEL_SMOOTHING_EPS
    ):
        super(TaxonomicFocalLoss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_classes = Config.NUM_CLASSES

        logger.info(f"Initializing TaxonomicFocalLoss (gamma={gamma}, eps={epsilon})")

        # Load taxonomy mapping: category_id -> genus_id
        # We assume category_ids are 0..N-1 and cover all classes
        df = get_taxonomy_mapping(load_cached_data=True)

        # Validation: Ensure we have the expected number of classes
        if len(df) != self.num_classes:
            logger.warning(
                f"Taxonomy mapping has {len(df)} classes, but Config.NUM_CLASSES is {self.num_classes}. "
                "This might indicate a mismatch or missing categories."
            )

        # Sort by category_id to ensure index alignment with model logits
        df = df.sort_values("category_id").reset_index(drop=True)

        # Extract genus_ids as a tensor aligned with class indices 0..N-1
        # shape: [NUM_CLASSES]
        genus_ids_np = df["genus_id"].values
        genus_ids = torch.tensor(genus_ids_np, dtype=torch.long)

        # Determine number of genera
        num_genera = genus_ids.max().item() + 1
        logger.info(f"Found {num_genera} unique genera across {len(df)} species.")

        # Create Genus-Species Mask: [Num_Genera, Num_Classes]
        # M[g, c] = 1.0 if species c is in genus g, else 0.0
        # We use one_hot on the genus_ids vector and transpose it
        # genus_ids is [C], one_hot is [C, G], transpose is [G, C]
        genus_species_mask = F.one_hot(genus_ids, num_classes=num_genera).float().T

        # Compute counts per genus: [G, 1]
        genus_counts = genus_species_mask.sum(dim=1, keepdim=True)

        # Register buffers to ensure they move to GPU with the model
        self.register_buffer("class_to_genus", genus_ids)
        self.register_buffer("genus_species_mask", genus_species_mask)
        self.register_buffer("genus_counts", genus_counts)

    def forward(self, logits, labels):
        """
        Args:
            logits: (Batch, Num_Classes) - Raw scores from ArcFace head
            labels: (Batch) - Ground truth class indices
        """
        # 1. Compute Probabilities
        # ArcFace logits are cosine similarities * scale.
        # We apply Softmax to get probabilities for Focal Loss calculation.
        probs = F.softmax(logits, dim=1)

        # 2. Construct Soft Targets based on Taxonomy
        # Get genus IDs for the current batch targets: [B]
        batch_genus_ids = self.class_to_genus[labels]

        # Retrieve sibling masks for the batch: [B, Num_Classes]
        # This selects the row from genus_species_mask corresponding to each target's genus
        # Result: 1.0 where class is in same genus as target, 0.0 otherwise
        batch_sibling_mask = F.embedding(batch_genus_ids, self.genus_species_mask)

        # Retrieve genus counts for the batch: [B, 1]
        batch_counts = F.embedding(batch_genus_ids, self.genus_counts)

        # Calculate smoothing value per sibling
        # If count > 1: val = epsilon / (count - 1)
        # If count == 1: val = 0 (handled by masking or explicit logic)
        # We clamp denominator to avoid division by zero for singletons
        denom = torch.clamp(batch_counts - 1, min=1.0)
        smooth_val = self.epsilon / denom

        # Apply smoothing value to all siblings (including self for now)
        targets = batch_sibling_mask * smooth_val

        # 3. Correct the Target Class Probability
        # We want the target class to have probability:
        #   1.0 if singleton (no siblings to smooth to)
        #   1.0 - epsilon if siblings exist

        # Create one-hot for true labels
        true_one_hot = torch.zeros_like(logits)
        true_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Remove the smoothing value from the true class position
        # (targets currently has smooth_val at the true class index)
        targets = targets * (1.0 - true_one_hot)

        # Determine probability for the true class
        is_singleton = (batch_counts == 1).float()
        self_prob = is_singleton * 1.0 + (1.0 - is_singleton) * (1.0 - self.epsilon)

        # Add true class probability to targets
        targets = targets + (true_one_hot * self_prob)

        # 4. Compute Focal Loss with Soft Targets
        # Formula: L = - sum( target * (1 - p)^gamma * log(p) )

        # Compute Focal Weight: (1 - p)^gamma
        # Note: We use the predicted probability of the specific class, not just the target class
        focal_weight = (1.0 - probs).pow(self.gamma)

        # Log Probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Element-wise loss terms
        loss_terms = targets * focal_weight * log_probs

        # Sum over classes, mean over batch
        loss = -loss_terms.sum(dim=1).mean()

        return loss
