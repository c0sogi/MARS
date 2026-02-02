import torch
import torch.nn as nn
from library.config import Config


class HAPSLoss(nn.Module):
    """
    Computes the composite loss for the Hybrid Anchor-Pairwise Sorting Network.

    Loss = L_anchor + lambda * L_pairwise

    - L_anchor: CrossEntropyLoss predicting which code cell precedes the markdown cell.
    - L_pairwise: BinaryCrossEntropyLoss predicting relative order of markdown pairs
                  within the same anchor bin.
    """

    def __init__(self):
        super(HAPSLoss, self).__init__()

        # Anchor Loss:
        # The target is a class index (0 to N_code).
        # We use ignore_index=-100 to mask out padded markdown cells in the batch.
        self.anchor_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Pairwise Loss:
        # The target is binary (0.0 or 1.0).
        # The model outputs raw logits, so we use BCEWithLogitsLoss for numerical stability.
        self.pairwise_criterion = nn.BCEWithLogitsLoss()

        # Weighting factor from config
        self.pairwise_weight = Config.pairwise_loss_weight

    def forward(self, outputs, batch):
        """
        Computes the loss.

        Args:
            outputs (dict): The output dictionary from HAPSModel.forward().
                - 'anchor_logits': Tensor of shape (Batch, MaxMD, MaxCode + 1)
                - 'pairwise_logits': Tensor of shape (NumPairs,) or None
            batch (dict): The batch dictionary from the dataloader.
                - 'anchor_labels': Tensor of shape (Batch, MaxMD) with class indices.
                - 'pairwise_labels': Tensor of shape (NumPairs,) with 0.0/1.0 targets.

        Returns:
            dict: A dictionary containing:
                - 'loss': The total weighted scalar loss (for backprop).
                - 'anchor_loss': The scalar anchor loss component (detached).
                - 'pairwise_loss': The scalar pairwise loss component (detached).
        """
        anchor_logits = outputs["anchor_logits"]
        anchor_labels = batch["anchor_labels"]

        # --- 1. Compute Anchor Loss ---
        # Reshape logits to (N_samples, N_classes) and labels to (N_samples)
        # N_samples = Batch * MaxMD
        # N_classes = MaxCode + 1

        batch_size, max_md, num_classes = anchor_logits.shape

        flat_logits = anchor_logits.view(-1, num_classes)
        flat_labels = anchor_labels.view(-1)

        anchor_loss = self.anchor_criterion(flat_logits, flat_labels)

        # --- 2. Compute Pairwise Loss ---
        # Pairwise logits/labels might be empty if no valid pairs were found in the batch.

        pairwise_logits = outputs.get("pairwise_logits")
        pairwise_labels = batch.get("pairwise_labels")

        pairwise_loss = torch.tensor(0.0, device=anchor_loss.device)

        if pairwise_logits is not None and pairwise_labels is not None:
            if pairwise_logits.numel() > 0:
                # Ensure labels match device and type
                pairwise_labels = pairwise_labels.to(
                    device=pairwise_logits.device, dtype=pairwise_logits.dtype
                )
                pairwise_loss = self.pairwise_criterion(
                    pairwise_logits, pairwise_labels
                )

        # --- 3. Combine Losses ---
        total_loss = anchor_loss + (self.pairwise_weight * pairwise_loss)

        return {
            "loss": total_loss,
            "anchor_loss": anchor_loss.detach(),
            "pairwise_loss": pairwise_loss.detach(),
        }
