import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Lovasz-Hinge Loss Implementation
# -------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error
    See Alg. 1 in https://arxiv.org/abs/1705.08790
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth labels (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = 0
        for i in range(len(logits)):
            # Flatten
            logit_flat = logits[i].view(-1)
            label_flat = labels[i].view(-1)
            if ignore is not None:
                valid = label_flat != ignore
                logit_flat = logit_flat[valid]
                label_flat = label_flat[valid]
            loss += lovasz_hinge_flat(logit_flat, label_flat)
        return loss / len(logits)
    else:
        # Flatten batch
        logit_flat = logits.view(-1)
        label_flat = labels.view(-1)
        if ignore is not None:
            valid = label_flat != ignore
            logit_flat = logit_flat[valid]
            label_flat = label_flat[valid]
        return lovasz_hinge_flat(logit_flat, label_flat)


class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, labels):
        return lovasz_hinge(
            logits, labels, per_image=self.per_image, ignore=self.ignore
        )


# -------------------------------------------------------------------------
# Composite Losses
# -------------------------------------------------------------------------


class TeacherLoss(nn.Module):
    """
    Loss function for the Specialist Teacher.
    Combination of Lovasz-Hinge and BCE.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, 1, H, W)
            labels: (B, 1, H, W)
        """
        # Ensure shapes match for Lovasz (which expects B, H, W or flattened)
        # If input is (B, 1, H, W), we squeeze.
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits_sq = logits.squeeze(1)
        else:
            logits_sq = logits

        if labels.dim() == 4 and labels.shape[1] == 1:
            labels_sq = labels.squeeze(1)
        else:
            labels_sq = labels

        bce_loss = self.bce(logits, labels.float())
        lovasz_loss = self.lovasz(logits_sq, labels_sq)

        return self.bce_weight * bce_loss + self.lovasz_weight * lovasz_loss


class StudentLoss(nn.Module):
    """
    Loss function for the Generalist Student (Multi-Task).
    Handles both Supervised (Labeled) and Unsupervised (Soft Pseudo-Label) batches.
    """

    def __init__(self, depth_weight=1.0):
        super().__init__()
        self.depth_weight = depth_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred_mask, pred_depth, target_mask, target_depth=None):
        """
        Args:
            pred_mask: (B, 1, H, W) Logits
            pred_depth: (B, 1) Scalar depth predictions
            target_mask: (B, 1, H, W) Binary masks (Supervised) or Soft Probs (Unsupervised)
            target_depth: (B, 1) Scalar depth targets. If None, assumes Unsupervised mode.
        """
        # Check mode based on presence of target_depth
        if target_depth is not None:
            # --- Supervised Mode ---
            # Loss = Lovasz + BCE + MSE(Depth)

            # Prepare for Lovasz (Squeeze channel)
            if pred_mask.dim() == 4 and pred_mask.shape[1] == 1:
                pred_mask_sq = pred_mask.squeeze(1)
            else:
                pred_mask_sq = pred_mask

            if target_mask.dim() == 4 and target_mask.shape[1] == 1:
                target_mask_sq = target_mask.squeeze(1)
            else:
                target_mask_sq = target_mask

            bce_loss = self.bce(pred_mask, target_mask.float())
            lovasz_loss = self.lovasz(pred_mask_sq, target_mask_sq)

            # Depth Loss
            depth_loss = self.mse(pred_depth, target_depth)

            return bce_loss + lovasz_loss + self.depth_weight * depth_loss

        else:
            # --- Unsupervised Mode ---
            # Loss = BCE (vs Soft Targets)
            # Lovasz is incompatible with soft targets.
            # Depth loss is ignored (no ground truth).

            # BCEWithLogitsLoss supports soft targets (probabilities) in target
            bce_loss = self.bce(pred_mask, target_mask)

            return bce_loss
