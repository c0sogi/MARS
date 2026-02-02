import torch
import torch.nn as nn
import torch.nn.functional as F


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) surrogate.

    Reference: https://github.com/bermanmaxim/LovaszSoftmax
    """

    def __init__(self, per_image=True, ignore_index=None):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) logits from the model.
            targets: (B, 1, H, W) or (B, H, W) binary ground truth masks (0 or 1).

        Returns:
            Scalar loss value.
        """
        # Squeeze channel dim if present
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        # Ensure float32 for stability
        logits = logits.float()
        targets = targets.float()

        if self.per_image:
            loss = 0.0
            batch_size = logits.size(0)
            for i in range(batch_size):
                loss += self._lovasz_hinge_flat(logits[i].view(-1), targets[i].view(-1))
            return loss / batch_size
        else:
            return self._lovasz_hinge_flat(logits.view(-1), targets.view(-1))

    def _lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss on flattened tensors.

        Args:
            logits: [P] Float, logits at each pixel.
            labels: [P] Float, binary ground truth masks (0 or 1).
        """
        if self.ignore_index is not None:
            valid = labels != self.ignore_index
            logits = logits[valid]
            labels = labels[valid]

        if len(labels) == 0:
            # If all pixels are ignored, return 0 loss
            return logits.sum() * 0.0

        signs = 2.0 * labels - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)

        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)

        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def _lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors.

        Args:
            gt_sorted: [P] labels sorted by error.
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()

        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union

        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

        return jaccard


class StableBCELoss(nn.Module):
    """
    Stable Binary Cross Entropy with Logits.
    Suitable for both binary targets (0/1) and soft targets (0.0-1.0).
    """

    def __init__(self):
        super(StableBCELoss, self).__init__()
        # BCEWithLogitsLoss combines Sigmoid and BCE for numerical stability
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) logits.
            targets: (B, 1, H, W) or (B, H, W) targets.
        """
        # Ensure float32
        logits = logits.float()
        targets = targets.float()

        return self.bce(logits, targets)


class DepthMSELoss(nn.Module):
    """
    Mean Squared Error Loss for the auxiliary depth regression head.
    """

    def __init__(self):
        super(DepthMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_depth, true_depth):
        """
        Args:
            pred_depth: (B, 1) or (B,) predicted depth (normalized).
            true_depth: (B, 1) or (B,) true depth (normalized).
        """
        # Ensure shapes match and are float32
        pred_depth = pred_depth.view(-1).float()
        true_depth = true_depth.view(-1).float()

        return self.mse(pred_depth, true_depth)


class TeacherComboLoss(nn.Module):
    """
    Composite loss for Stage 1 Teacher Training.
    Sum of Lovasz-Hinge and BCE.
    """

    def __init__(self):
        super(TeacherComboLoss, self).__init__()
        self.lovasz = LovaszHingeLoss(per_image=True)
        self.bce = StableBCELoss()

    def forward(self, logits, targets):
        # Calculate individual losses
        loss_lovasz = self.lovasz(logits, targets)
        loss_bce = self.bce(logits, targets)

        # Sum them up
        return loss_lovasz + loss_bce
