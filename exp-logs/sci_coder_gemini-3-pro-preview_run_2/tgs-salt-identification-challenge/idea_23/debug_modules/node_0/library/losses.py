import torch
import torch.nn as nn
import torch.nn.functional as F


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly using the Lovasz extension.
    Based on 'The Lovasz-Softmax loss: A tractable surrogate for the optimization
    of the intersection-over-union measure in neural networks'.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, labels):
        """
        Computes the Lovasz-Hinge loss.

        Args:
            logits (torch.Tensor): Logits from the model. Shape (B, 1, H, W) or (B, H, W).
            labels (torch.Tensor): Binary ground truth masks. Shape (B, 1, H, W) or (B, H, W).

        Returns:
            torch.Tensor: Scalar loss value averaged over the batch.
        """
        # Squeeze channel dimension if present (B, 1, H, W) -> (B, H, W)
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if labels.dim() == 4:
            labels = labels.squeeze(1)

        batch_size = logits.size(0)
        loss = 0.0

        # Calculate loss per image as per task description
        for i in range(batch_size):
            loss += self._lovasz_hinge_flat(logits[i].view(-1), labels[i].view(-1))

        return loss / batch_size

    def _lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Jaccard loss w.r.t sorted errors.
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union

        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

        return jaccard

    def _lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss for a single flattened image.

        Args:
            logits (torch.Tensor): Flattened logits (N,).
            labels (torch.Tensor): Flattened binary labels (N,).
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.0

        # Signs: 1 if label is 1, -1 if label is 0
        signs = 2.0 * labels.float() - 1.0

        # Hinge Loss errors: relu(1 - logits * signs)
        errors = 1.0 - logits * signs

        # Sort errors in descending order
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]

        # Compute gradients of the Lovasz extension
        grad = self._lovasz_grad(gt_sorted)

        # Dot product of errors (relu) and gradients
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


class CombinedLoss(nn.Module):
    """
    Combined Loss: Sum of Lovasz-Hinge and Binary Cross Entropy.
    This combination stabilizes training (BCE) while optimizing the specific metric (Lovasz).
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, logits, labels):
        """
        Computes the combined loss.

        Args:
            logits (torch.Tensor): Logits from the model.
            labels (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Sum of BCE and Lovasz loss.
        """
        # Ensure labels are float for BCE
        if not labels.is_floating_point():
            labels = labels.float()

        # BCEWithLogitsLoss handles broadcasting, but explicit matching is safer
        # If logits are (B, 1, H, W) and labels are (B, H, W), unsqueeze labels
        if logits.dim() == 4 and labels.dim() == 3:
            labels_bce = labels.unsqueeze(1)
        else:
            labels_bce = labels

        bce_loss = self.bce(logits, labels_bce)
        lovasz_loss = self.lovasz(logits, labels)

        return bce_loss + lovasz_loss
