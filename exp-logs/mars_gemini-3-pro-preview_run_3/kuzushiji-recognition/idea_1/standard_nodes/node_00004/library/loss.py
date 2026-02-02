import torch
import torch.nn as nn
import torch.nn.functional as F


def _transpose_and_gather_feat(feat, ind):
    """
    Transposes the feature map and gathers features at specific indices.

    Args:
        feat (torch.Tensor): Feature map (B, C, H, W).
        ind (torch.Tensor): Indices (B, K).

    Returns:
        torch.Tensor: Gathered features (B, K, C).
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
    feat = feat.view(feat.size(0), -1, feat.size(3))  # (B, H*W, C)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))  # (B, K, C)
    feat = feat.gather(1, ind)  # (B, K, C)
    return feat


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss for heatmap regression (CornerNet/CenterNet variant).
    Penalizes background pixels but reduces penalty for pixels near ground truth centers.
    """

    def __init__(self):
        super(ModifiedFocalLoss, self).__init__()

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Predicted heatmap (B, C, H, W), values in [0, 1].
            gt (torch.Tensor): Ground truth heatmap (B, C, H, W), values in [0, 1].
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        # Clamp for numerical stability
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss


class RegL1Loss(nn.Module):
    """
    L1 Loss for regression tasks (offsets) masked by object existence.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred (torch.Tensor): Predicted values (B, K, C).
            target (torch.Tensor): Target values (B, K, C).
            mask (torch.Tensor): Validity mask (B, K).
        """
        expand_mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction="sum")

        # Normalize by number of objects
        num_objs = mask.float().sum() + 1e-4
        loss = loss / num_objs
        return loss


class SparseCenterNetLoss(nn.Module):
    """
    Composite loss function for Sparse-Query CenterNet.
    Computes Heatmap Loss, Offset Regression Loss, and Sparse Classification Loss.
    """

    def __init__(self, classifier, lambda_hm=1.0, lambda_reg=1.0, lambda_cls=1.0):
        """
        Args:
            classifier (nn.Module): The MLP classifier from the model to compute class logits.
            lambda_hm (float): Weight for heatmap loss.
            lambda_reg (float): Weight for regression offset loss.
            lambda_cls (float): Weight for classification loss.
        """
        super(SparseCenterNetLoss, self).__init__()
        self.classifier = classifier
        self.lambda_hm = lambda_hm
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls

        self.hm_loss = ModifiedFocalLoss()
        self.reg_loss = RegL1Loss()
        self.cls_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, outputs, batch):
        """
        Args:
            outputs (tuple): (hm, reg, emb) from model forward pass.
            batch (dict): Batch dictionary containing targets.

        Returns:
            tuple: (total_loss, stats_dictionary)
        """
        pred_hm, pred_reg, pred_emb = outputs

        gt_hm = batch["hm"]
        gt_reg = batch["reg"]
        gt_ind = batch["ind"]
        gt_cls_ids = batch["cls_ids"]
        reg_mask = batch["reg_mask"]

        # 1. Heatmap Loss
        # Apply sigmoid to convert logits to probabilities
        pred_hm = torch.sigmoid(pred_hm)
        loss_hm = self.hm_loss(pred_hm, gt_hm)

        # 2. Regression (Offset) Loss
        # Gather predicted offsets at GT locations
        pred_reg_gathered = _transpose_and_gather_feat(pred_reg, gt_ind)
        loss_reg = self.reg_loss(pred_reg_gathered, gt_reg, reg_mask)

        # 3. Classification Loss
        # Gather embeddings at GT locations: (B, K, EmbDim)
        pred_emb_gathered = _transpose_and_gather_feat(pred_emb, gt_ind)

        # Pass gathered embeddings through the classifier MLP: (B, K, NumClasses)
        cls_logits = self.classifier(pred_emb_gathered)

        # Flatten for CrossEntropyLoss
        B, K, C = cls_logits.shape
        cls_logits_flat = cls_logits.view(-1, C)
        gt_cls_ids_flat = gt_cls_ids.view(-1)
        reg_mask_flat = reg_mask.view(-1).float()

        # Compute CE loss and mask out invalid objects (padding)
        loss_cls_raw = self.cls_loss(cls_logits_flat, gt_cls_ids_flat)
        loss_cls = (loss_cls_raw * reg_mask_flat).sum() / (reg_mask_flat.sum() + 1e-4)

        # Total Loss
        loss = (
            self.lambda_hm * loss_hm
            + self.lambda_reg * loss_reg
            + self.lambda_cls * loss_cls
        )

        stats = {
            "loss": loss.item(),
            "loss_hm": loss_hm.item(),
            "loss_reg": loss_reg.item(),
            "loss_cls": loss_cls.item(),
        }

        return loss, stats
